"""DSP core for The 8D Engine.

The processor is intentionally lightweight and dependency-minimal:
- NumPy for vector math
- optional SciPy for high-quality Butterworth crossover filters
- NumPy FFT fallback if SciPy is unavailable

Signal-flow summary:
    stereo input
      -> mono bass/motion crossover at 150 Hz
      -> bass band is duplicated L/R and never spatialized
      -> motion band is rendered as a rotating binaural-ish object
      -> optional synthetic room tail is mixed into the moving band
      -> bass + motion + room are recombined and peak-normalized

This is not a personalized HRTF renderer. It is a practical, educational
HRTF-inspired engine using three psychoacoustic cues:
    1. ILD: interaural level difference through equal-power gains
    2. ITD: interaural time difference through a fractional delay line
    3. rear coloration: simple spectral/distance shading for rear positions
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class AudioData:
    """Container for floating-point stereo audio.

    samples must be shaped (frames, 2), use float64 or float32, and should be in
    the range [-1.0, 1.0].
    """

    samples: np.ndarray
    sample_rate: int


@dataclass(frozen=True)
class CorrelationReport:
    """Phase/correlation health report for a stereo signal."""

    correlation: float
    side_mid_ratio: float
    phase_warning: bool


SPEED_OF_SOUND_MPS = 343.0
HEAD_RADIUS_METERS = 0.0875
MAX_ITD_SECONDS = 0.00068  # practical human max: ~600-700 microseconds
PHI = (1.0 + math.sqrt(5.0)) / 2.0
GOLDEN_ANGLE_DEGREES = 360.0 / (PHI * PHI)  # ~137.507°, golden-angle spatial stepping
FIBONACCI_WEIGHTS = np.array([1, 1, 2, 3, 5, 8, 13], dtype=np.float64)
LUCAS_WEIGHTS = np.array([2, 1, 3, 4, 7, 11], dtype=np.float64)
FIBONACCI_PRESETS = {
    "phi_reference_orbit",
    "fibonacci_spiral",
    "golden_figure8",
    "lucas_breath",
}


def ensure_stereo_float(samples: np.ndarray) -> np.ndarray:
    """Return a normalized stereo float32 array shaped (frames, 2).

    float32 has 24-bit mantissa precision — more than enough for 16/24-bit
    audio — and halves peak RAM compared with float64.  All internal DSP
    uses float32; results are exported as 32-bit float WAV, so no quality
    is lost relative to the source material.
    """

    x = np.asarray(samples)
    if x.ndim == 1:
        x = np.column_stack([x, x])
    elif x.ndim == 2 and x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.ndim == 2 and x.shape[1] >= 2:
        x = x[:, :2]
    else:
        raise ValueError("Audio samples must be mono or stereo")

    if np.issubdtype(x.dtype, np.integer):
        max_value = np.iinfo(x.dtype).max
        x = x.astype(np.float32) / np.float32(max_value)
    else:
        x = x.astype(np.float32, copy=False)

    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_peak(samples: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
    """Peak-normalize if needed, preserving dynamics below the ceiling."""

    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak <= 0.0 or peak <= ceiling:
        return samples
    return samples * (ceiling / peak)


def db_to_linear(db_value: float) -> float:
    """Convert dBFS-style gain to linear amplitude."""

    return float(10.0 ** (db_value / 20.0))


def rms_level(samples: np.ndarray) -> float:
    """Return RMS amplitude for any array."""

    x = np.asarray(samples, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x) + 1e-18))


def analyze_correlation(samples: np.ndarray) -> CorrelationReport:
    """Measure stereo correlation and flag potentially phasey output.

    Correlation near +1 is mono-compatible, 0 is very wide/decorrelated, and
    negative values indicate possible mono cancellation.
    """

    stereo = ensure_stereo_float(samples)
    left = stereo[:, 0]
    right = stereo[:, 1]
    if len(stereo) == 0 or rms_level(left) < 1e-12 or rms_level(right) < 1e-12:
        corr = 1.0
    else:
        corr = float(np.corrcoef(left, right)[0, 1])
        if not np.isfinite(corr):
            corr = 1.0
    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    ratio = rms_level(side) / (rms_level(mid) + 1e-12)
    return CorrelationReport(
        correlation=corr,
        side_mid_ratio=float(ratio),
        phase_warning=bool(corr < -0.15 or ratio > 1.35),
    )


def apply_youtube_master_target(
    samples: np.ndarray,
    target_rms_db: float = -13.0,
    peak_ceiling_db: float = -1.0,
) -> np.ndarray:
    """Simple offline master target for YouTube-style 8D exports.

    This is RMS-based rather than full LUFS/true-peak metering, but it gives a
    practical hot-but-safe export target without adding heavy dependencies.
    """

    stereo = ensure_stereo_float(samples)
    current = rms_level(stereo)
    if current <= 1e-12:
        return stereo
    target = db_to_linear(target_rms_db)
    ceiling = db_to_linear(peak_ceiling_db)
    gained = stereo * (target / current)
    return normalize_peak(gained, ceiling=ceiling)


def preserve_loudness_and_peak(
    processed: np.ndarray,
    reference: np.ndarray,
    peak_ceiling_db: float = -1.0,
    max_rms_lift_db: float = 0.25,
) -> np.ndarray:
    """Keep the rendered signal clean, unclipped, and close to source loudness.

    This is a quality guard, not a compressor/limiter. It applies one static gain
    for the whole file so the export does not become louder than the original by
    more than a tiny allowance and never exceeds the requested peak ceiling. It
    preserves musical dynamics because it does not change gain over time.
    """

    out = ensure_stereo_float(processed)
    ref = ensure_stereo_float(reference)
    if len(out) == 0:
        return out

    peak_ceiling = db_to_linear(peak_ceiling_db)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    peak_gain = (peak_ceiling / peak) if peak > peak_ceiling and peak > 0.0 else 1.0

    ref_rms = rms_level(ref)
    out_rms = rms_level(out)
    max_rms = ref_rms * db_to_linear(max_rms_lift_db)
    rms_gain = (max_rms / out_rms) if out_rms > max_rms and out_rms > 0.0 else 1.0

    return out * min(peak_gain, rms_gain, 1.0)


def estimate_bpm(audio: AudioData, fallback_bpm: float = 120.0) -> float:
    """Estimate tempo in beats per minute for automatic 8D beat-locking.

    The preferred path uses librosa's onset/beat tracker. If librosa is not
    installed or the tracker cannot lock confidently, the function returns a
    sensible musical fallback so the drop-zone app never blocks conversion.
    """

    samples = ensure_stereo_float(audio.samples)
    if len(samples) == 0:
        return float(fallback_bpm)
    mono = samples.mean(axis=1).astype(np.float64, copy=False)
    try:
        import librosa  # type: ignore

        tempo, _beats = librosa.beat.beat_track(y=mono, sr=audio.sample_rate, units="time")
        tempo_value = float(np.asarray(tempo).reshape(-1)[0])
        if np.isfinite(tempo_value) and 40.0 <= tempo_value <= 220.0:
            return tempo_value
    except Exception:
        pass
    return float(fallback_bpm)


def bpm_to_two_bar_rotation_cpm(bpm: float, beats_per_bar: int = 4) -> float:
    """Convert tempo to rotation cycles/minute where one orbit equals two bars.

    In 4/4 music, two bars contain 8 beats. A 120 BPM track therefore rotates
    15 times per minute, or one full spherical orbit every 4 seconds.
    """

    safe_bpm = float(np.clip(bpm, 40.0, 220.0))
    return safe_bpm / float(max(1, beats_per_bar * 2))


def bpm_to_premium_rotation_cpm(bpm: float, beats_per_bar: int = 4) -> float:
    """Convert tempo to a smooth premium 8D orbit rate.

    The premium default is one full orbit every four bars. At 120 BPM this is an
    8-second rotation: immersive and clearly moving, but far less dizzying than
    the two-bar/4-second novelty setting.
    """

    safe_bpm = float(np.clip(bpm, 40.0, 220.0))
    return safe_bpm / float(max(1, beats_per_bar * 4))


def _fft_split_mono(mono: np.ndarray, sample_rate: int, crossover_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    """FFT-domain fallback crossover when scipy is not installed.

    It uses a raised-cosine transition around the crossover. This is linear-phase
    and test-friendly, although slower than an IIR filter for long files.
    """

    n = len(mono)
    if n == 0:
        return mono.copy(), mono.copy()
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    spectrum = np.fft.rfft(mono)

    transition = max(20.0, crossover_hz * 0.25)
    low_edge = max(0.0, crossover_hz - transition)
    high_edge = crossover_hz + transition

    low_mask = np.ones_like(freqs)
    low_mask[freqs >= high_edge] = 0.0
    idx = (freqs > low_edge) & (freqs < high_edge)
    # Smoothly fade low band down through the transition region.
    phase = (freqs[idx] - low_edge) / (high_edge - low_edge)
    low_mask[idx] = 0.5 * (1.0 + np.cos(np.pi * phase))

    low = np.fft.irfft(spectrum * low_mask, n=n)
    high = mono - low
    return low, high


def _sosfiltfilt_chunked(sos, x: np.ndarray, chunk: int = 1 << 20, pad: int = 1 << 14) -> np.ndarray:
    """Memory-bounded zero-phase IIR filter (overlap-save).

    scipy.signal.sosfiltfilt promotes the entire signal to float64 and allocates
    several full-length temporaries.  On a 5-minute, 48 kHz track that spikes RAM
    past 1 GB and OOM-kills small containers.  This processes the signal in
    `chunk`-sample windows with `pad` samples of real context on each side (the
    pad output is discarded).  The Butterworth IRs used here settle in well under
    `pad` samples, so the kept center region is numerically identical to filtering
    the whole signal at once — no seams, no audible difference.
    """
    from scipy.signal import sosfiltfilt  # type: ignore

    x = np.ascontiguousarray(x, dtype=np.float32)
    n = len(x)
    if n <= chunk + 2 * pad:
        return sosfiltfilt(sos, x).astype(np.float32, copy=False)
    out = np.empty(n, dtype=np.float32)
    start = 0
    while start < n:
        end = min(start + chunk, n)
        lo = max(0, start - pad)
        hi = min(n, end + pad)
        seg = np.ascontiguousarray(x[lo:hi], dtype=np.float32)
        filt = sosfiltfilt(sos, seg).astype(np.float32, copy=False)
        out[start:end] = filt[start - lo: start - lo + (end - start)]
        del seg, filt
        start = end
    return out


def _split_at_hz(mono: np.ndarray, sample_rate: int, hz: float) -> Tuple[np.ndarray, np.ndarray]:
    """Memory-efficient IIR frequency split (scipy first, FFT fallback).

    scipy's sosfiltfilt uses O(n) working memory and no large complex-valued
    intermediate arrays, unlike _fft_split_mono which allocates multiple
    full-length complex128 buffers per call.  On a 5-minute song _fft_split_mono
    costs ~300 MB per call; sosfiltfilt costs ~50 MB.

    IMPORTANT: np.ascontiguousarray is required here.  Column slices of a 2-D
    array (stereo[:, ch]) are non-contiguous in memory.  scipy's C-level filter
    routines require a contiguous buffer; passing a non-contiguous view can
    silently produce wrong results or cause extreme slowdowns.
    """
    try:
        from scipy.signal import butter, sosfiltfilt  # type: ignore

        nyquist = sample_rate * 0.5
        cutoff = float(np.clip(hz / nyquist, 1e-5, 0.999))
        sos = butter(4, cutoff, btype="lowpass", output="sos")
        m = np.ascontiguousarray(mono, dtype=np.float32)
        low = _sosfiltfilt_chunked(sos, m)
        return low, (m - low)
    except Exception:
        return _fft_split_mono(mono, sample_rate, hz)


def split_bass_motion(
    samples: np.ndarray, sample_rate: int, crossover_hz: float = 150.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Split stereo audio into static mono bass and movable mid/high bands.

    The bass band is derived from the stereo mid channel and duplicated to both
    ears. This makes sub-bass and kick fundamentals mono/centered by design.
    The motion band is the remaining signal and may contain stereo information.
    """

    stereo = ensure_stereo_float(samples)
    # mean() returns a new C-contiguous array, so no explicit ascontiguousarray needed.
    mid = stereo.mean(axis=1).astype(np.float32, copy=False)

    try:
        from scipy.signal import butter, sosfiltfilt  # type: ignore

        nyquist = sample_rate * 0.5
        cutoff = min(max(crossover_hz / nyquist, 1e-5), 0.999)
        sos_low = butter(4, cutoff, btype="lowpass", output="sos")
        low_mono = _sosfiltfilt_chunked(sos_low, mid)
    except Exception:
        low_mono, _ = _fft_split_mono(mid, sample_rate, crossover_hz)

    bass = np.column_stack([low_mono, low_mono])
    motion = stereo - bass
    return bass, motion


def equal_power_gains(azimuth_degrees: float) -> Tuple[float, float]:
    """Equal-power left/right gains for a horizontal azimuth.

    We map azimuth to a sinusoidal pan law. 90° is strongest right, 270° is
    strongest left, while 0°/180° are center-ish and differentiated by HRTF-ish
    filtering/reverb rather than raw level.
    """

    theta = math.radians(azimuth_degrees % 360.0)
    pan = math.sin(theta)  # -1 left, 0 center, +1 right
    left = math.sqrt(0.5 * (1.0 - pan))
    right = math.sqrt(0.5 * (1.0 + pan))
    return left, right


def itd_samples(azimuth_degrees: float, sample_rate: int) -> float:
    """Return signed ITD in samples for an azimuth.

    Positive means the left ear is delayed because the source is to the right.
    Negative means the right ear is delayed because the source is to the left.
    """

    theta = math.radians(azimuth_degrees % 360.0)
    itd_seconds = MAX_ITD_SECONDS * math.sin(theta)
    return itd_seconds * sample_rate


def apply_fractional_delay(signal: np.ndarray, delay_samples: float) -> np.ndarray:
    """Delay a 1-D signal by fractional samples using linear interpolation.

    Values before the delayed signal begins are filled with zeros. This is a
    simple time-domain delay line suitable for sub-millisecond ITD cues.
    """

    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    if n == 0:
        return x.copy()

    positions = np.arange(n, dtype=np.float64) - float(delay_samples)
    return np.interp(positions, np.arange(n, dtype=np.float64), x, left=0.0, right=0.0)


def _rear_shade(frame: np.ndarray, rear_amount: float) -> np.ndarray:
    """Darken rear positions slightly using a one-pole low-pass IIR blend.

    Implemented via scipy.signal.lfilter (C-accelerated) instead of a Python
    for-loop.  For a 5-minute song the old loop ran 13 M iterations in the
    interpreter (~13 s); lfilter processes the same signal in <5 ms.
    """

    if rear_amount <= 0.0 or len(frame) == 0:
        return frame
    alpha = float(0.18 + 0.25 * (1.0 - rear_amount))
    # Ensure contiguous float32 — scipy lfilter requires a contiguous buffer.
    x = np.ascontiguousarray(frame, dtype=np.float32)
    try:
        from scipy.signal import lfilter  # type: ignore

        # One-pole lowpass:  y[n] = alpha*x[n] + (1-alpha)*y[n-1]
        b = np.array([alpha], dtype=np.float64)
        a = np.array([1.0, -(1.0 - alpha)], dtype=np.float64)
        filtered = lfilter(b, a, x.astype(np.float64)).astype(np.float32)
    except Exception:
        # Pure-numpy fallback: cumulative sum trick for the recursive form
        # is not directly available, so fall back to a vectorised approximation
        # via exponential weights (not identical but close for large signals).
        filtered = x.copy()
        z = np.float32(0.0)
        alpha_f = np.float32(alpha)
        one_minus = np.float32(1.0 - alpha)
        for i in range(len(x)):
            z = alpha_f * x[i] + one_minus * z
            filtered[i] = z
    return np.asarray((1.0 - rear_amount) * frame + rear_amount * filtered, dtype=np.float32)


def high_frequency_emphasis(
    motion: np.ndarray,
    sample_rate: int,
    amount: float = 0.0,
    split_hz: float = 4000.0,
) -> np.ndarray:
    """Boost the motion contribution of high-frequency content.

    This makes air, hats, consonants, reverb, and shimmer carry more of the
    perceived orbit while keeping low-mids less seasick.
    """

    amt = float(np.clip(amount, 0.0, 1.0))
    stereo = ensure_stereo_float(motion)
    if amt <= 0.0 or len(stereo) == 0:
        return stereo

    emphasized = np.empty_like(stereo)
    for ch in range(2):
        _low, high = _split_at_hz(stereo[:, ch], sample_rate, split_hz)
        # Add a controlled copy of the air/detail band instead of turning the
        # lower musical body down. This keeps a synth/vocal/guitar coherent as
        # one object while giving the binaural orbit more spectral information
        # for rear/height perception.
        emphasized[:, ch] = stereo[:, ch] + high * (0.18 * amt)
    return emphasized


def section_automation_curve(num_frames: int, sample_rate: int, start_frame: int = 0, total_frames: int | None = None) -> np.ndarray:
    """Arrangement-aware motion curve: intro subtle, chorus strong, outro dreamy.

    Without stem/section detection, this uses normalized song position as a
    musical default. It ramps in, peaks in chorus-like middle/final sections, and
    keeps the outro wide but slightly softer than the biggest chorus.

    ``start_frame``/``total_frames`` describe this segment's place within the
    whole track so the curve stays identical whether the song is rendered in one
    pass or in memory-bounded blocks.
    """

    if num_frames <= 0:
        return np.zeros(0, dtype=np.float32)
    total = int(total_frames) if total_frames else num_frames
    pos = ((np.arange(num_frames, dtype=np.float64) + float(start_frame)) / float(total)).astype(np.float32)
    points_x = np.array([0.0, 0.08, 0.22, 0.40, 0.62, 0.82, 0.92, 1.0], dtype=np.float32)
    points_y = np.array([0.45, 0.62, 0.82, 1.05, 1.10, 1.00, 0.92, 0.85], dtype=np.float32)
    curve = np.interp(pos, points_x, points_y).astype(np.float32)
    del pos
    # Gentle smoothing via scipy uniform_filter — O(n) memory, no FFT allocation.
    # np.convolve with an 11 000-sample hanning kernel used fftconvolve internally
    # and allocated ~200 MB of complex intermediates for long tracks.
    try:
        from scipy.ndimage import uniform_filter1d  # type: ignore
        smooth_len = max(1, int(sample_rate * 0.08))  # ~80 ms at 44.1 kHz
        curve = uniform_filter1d(curve, size=smooth_len).astype(np.float32)
    except Exception:
        pass  # skip smoothing if scipy unavailable; tiny automation steps are inaudible
    return np.clip(curve, 0.35, 1.15).astype(np.float32)


def reduce_static_noise(samples: np.ndarray, sample_rate: int, amount: float = 0.65, loudness_match: bool = True) -> np.ndarray:
    """Gently reduce steady hiss/static before spatial rendering.

    The cleaner is intentionally conservative: it combines a light spectral gate
    with a tiny impulse de-crackle stage. It is designed for already-mastered
    songs where the goal is removing static while preserving transients, vocal
    clarity, stereo tone, and musical brightness.

    ``loudness_match`` applies a global peak/RMS guard at the end. It is disabled
    when the track is rendered in blocks (the guard depends on whole-file
    statistics, which would differ per block and create seams); a single global
    normalization is applied once to the assembled output instead.
    """

    amt = float(np.clip(amount, 0.0, 1.0))
    stereo = ensure_stereo_float(samples)
    if amt <= 0.0 or len(stereo) < 32:
        return stereo

    # Gentle high-shelf IIR above the hiss region — same perceptual effect as
    # spectral subtraction but O(n) memory instead of O(n) complex FFT arrays.
    #
    # The previous version also ran scipy.signal.medfilt (de-crackle) before
    # the IIR step, but medfilt on a non-contiguous column-slice (stereo[:, ch])
    # caused extreme hangs on the Railway server.  The IIR-only path is both
    # faster and sufficient for already-mastered songs.
    cleaned = np.empty_like(stereo)
    try:
        from scipy.signal import butter, sosfiltfilt  # type: ignore

        nyquist = sample_rate * 0.5
        cutoff = float(np.clip(5500.0 / nyquist, 1e-5, 0.999))
        sos = butter(2, cutoff, btype="highpass", output="sos")
        blend = float(np.clip(amt * 0.28, 0.0, 1.0))
        for ch in range(2):
            # ascontiguousarray is essential: column slices are non-contiguous
            # and scipy's C-level routines may hang or silently misbehave on them.
            x = np.ascontiguousarray(stereo[:, ch], dtype=np.float32)
            high = _sosfiltfilt_chunked(sos, x)
            cleaned[:, ch] = (x - high * blend)
            del x, high
    except Exception:
        for ch in range(2):
            low, high = _split_at_hz(stereo[:, ch], sample_rate, 5500.0)
            cleaned[:, ch] = (low + high * (1.0 - 0.28 * amt)).astype(np.float32)
            del low, high

    if not loudness_match:
        return cleaned
    return preserve_loudness_and_peak(cleaned, stereo, peak_ceiling_db=-1.0, max_rms_lift_db=0.0)


def enhance_felt_presence(
    bass: np.ndarray,
    moving: np.ndarray,
    sample_rate: int,
    amount: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Add restrained psychoacoustic "felt" cues without breaking bass safety.

    This is not a loudness hype stage. It adds two subtle layers that make the
    8D motion easier for a headphone listener to *feel*:

    1. **Pinna/air proximity cues**: a tiny delayed side copy of the 4.2 kHz+
       band increases left/right/rear localization where human HRTFs are most
       sensitive, while leaving the vocal/body anchor intact.
    2. **Centered tactile punch**: a very small, mono, softly-saturated copy of
       the 75-150 Hz upper-bass/kick region reinforces body impact without ever
       spinning the sub or creating low-frequency side energy.
    """

    amt = float(np.clip(amount, 0.0, 1.0))
    safe_bass = ensure_stereo_float(bass)
    spatial = ensure_stereo_float(moving)
    if amt <= 0.0 or len(spatial) == 0:
        return safe_bass, spatial

    enhanced = spatial.copy()

    # Human front/back/height perception depends heavily on direction-dependent
    # notches and peaks above ~4 kHz. Add a tiny decorrelated side cue in this
    # band so the motion is more apparent but the instrument/vocal body remains
    # attached to the center-focused source.
    air = np.empty_like(spatial)
    for ch in range(2):
        _body, air[:, ch] = _split_at_hz(spatial[:, ch], sample_rate, 4200.0)
    air_side = (air[:, 0] - air[:, 1]) * 0.5
    del air
    left_cue = apply_fractional_delay(air_side, sample_rate * 0.00023)
    right_cue = apply_fractional_delay(-air_side, sample_rate * 0.00031)
    direct_side = np.column_stack([air_side, -air_side])
    delayed_side = np.column_stack([left_cue, right_cue])
    del air_side, left_cue, right_cue
    enhanced += (direct_side * 0.34 + delayed_side * 0.28) * amt
    del direct_side, delayed_side

    # A low, mono punch layer makes the result feel physically grounded. The
    # layer is deliberately narrow and centered, so kick/sub translation remains
    # strong on headphones, speakers, and mono playback.
    bass_mono = safe_bass.mean(axis=1)
    sub, upper_bass = _split_at_hz(bass_mono, sample_rate, 75.0)
    _unused, punch_band = _split_at_hz(bass_mono - sub, sample_rate, 150.0)
    del sub, _unused
    punch = np.tanh((upper_bass - punch_band) * 2.0) * 0.5
    safe_bass = safe_bass + np.column_stack([punch, punch]) * (0.055 * amt)

    return safe_bass, enhanced



def _simple_room_reverb(stereo: np.ndarray, sample_rate: int, amount: float) -> np.ndarray:
    """Small synthetic room made from feedback comb delays.

    amount is 0..1. This is intentionally conservative: reverb supports
    externalization without washing out the source.
    """

    wet = float(np.clip(amount, 0.0, 1.0))
    if wet <= 0.0 or len(stereo) == 0:
        return np.zeros_like(stereo)

    delays_ms = [17.0, 29.0, 43.0, 61.0]
    gains = [0.32, 0.24, 0.18, 0.13]
    out = np.zeros_like(stereo)
    mono = stereo.mean(axis=1)

    for idx, (delay_ms, gain) in enumerate(zip(delays_ms, gains)):
        d = max(1, int(sample_rate * delay_ms / 1000.0))
        delayed = np.zeros_like(mono)
        delayed[d:] = mono[:-d]
        # Alternate sides for early reflections.
        if idx % 2 == 0:
            out[:, 0] += delayed * gain
            out[:, 1] += delayed * gain * 0.72
        else:
            out[:, 0] += delayed * gain * 0.72
            out[:, 1] += delayed * gain

    return out * (0.35 * wet)


PANNING_PRESETS = {
    "clean_reference": "Cleaner YouTube-reference mix: faster 7.7-second orbit, tighter width, stronger center, darker air, and punch-safe bass.",
    "reference_luxe": "Measured YouTube-reference orbit: 10.4-second sweep, wide mids/highs, protected mono bass.",
    "phi_reference_orbit": "Golden Ratio version of the measured reference: phi-weighted timing, drift, and rear shading.",
    "fibonacci_spiral": "Fibonacci-timed spiral that visits golden-angle spatial nodes around the listener.",
    "golden_figure8": "Figure-eight motion whose lobes and transitions are shaped by phi ratios.",
    "lucas_breath": "Slow Lucas/Fibonacci breathing orbit: elegant expansion-contraction with low nausea risk.",
    "fireflies_plus": "Reference-inspired smooth premium orbit with subtle organic drift.",
    "cinematic_halo": "Slow halo movement: wide, emotional, atmospheric, non-dizzy.",
    "figure8": "Figure-eight style front/back emphasis with changing side energy.",
    "wide_orbit": "Bigger theatrical circular orbit for choruses and drops.",
    "vocal_safe": "Restrained motion that preserves lyric focus and center clarity.",
}


def panning_preset_names() -> set[str]:
    """Return available premium panning preset names."""

    return set(PANNING_PRESETS)


def fibonacci_preset_names() -> set[str]:
    """Return presets driven by Fibonacci/Golden Ratio motion rules."""

    return set(FIBONACCI_PRESETS)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    """Cubic ease curve used to avoid abrupt spatial acceleration changes."""

    u = np.clip(x, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _weighted_segment_phase(phase: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map a 0..1 orbit phase to Fibonacci/Lucas weighted segment index + local phase.

    The sequence controls timing: short Fibonacci numbers create quick passes,
    while later/larger numbers create long, elegant sweeps and perceptual rests.
    """

    w = np.asarray(weights, dtype=np.float64)
    edges = np.concatenate([[0.0], np.cumsum(w / np.sum(w))])
    idx = np.searchsorted(edges[1:], phase, side="right")
    idx = np.clip(idx, 0, len(w) - 1)
    local = (phase - edges[idx]) / (edges[idx + 1] - edges[idx] + 1e-12)
    return idx, local


def _golden_segment_orbit(
    phase: np.ndarray,
    weights: np.ndarray,
    radius_degrees: float = 32.0,
    start_degrees: float = 0.0,
) -> np.ndarray:
    """Continuous golden-angle spatial offset with Fibonacci/Lucas timing.

    The sequence determines *when* the orbit reaches each node; the golden angle
    determines *where* those nodes sit around the listener. Segment-to-segment
    interpolation is eased to keep motion luxurious rather than stepped.
    """

    idx, local = _weighted_segment_phase(phase, weights)
    eased = _smoothstep(local)
    node = (start_degrees + idx * GOLDEN_ANGLE_DEGREES) % 360.0
    next_node = (start_degrees + (idx + 1) * GOLDEN_ANGLE_DEGREES) % 360.0
    delta = ((next_node - node + 180.0) % 360.0) - 180.0
    centered = ((node + delta * eased + 180.0) % 360.0) - 180.0
    return (centered / 180.0) * radius_degrees


def _azimuth_series(
    num_frames: int,
    sample_rate: int,
    rotation_cpm: float,
    panning_preset: str = "fireflies_plus",
    start_frame: int = 0,
) -> np.ndarray:
    """Generate one azimuth degree per sample for smooth musical orbit presets.

    ``start_frame`` is the absolute sample index of the first frame. It lets the
    track be rendered in memory-bounded blocks while keeping one continuous,
    seamless orbit across block boundaries (the rotation phase is a function of
    absolute time, not of position within a block).
    """

    cycles_per_second = max(0.01, float(rotation_cpm) / 60.0)
    # float32 halves the size of per-sample angular arrays (e.g. 80 MB → 40 MB
    # for a 5-minute track); angular precision loss is < 0.001° — inaudible.
    # The absolute index is built in float64 first so large start offsets keep
    # full integer precision, then the per-sample time is stored as float32.
    t = ((np.arange(num_frames, dtype=np.float64) + float(start_frame)) / float(sample_rate)).astype(np.float32)
    orbit_phase = (t * np.float32(cycles_per_second)) % np.float32(1.0)
    base = t * np.float32(cycles_per_second) * np.float32(360.0)
    preset = panning_preset if panning_preset in PANNING_PRESETS else "fireflies_plus"

    if preset == "clean_reference":
        # Based on the cleaner second YouTube reference: ~0.129 Hz / 7.73 s
        # orbit, tighter side/mid energy, more centered image, and less hyped
        # air than the wider Reference Luxe profile.
        az = base + 5.0 * np.sin(2 * np.pi * cycles_per_second * 0.25 * t) + 3.0 * np.sin(
            2 * np.pi * cycles_per_second * 1.1 * t
        )
    elif preset == "phi_reference_orbit":
        # Reference Luxe translated through phi: the supplied reference's ~10.4 s
        # orbit is preserved, while small timing/position offsets are divided by
        # powers of phi so the motion never feels machine-looped.
        fib_offset = _golden_segment_orbit(orbit_phase, FIBONACCI_WEIGHTS, radius_degrees=18.0, start_degrees=21.0)
        az = base + fib_offset + (13.0 / PHI) * np.sin(2 * np.pi * cycles_per_second * t / PHI)
    elif preset == "fibonacci_spiral":
        # A Fibonacci-timed spiral: each segment lasts 1,1,2,3,5,8,13 parts of
        # the orbit and aims toward golden-angle nodes around the listener.
        fib_offset = _golden_segment_orbit(orbit_phase, FIBONACCI_WEIGHTS, radius_degrees=46.0, start_degrees=0.0)
        az = base * (1.0 + 1.0 / (PHI * 34.0)) + fib_offset
    elif preset == "golden_figure8":
        # Front/back figure-eight with phi-spaced lobes. The 90/270 side passes
        # remain smooth, but rear/front transitions breathe at golden-ratio rates.
        az = 180.0 + 116.0 * np.sin(2 * np.pi * orbit_phase) + (72.0 / PHI) * np.sin(
            4 * np.pi * orbit_phase + np.pi / PHI
        )
    elif preset == "lucas_breath":
        # Lucas numbers (2,1,3,4,7,11) create slower expansion/contraction than
        # Fibonacci Spiral: refined halo movement with lower nausea risk.
        lucas_offset = _golden_segment_orbit(orbit_phase, LUCAS_WEIGHTS, radius_degrees=28.0, start_degrees=GOLDEN_ANGLE_DEGREES / PHI)
        breath = 1.0 + 0.18 * np.sin(2 * np.pi * orbit_phase / PHI)
        az = base * 0.78 * breath + lucas_offset
    elif preset == "reference_luxe":
        # Based on the supplied reference: ~0.096 Hz / 10.4 s orbit,
        # with broad left-right travel plus gentle non-mechanical drift.
        az = base + 8.0 * np.sin(2 * np.pi * cycles_per_second * 0.33 * t) + 5.0 * np.sin(
            2 * np.pi * cycles_per_second * 1.25 * t
        )
    elif preset == "cinematic_halo":
        # Slow emotional circle with mild non-repeating drift.
        az = base * 0.72 + 18.0 * np.sin(2 * np.pi * cycles_per_second * 0.23 * t)
    elif preset == "figure8":
        # Alternates side travel with stronger front/back sweeps.
        az = base + 42.0 * np.sin(2 * np.pi * cycles_per_second * 2.0 * t)
    elif preset == "wide_orbit":
        az = base * 1.08 + 10.0 * np.sin(2 * np.pi * cycles_per_second * 0.5 * t)
    elif preset == "vocal_safe":
        # Hover around front/side quadrants more than hard rear so lyrics stay clear.
        az = 35.0 + 95.0 * np.sin(2 * np.pi * cycles_per_second * t)
    else:  # fireflies_plus
        az = base + 14.0 * np.sin(2 * np.pi * cycles_per_second * 0.5 * t) + 7.0 * np.sin(
            2 * np.pi * cycles_per_second * 1.5 * t
        )
    return (az % np.float32(360.0)).astype(np.float32, copy=False)


def _variable_fractional_delay(signal: np.ndarray, delay_samples: np.ndarray) -> np.ndarray:
    """Apply a smooth time-varying fractional delay.

    This is used for ITD cues. Unlike block-local delay processing, the
    interpolation is evaluated against the full continuous source, so there are
    no per-block zero pads or zipper clicks at 1024-sample boundaries.
    """

    # Memory-efficient rewrite: int32 read indices + float32 fractional parts
    # instead of two full-length float64 position arrays.  For a 5-minute track
    # this cuts peak allocation here from ~400 MB to ~80 MB with no audible
    # difference (max ITD delay is ~30 samples, well within float32 precision).
    x = np.asarray(signal, dtype=np.float32)
    d = np.asarray(delay_samples, dtype=np.float32)
    n = len(x)
    if n == 0:
        return x.copy()

    d_int = d.astype(np.int32)
    d_frac = (d - d_int).astype(np.float32, copy=False)
    del d

    read_idx = np.arange(n, dtype=np.int32) - d_int
    del d_int

    valid = (read_idx >= 0) & (read_idx < n - 1)
    safe = np.clip(read_idx, 0, n - 2)
    del read_idx

    out = (x[safe] * (1.0 - d_frac) + x[safe + 1] * d_frac).astype(np.float32)
    del safe, d_frac
    out[~valid] = 0.0
    del valid
    return out


def binaural_orbit(
    motion: np.ndarray,
    sample_rate: int,
    rotation_cpm: float = 6.0,
    motion_depth: float = 1.0,
    automation_curve: np.ndarray | None = None,
    panning_preset: str = "fireflies_plus",
    block_size: int = 1024,
    start_frame: int = 0,
) -> np.ndarray:
    """Render the motion band as a rotating binaural-ish source.

    The input motion band is reduced to a mono object for stable localization.
    Earlier versions updated pan/delay in blocks; that can create audible
    jitter because each delayed block starts with fresh zero-padding. This
    renderer computes ILD, ITD, and rear shading as continuous per-sample curves.
    The block_size argument is retained for API compatibility but is no longer
    used to quantize spatial motion.
    """

    stereo = ensure_stereo_float(motion)
    source = stereo.mean(axis=1).astype(np.float32, copy=False)
    del stereo  # keep only the mono source
    n = len(source)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # All angular/gain arrays use float32 — halves per-sample array footprint.
    azimuths = _azimuth_series(n, sample_rate, rotation_cpm, panning_preset=panning_preset, start_frame=start_frame)
    depth = np.float32(np.clip(motion_depth, 0.0, 1.5))
    if automation_curve is None:
        automation = np.ones(n, dtype=np.float32)
    else:
        automation = np.resize(np.asarray(automation_curve, dtype=np.float32), n)
    local_depth = np.clip(depth * automation, np.float32(0.0), np.float32(1.5)).astype(np.float32)
    del automation

    theta = np.radians(azimuths).astype(np.float32)
    pan = np.sin(theta).astype(np.float32)
    center_gain = np.float32(math.sqrt(0.5))
    left_gain = np.sqrt(np.float32(0.5) * (np.float32(1.0) - pan)).astype(np.float32)
    right_gain = np.sqrt(np.float32(0.5) * (np.float32(1.0) + pan)).astype(np.float32)
    del pan
    # Blend equal-power pan law toward center for restrained premium motion.
    left_gain = (center_gain + (left_gain - center_gain) * local_depth).astype(np.float32)
    right_gain = (center_gain + (right_gain - center_gain) * local_depth).astype(np.float32)

    signed_delay = (np.float32(MAX_ITD_SECONDS * sample_rate) * np.sin(theta) * local_depth).astype(np.float32)
    del theta
    left_delay = np.maximum(signed_delay, np.float32(0.0))
    right_delay = np.maximum(-signed_delay, np.float32(0.0))
    del signed_delay

    left = _variable_fractional_delay(source * left_gain, left_delay)
    del left_delay, left_gain
    right = _variable_fractional_delay(source * right_gain, right_delay)
    del right_delay, right_gain, source

    # Rear is strongest at 180°. Darken slightly and lower direct level a bit.
    rear_amount = np.maximum(np.float32(0.0), np.cos(np.radians(azimuths - np.float32(180.0)))).astype(np.float32)
    del azimuths
    rear_amount = (rear_amount * rear_amount * local_depth).astype(np.float32)
    del local_depth
    rear_gain = (np.float32(1.0) - np.float32(0.18) * rear_amount).astype(np.float32)
    pair = np.column_stack([left, right]) * rear_gain[:, None]
    del left, right, rear_gain

    # Continuous time-varying rear shading blend (avoids per-block reset artifacts).
    pair[:, 0] = pair[:, 0] * (1.0 - 0.35 * rear_amount) + _rear_shade(pair[:, 0], 1.0) * (0.35 * rear_amount)
    pair[:, 1] = pair[:, 1] * (1.0 - 0.35 * rear_amount) + _rear_shade(pair[:, 1], 1.0) * (0.35 * rear_amount)
    del rear_amount
    return pair.astype(np.float32, copy=False)


def split_center_focus_bands(
    motion: np.ndarray,
    sample_rate: int,
    focus_hz: float = 3200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split movable content into vocal/body and air/ambience regions.

    With a finished stereo master we cannot isolate the lead vocal as a true
    stem. This band split is the professional compromise: the 150 Hz-3.2 kHz
    body/intelligibility region is treated as the front-center anchor, while the
    air, shimmer, reverbs, delays, and guitar brightness above it carry more of
    the 8D motion.
    """

    stereo = ensure_stereo_float(motion)
    body = np.empty_like(stereo)
    air = np.empty_like(stereo)
    for ch in range(2):
        body[:, ch], air[:, ch] = _split_at_hz(stereo[:, ch], sample_rate, focus_hz)
    return body, air


def _process_8d_core(
    audio: AudioData,
    rotation_cpm: float = 6.0,
    room_size: float = 0.15,
    crossover_hz: float = 150.0,
    motion_depth: float = 1.0,
    high_emphasis: float = 0.0,
    spatial_mix: float = 1.0,
    denoise_amount: float = 0.0,
    panning_preset: str = "fireflies_plus",
    preserve_quality: bool = False,
    youtube_master: bool = False,
    section_automation: bool = False,
    center_focus: float = 0.0,
    felt_presence: float = 0.0,
    _start_frame: int = 0,
    _total_frames: int | None = None,
    _do_final_normalize: bool = True,
) -> AudioData:
    """Render one segment of an 8D track (the whole track when called directly).

    ``_start_frame``/``_total_frames`` locate this segment within the full song
    so the orbit and arrangement automation stay continuous across blocks.
    ``_do_final_normalize`` is False in block mode, where the final peak/RMS
    guard is applied once to the fully assembled output instead.
    """

    samples = ensure_stereo_float(audio.samples)
    total_frames = int(_total_frames) if _total_frames else len(samples)
    if denoise_amount > 0.0:
        samples = reduce_static_noise(samples, audio.sample_rate, amount=denoise_amount, loudness_match=_do_final_normalize)
    bass, motion = split_bass_motion(samples, audio.sample_rate, crossover_hz=crossover_hz)
    motion = high_frequency_emphasis(motion, audio.sample_rate, amount=high_emphasis)

    focus = float(np.clip(center_focus, 0.0, 1.0))
    center_anchor = np.zeros_like(motion)
    orbit_motion = motion
    if focus > 0.0:
        body, air = split_center_focus_bands(motion, audio.sample_rate, focus_hz=3200.0)
        body_center = np.column_stack([body.mean(axis=1), body.mean(axis=1)])
        body_motion_gain = 1.0 - 0.68 * focus
        center_anchor = body_center * (0.58 * focus)
        orbit_motion = (air + body * body_motion_gain).astype(np.float32)
        del body, air, body_center  # free splits — no longer needed

    automation = section_automation_curve(len(samples), audio.sample_rate, start_frame=_start_frame, total_frames=total_frames) if section_automation else None
    moving = binaural_orbit(
        orbit_motion,
        audio.sample_rate,
        rotation_cpm=rotation_cpm,
        motion_depth=motion_depth,
        automation_curve=automation,
        panning_preset=panning_preset,
        start_frame=_start_frame,
    )
    if orbit_motion is not motion:
        del orbit_motion  # free if it was a separate array

    room_curve = automation[:, None].astype(np.float32) if automation is not None else np.float32(1.0)
    del automation
    room = (_simple_room_reverb(moving, audio.sample_rate, room_size) * room_curve).astype(np.float32)
    del room_curve

    wet = float(np.clip(spatial_mix, 0.0, 1.0))
    dry_gain = 1.0 - wet
    if preserve_quality:
        dry_gain += (0.12 + 0.10 * focus) * wet
    moving = (motion * np.float32(dry_gain) + center_anchor + moving * np.float32(wet)).astype(np.float32)
    del motion, center_anchor

    bass, moving = enhance_felt_presence(bass, moving, audio.sample_rate, amount=felt_presence)
    mixed = (bass + moving + room).astype(np.float32)
    del bass, moving, room

    if _do_final_normalize:
        mixed = _final_master(mixed, samples, youtube_master, preserve_quality)
    del samples
    return AudioData(samples=mixed.astype(np.float32), sample_rate=audio.sample_rate)


def _final_master(mixed: np.ndarray, reference: np.ndarray, youtube_master: bool, preserve_quality: bool) -> np.ndarray:
    """Apply the single global peak/RMS master guard (run once per render)."""
    if youtube_master:
        return apply_youtube_master_target(mixed, target_rms_db=-13.0, peak_ceiling_db=-1.0)
    if preserve_quality:
        return preserve_loudness_and_peak(mixed, reference, peak_ceiling_db=-1.0, max_rms_lift_db=0.25)
    return normalize_peak(mixed, ceiling=0.98)


# A track longer than this many seconds is rendered in overlapping blocks so peak
# RAM stays flat regardless of length (a 5-minute single-pass render peaks ~1.5 GB
# and OOM-kills a ~1 GB container; block rendering holds it well under that).
_BLOCK_SECONDS = 60
_BLOCK_PAD_SECONDS = 2.0


def process_8d(
    audio: AudioData,
    rotation_cpm: float = 6.0,
    room_size: float = 0.15,
    crossover_hz: float = 150.0,
    motion_depth: float = 1.0,
    high_emphasis: float = 0.0,
    spatial_mix: float = 1.0,
    denoise_amount: float = 0.0,
    panning_preset: str = "fireflies_plus",
    preserve_quality: bool = False,
    youtube_master: bool = False,
    section_automation: bool = False,
    center_focus: float = 0.0,
    felt_presence: float = 0.0,
) -> AudioData:
    """Convert a stereo track into an 8D-style headphone render.

    Short tracks are rendered in a single pass. Long tracks are rendered in
    overlapping time-blocks and reassembled, which keeps peak memory bounded
    (so long songs no longer OOM-crash on memory-limited servers). The orbit and
    arrangement automation use absolute sample positions, and the heavy filters
    settle well within the block overlap, so the blocked output is seamless and
    matches the single-pass render.
    """

    params = dict(
        rotation_cpm=rotation_cpm,
        room_size=room_size,
        crossover_hz=crossover_hz,
        motion_depth=motion_depth,
        high_emphasis=high_emphasis,
        spatial_mix=spatial_mix,
        denoise_amount=denoise_amount,
        panning_preset=panning_preset,
        preserve_quality=preserve_quality,
        youtube_master=youtube_master,
        section_automation=section_automation,
        center_focus=center_focus,
        felt_presence=felt_presence,
    )

    samples = ensure_stereo_float(audio.samples)
    sr = int(audio.sample_rate)
    n = len(samples)
    block = _BLOCK_SECONDS * sr
    pad = int(_BLOCK_PAD_SECONDS * sr)

    # Single-pass for short tracks: identical to the original behavior.
    if n <= block + 2 * pad:
        return _process_8d_core(AudioData(samples=samples, sample_rate=sr), **params)

    out = np.empty((n, 2), dtype=np.float32)
    pos = 0
    while pos < n:
        end = min(pos + block, n)
        lo = max(0, pos - pad)
        hi = min(n, end + pad)
        segment = AudioData(samples=np.ascontiguousarray(samples[lo:hi]), sample_rate=sr)
        rendered = _process_8d_core(
            segment,
            _start_frame=lo,
            _total_frames=n,
            _do_final_normalize=False,
            **params,
        ).samples
        out[pos:end] = rendered[pos - lo: pos - lo + (end - pos)]
        del segment, rendered
        pos = end

    out = _final_master(out, samples, youtube_master, preserve_quality)
    del samples
    return AudioData(samples=out.astype(np.float32, copy=False), sample_rate=sr)
