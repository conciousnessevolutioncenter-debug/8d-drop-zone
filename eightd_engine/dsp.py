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


def ensure_stereo_float(samples: np.ndarray) -> np.ndarray:
    """Return a normalized stereo float64 array shaped (frames, 2)."""

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
        x = x.astype(np.float64) / max_value
    else:
        x = x.astype(np.float64, copy=False)

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


def split_bass_motion(
    samples: np.ndarray, sample_rate: int, crossover_hz: float = 150.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Split stereo audio into static mono bass and movable mid/high bands.

    The bass band is derived from the stereo mid channel and duplicated to both
    ears. This makes sub-bass and kick fundamentals mono/centered by design.
    The motion band is the remaining signal and may contain stereo information.
    """

    stereo = ensure_stereo_float(samples)
    mid = stereo.mean(axis=1)

    try:
        from scipy.signal import butter, sosfiltfilt  # type: ignore

        nyquist = sample_rate * 0.5
        cutoff = min(max(crossover_hz / nyquist, 1e-5), 0.999)
        sos_low = butter(4, cutoff, btype="lowpass", output="sos")
        low_mono = sosfiltfilt(sos_low, mid)
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
    """Darken rear positions slightly using a one-pole low-pass blend."""

    if rear_amount <= 0.0 or len(frame) == 0:
        return frame
    alpha = 0.18 + 0.25 * (1.0 - rear_amount)
    filtered = np.empty_like(frame)
    z = 0.0
    for i, value in enumerate(frame):
        z = z + alpha * (value - z)
        filtered[i] = z
    return (1.0 - rear_amount) * frame + rear_amount * filtered


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
        _low, high = _fft_split_mono(stereo[:, ch], sample_rate, split_hz)
        # Add a controlled copy of the air/detail band instead of turning the
        # lower musical body down. This keeps a synth/vocal/guitar coherent as
        # one object while giving the binaural orbit more spectral information
        # for rear/height perception.
        emphasized[:, ch] = stereo[:, ch] + high * (0.18 * amt)
    return emphasized


def section_automation_curve(num_frames: int, sample_rate: int) -> np.ndarray:
    """Arrangement-aware motion curve: intro subtle, chorus strong, outro dreamy.

    Without stem/section detection, this uses normalized song position as a
    musical default. It ramps in, peaks in chorus-like middle/final sections, and
    keeps the outro wide but slightly softer than the biggest chorus.
    """

    if num_frames <= 0:
        return np.zeros(0, dtype=np.float64)
    pos = np.linspace(0.0, 1.0, num_frames, endpoint=False)
    points_x = np.array([0.0, 0.08, 0.22, 0.40, 0.62, 0.82, 0.92, 1.0])
    points_y = np.array([0.45, 0.62, 0.82, 1.05, 1.10, 1.00, 0.92, 0.85])
    curve = np.interp(pos, points_x, points_y)
    # Gentle smoothing avoids audible jumps at section boundaries.
    smooth_len = max(1, int(sample_rate * 0.25))
    if smooth_len > 1 and len(curve) > smooth_len:
        kernel = np.hanning(smooth_len)
        kernel /= np.sum(kernel)
        curve = np.convolve(curve, kernel, mode="same")
    return np.clip(curve, 0.35, 1.15)


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


def _azimuth_series(num_frames: int, sample_rate: int, rotation_cpm: float) -> np.ndarray:
    """Generate one azimuth degree per sample for smooth circular orbit."""

    cycles_per_second = max(0.01, float(rotation_cpm) / 60.0)
    t = np.arange(num_frames, dtype=np.float64) / float(sample_rate)
    return (t * cycles_per_second * 360.0) % 360.0


def binaural_orbit(
    motion: np.ndarray,
    sample_rate: int,
    rotation_cpm: float = 6.0,
    motion_depth: float = 1.0,
    automation_curve: np.ndarray | None = None,
    block_size: int = 1024,
) -> np.ndarray:
    """Render the motion band as a rotating binaural-ish source.

    The input motion band is reduced to a mono object for stable localization.
    Each processing block gets an azimuth, equal-power gains, ITD delay, and
    subtle rear shading. Blocks are crossfaded by using short block sizes.
    """

    stereo = ensure_stereo_float(motion)
    source = stereo.mean(axis=1)
    n = len(source)
    out = np.zeros((n, 2), dtype=np.float64)
    azimuths = _azimuth_series(n, sample_rate, rotation_cpm)
    depth = float(np.clip(motion_depth, 0.0, 1.5))
    if automation_curve is None:
        automation = np.ones(n, dtype=np.float64)
    else:
        automation = np.resize(np.asarray(automation_curve, dtype=np.float64), n)

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = source[start:end]
        az = float(azimuths[start])
        local_depth = float(np.clip(depth * np.mean(automation[start:end]), 0.0, 1.5))
        left_gain, right_gain = equal_power_gains(az)
        signed_delay = itd_samples(az, sample_rate) * local_depth

        # Blend the equal-power pan gains back toward center to control extremity.
        center_gain = math.sqrt(0.5)
        left_gain = center_gain + (left_gain - center_gain) * local_depth
        right_gain = center_gain + (right_gain - center_gain) * local_depth

        left = block * left_gain
        right = block * right_gain
        if signed_delay > 0:
            left = apply_fractional_delay(left, signed_delay)
        elif signed_delay < 0:
            right = apply_fractional_delay(right, -signed_delay)

        # Rear is strongest at 180°. Darken slightly and lower direct level a bit.
        rear_amount = max(0.0, math.cos(math.radians(az - 180.0)))
        rear_amount = rear_amount * rear_amount * local_depth
        rear_gain = 1.0 - 0.18 * rear_amount
        pair = np.column_stack([left, right]) * rear_gain
        pair[:, 0] = _rear_shade(pair[:, 0], 0.35 * rear_amount)
        pair[:, 1] = _rear_shade(pair[:, 1], 0.35 * rear_amount)
        out[start:end] = pair

    return out


def process_8d(
    audio: AudioData,
    rotation_cpm: float = 6.0,
    room_size: float = 0.15,
    crossover_hz: float = 150.0,
    motion_depth: float = 1.0,
    high_emphasis: float = 0.0,
    spatial_mix: float = 1.0,
    preserve_quality: bool = False,
    youtube_master: bool = False,
    section_automation: bool = False,
) -> AudioData:
    """Convert a stereo track into an 8D-style headphone render."""

    samples = ensure_stereo_float(audio.samples)
    bass, motion = split_bass_motion(samples, audio.sample_rate, crossover_hz=crossover_hz)
    motion = high_frequency_emphasis(motion, audio.sample_rate, amount=high_emphasis)
    automation = section_automation_curve(len(samples), audio.sample_rate) if section_automation else None
    moving = binaural_orbit(
        motion,
        audio.sample_rate,
        rotation_cpm=rotation_cpm,
        motion_depth=motion_depth,
        automation_curve=automation,
    )
    room_curve = automation[:, None] if automation is not None else 1.0
    room = _simple_room_reverb(moving, audio.sample_rate, room_size) * room_curve
    wet = float(np.clip(spatial_mix, 0.0, 1.0))
    dry_gain = 1.0 - wet
    if preserve_quality:
        # Keep a quiet unspatialized mid/high "clarity bed" under the orbit so
        # vocals, transient detail, and original stereo tone are enhanced rather
        # than hollowed out by the mono moving object.
        dry_gain += 0.12 * wet
    moving = motion * dry_gain + moving * wet
    mixed = bass + moving + room
    if youtube_master:
        mixed = apply_youtube_master_target(mixed, target_rms_db=-13.0, peak_ceiling_db=-1.0)
    elif preserve_quality:
        mixed = preserve_loudness_and_peak(mixed, samples, peak_ceiling_db=-1.0, max_rms_lift_db=0.25)
    else:
        mixed = normalize_peak(mixed, ceiling=0.98)
    return AudioData(samples=mixed, sample_rate=audio.sample_rate)
