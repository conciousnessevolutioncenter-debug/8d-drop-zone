"""Stem-aware spatial rendering for The 8D Engine.

This module separates the *mix decision layer* from the lower-level DSP core:
- optional Demucs separation can provide real vocals/drums/bass/other stems
- deterministic role plans decide how each element should move
- the existing premium 8D renderer processes only the stems that should move

If Demucs is not installed, callers can still use the same pipeline with supplied
stems or fall back to the classic full-mix renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import importlib.util
import subprocess
import sys
import tempfile
from typing import Mapping

import numpy as np

from .audio_io import load_audio
from .dsp import AudioData, ensure_stereo_float, normalize_peak, preserve_loudness_and_peak, process_8d


class StemRole(Enum):
    """Musical role used for stem-specific spatial mix decisions."""

    VOCAL = "vocal"
    PERCUSSION = "percussion"
    BASS = "bass"
    INSTRUMENT = "instrument"
    AMBIENCE = "ambience"


@dataclass(frozen=True)
class StemData:
    """Floating-point stereo stem audio."""

    samples: np.ndarray
    sample_rate: int


@dataclass(frozen=True)
class StemSpatialSettings:
    """Per-stem spatial treatment."""

    role: StemRole
    room_size: float
    motion_depth: float
    high_emphasis: float
    spatial_mix: float
    center_focus: float
    felt_presence: float = 0.0


class StemSeparationUnavailable(RuntimeError):
    """Raised when an AI stem model is requested but unavailable."""


def available_stem_mode() -> dict[str, str]:
    """Report the best stem mode available in this Python environment."""

    if importlib.util.find_spec("demucs") is not None:
        return {
            "mode": "demucs",
            "message": "Demucs is available: AI stems can be separated into vocals, drums, bass, and other before spatial rendering.",
        }
    return {
        "mode": "hybrid_fallback",
        "message": "Demucs is not installed: supplied stems can be spatialized, or the app can fall back to the classic full-mix render.",
    }


def infer_stem_role(name: str) -> StemRole:
    """Infer a musical role from a stem filename/model label."""

    label = name.lower().replace("-", "_").replace(" ", "_")
    if any(term in label for term in ("vocal", "voice", "singer", "lead")):
        return StemRole.VOCAL
    if any(term in label for term in ("drum", "percussion", "kick", "snare", "hat", "cymbal")):
        return StemRole.PERCUSSION
    if any(term in label for term in ("bass", "sub", "808")):
        return StemRole.BASS
    if any(term in label for term in ("ambience", "reverb", "room", "fx", "pad", "atmos")):
        return StemRole.AMBIENCE
    return StemRole.INSTRUMENT


def settings_for_role(role: StemRole) -> StemSpatialSettings:
    """Return safe premium defaults for a musical role."""

    if role is StemRole.VOCAL:
        # Lead stays front/center; only air and room bloom around it.
        return StemSpatialSettings(role, room_size=0.18, motion_depth=0.40, high_emphasis=0.55, spatial_mix=0.34, center_focus=0.94, felt_presence=0.24)
    if role is StemRole.PERCUSSION:
        # Keep transient punch stable while allowing hats/cymbals to widen.
        return StemSpatialSettings(role, room_size=0.12, motion_depth=0.48, high_emphasis=0.62, spatial_mix=0.38, center_focus=0.72, felt_presence=0.46)
    if role is StemRole.BASS:
        # Bass/kick/sub are non-negotiably centered.
        return StemSpatialSettings(role, room_size=0.0, motion_depth=0.0, high_emphasis=0.0, spatial_mix=0.0, center_focus=1.0, felt_presence=0.0)
    if role is StemRole.AMBIENCE:
        return StemSpatialSettings(role, room_size=0.30, motion_depth=0.78, high_emphasis=0.78, spatial_mix=0.76, center_focus=0.20, felt_presence=0.36)
    return StemSpatialSettings(role, room_size=0.24, motion_depth=0.72, high_emphasis=0.72, spatial_mix=0.66, center_focus=0.36, felt_presence=0.34)


def build_stem_spatial_plan(stems: Mapping[str, StemData]) -> dict[str, StemSpatialSettings]:
    """Create the per-stem spatial plan for a set of separated stems."""

    return {name: settings_for_role(infer_stem_role(name)) for name in stems}


def _fit_length(samples: np.ndarray, frames: int) -> np.ndarray:
    stereo = ensure_stereo_float(samples)
    if len(stereo) == frames:
        return stereo
    fitted = np.zeros((frames, 2), dtype=np.float64)
    n = min(frames, len(stereo))
    if n:
        fitted[:n] = stereo[:n]
    return fitted


def _center_mono(samples: np.ndarray) -> np.ndarray:
    stereo = ensure_stereo_float(samples)
    mono = stereo.mean(axis=1)
    return np.column_stack([mono, mono])


def process_stem_spatial_mix(
    stems: Mapping[str, StemData],
    reference: AudioData,
    rotation_cpm: float = 6.0,
    panning_preset: str = "cinematic_halo",
    plan: Mapping[str, StemSpatialSettings] | None = None,
) -> AudioData:
    """Spatialize separated musical elements and recombine them.

    Bass roles are kept mono/static. Vocals are anchored front-center. Instruments,
    ambience, and high-frequency percussion receive stronger motion and room cues.
    The final render is guarded against excess gain/clipping against the reference.
    """

    if not stems:
        raise ValueError("At least one stem is required for stem spatial mixing")

    ref_samples = ensure_stereo_float(reference.samples)
    frames = len(ref_samples)
    sr = int(reference.sample_rate)
    spatial_plan = dict(plan or build_stem_spatial_plan(stems))
    mixed = np.zeros((frames, 2), dtype=np.float64)

    for name, stem in stems.items():
        if int(stem.sample_rate) != sr:
            raise ValueError(f"Stem {name!r} sample rate {stem.sample_rate} does not match reference {sr}")
        settings = spatial_plan.get(name, settings_for_role(infer_stem_role(name)))
        stem_samples = _fit_length(stem.samples, frames)
        if settings.role is StemRole.BASS:
            mixed += _center_mono(stem_samples)
            continue
        rendered = process_8d(
            AudioData(stem_samples, sr),
            rotation_cpm=rotation_cpm,
            room_size=settings.room_size,
            crossover_hz=150.0,
            motion_depth=settings.motion_depth,
            high_emphasis=settings.high_emphasis,
            spatial_mix=settings.spatial_mix,
            denoise_amount=0.0,
            panning_preset=panning_preset,
            preserve_quality=True,
            youtube_master=False,
            section_automation=True,
            center_focus=settings.center_focus,
            felt_presence=settings.felt_presence,
        ).samples
        mixed += rendered

    guarded = preserve_loudness_and_peak(mixed, ref_samples, peak_ceiling_db=-1.0, max_rms_lift_db=0.20)
    guarded = normalize_peak(guarded, ceiling=0.98)
    return AudioData(guarded, sr)


def separate_stems_from_file(src: Path, work_dir: Path | None = None, model: str = "htdemucs") -> dict[str, StemData]:
    """Separate an audio file with Demucs and return loaded stems.

    The function intentionally depends on the Demucs CLI/module only when called.
    This keeps normal app startup fast and lets installations without Demucs keep
    using the classic renderer.
    """

    if importlib.util.find_spec("demucs") is None:
        raise StemSeparationUnavailable("Demucs is not installed in this environment")

    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(src)
    base_dir = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="8d_demucs_"))
    base_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "demucs", "-n", model, "-o", str(base_dir), str(src)]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=60 * 30)
    if proc.returncode != 0:
        raise StemSeparationUnavailable((proc.stderr or proc.stdout or "Demucs separation failed").strip())

    stem_dir = base_dir / model / src.stem
    if not stem_dir.exists():
        candidates = list(base_dir.glob(f"*/{src.stem}"))
        if candidates:
            stem_dir = candidates[0]
    stems: dict[str, StemData] = {}
    for wav in stem_dir.glob("*.wav"):
        audio = load_audio(wav)
        stems[wav.stem] = StemData(audio.samples, audio.sample_rate)
    if not stems:
        raise StemSeparationUnavailable("Demucs completed but no stem WAV files were found")
    return stems
