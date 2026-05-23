"""Audio file loading/export helpers for The 8D Engine.

Primary path: soundfile for WAV/FLAC/OGG.
MP3 path: pydub + FFmpeg when available.

The DSP core never knows about file formats; it only receives/returns AudioData.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import wave

import numpy as np

from .dsp import AudioData, ensure_stereo_float, normalize_peak


COMMON_AUDIO_INPUTS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".m4a", ".aac", ".wma", ".opus", ".alac"}
# Backward-compatible name used by the GUI for hints only. Decoding is no longer
# gated by extension; load_audio tries soundfile first and then FFmpeg/pydub.
SUPPORTED_INPUTS = COMMON_AUDIO_INPUTS
SUPPORTED_OUTPUTS = {".wav", ".flac", ".ogg", ".mp3"}


def load_audio(path: str | Path) -> AudioData:
    """Load an audio file as stereo float samples.

    SoundFile handles lossless formats directly. Pydub/FFmpeg is used as a
    broad fallback for compressed or unusual containers. The function does not
    reject by extension; it attempts to decode whatever file the user drops.
    """

    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(src)

    try:
        import soundfile as sf  # type: ignore

        samples, sample_rate = sf.read(str(src), always_2d=True, dtype="float64")
        return AudioData(samples=ensure_stereo_float(samples), sample_rate=int(sample_rate))
    except Exception as sf_error:
        if src.suffix.lower() == ".wav":
            try:
                return _load_wav_stdlib(src)
            except Exception:
                pass

        try:
            from pydub import AudioSegment  # type: ignore
        except Exception as import_error:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "Could not load audio. WAV works without extras; for broad audio format support install "
                "soundfile, pydub, and FFmpeg."
            ) from import_error

        try:
            segment = AudioSegment.from_file(str(src))
        except Exception as pydub_error:  # pragma: no cover - environment-specific
            raise RuntimeError(f"Could not decode {src}: {sf_error}; {pydub_error}") from pydub_error

        sample_rate = segment.frame_rate
        channels = segment.channels
        raw = np.array(segment.get_array_of_samples())
        if channels > 1:
            raw = raw.reshape((-1, channels))[:, :2]
        max_value = float(1 << (8 * segment.sample_width - 1))
        samples = raw.astype(np.float64) / max_value
        return AudioData(samples=ensure_stereo_float(samples), sample_rate=sample_rate)


def export_audio(audio: AudioData, path: str | Path) -> Path:
    """Export stereo audio to WAV/FLAC/OGG/MP3.

    MP3 export requires pydub and FFmpeg. WAV is recommended for best quality.
    """

    dst = Path(path)
    suffix = dst.suffix.lower()
    if suffix not in SUPPORTED_OUTPUTS:
        raise ValueError(f"Unsupported output format: {suffix}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    samples = normalize_peak(ensure_stereo_float(audio.samples), ceiling=0.98)

    if suffix == ".mp3":
        try:
            from pydub import AudioSegment  # type: ignore
            import soundfile as sf  # type: ignore
        except Exception as import_error:  # pragma: no cover - environment-specific
            raise RuntimeError("MP3 export requires soundfile, pydub, and FFmpeg") from import_error

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            sf.write(str(tmp_path), samples, audio.sample_rate, subtype="PCM_16")
            AudioSegment.from_wav(str(tmp_path)).export(str(dst), format="mp3", bitrate="320k")
        finally:
            tmp_path.unlink(missing_ok=True)
        return dst

    try:
        import soundfile as sf  # type: ignore
    except Exception as import_error:  # pragma: no cover - environment-specific
        if suffix == ".wav":
            _export_wav_stdlib(AudioData(samples=samples, sample_rate=audio.sample_rate), dst)
            return dst
        raise RuntimeError("Export requires soundfile. Install with: pip install soundfile") from import_error

    subtype = "FLOAT" if suffix == ".wav" else None
    sf.write(str(dst), samples, audio.sample_rate, subtype=subtype)
    return dst


def _export_wav_stdlib(audio: AudioData, path: Path) -> None:
    """Write 16-bit PCM WAV using only Python's standard library."""

    samples = np.clip(ensure_stereo_float(audio.samples), -1.0, 1.0)
    pcm = (samples * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(audio.sample_rate)
        wav.writeframes(pcm.tobytes())


def _load_wav_stdlib(path: Path) -> AudioData:
    """Read PCM WAV using only Python's standard library."""

    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise ValueError("stdlib WAV fallback currently supports 16-bit PCM WAV only")
    raw = np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        raw = raw.reshape((-1, channels))[:, :2]
    samples = raw.astype(np.float64) / 32768.0
    return AudioData(samples=ensure_stereo_float(samples), sample_rate=sample_rate)
