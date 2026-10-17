"""Managed imports and bounded-memory waveform extraction."""
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import BinaryIO
import numpy as np
import soundfile as sf
from .config import StudioConfig
from eightd_engine.audio_io import _find_ffmpeg


def decode_to_wav(source: Path, destination: Path, max_seconds: float = 3600):
    try:
        with sf.SoundFile(source) as reader:
            with sf.SoundFile(destination, 'w', samplerate=reader.samplerate,
                              channels=reader.channels, subtype='FLOAT', format='WAV') as writer:
                for block in reader.blocks(blocksize=65536, dtype='float32', always_2d=True):
                    if not np.isfinite(block).all():
                        raise ValueError('Audio contains invalid sample values.')
                    writer.write(block)
        return
    except sf.LibsndfileError:
        destination.unlink(missing_ok=True)
    try:
        subprocess.run([_find_ffmpeg(), '-nostdin', '-v', 'error', '-y', '-i', str(source),
                        '-t', str(max_seconds + 1), '-vn', '-c:a', 'pcm_f32le', str(destination)],
                       check=True, capture_output=True, timeout=180)
    except (RuntimeError, subprocess.SubprocessError, OSError) as exc:
        raise ValueError('Cannot decode this audio. For MP3/M4A, install FFmpeg and restart Studio.') from exc


def import_audio(stream: BinaryIO, filename: str, config: StudioConfig) -> dict:
    ident = uuid.uuid4().hex
    folder = config.data_dir / 'assets' / ident
    folder.mkdir(parents=True)
    original = folder / ('original' + Path(filename).suffix[:12])
    decoded = folder / 'audio.wav'
    try:
        total = 0
        with original.open('wb') as output:
            while chunk := stream.read(1024 * 1024):
                total += len(chunk)
                if total > config.max_bytes:
                    raise ValueError('Audio exceeds the 200 MB upload limit.')
                output.write(chunk)
        if total == 0:
            raise ValueError('The uploaded file is empty.')
        # Inspect native formats before decoding large inputs.
        try:
            info = sf.info(original)
        except sf.LibsndfileError:
            info = None
        if info is not None and info.duration > config.max_seconds:
            raise ValueError('Audio exceeds the 60-minute duration limit.')
        decode_to_wav(original, decoded, config.max_seconds)
        info = sf.info(decoded)
        if info.frames < 1 or info.samplerate <= 0 or info.duration > config.max_seconds:
            raise ValueError('Audio must be nonempty and at most 60 minutes long.')
        for block in sf.blocks(decoded, blocksize=65536, dtype='float32', always_2d=True):
            if not np.isfinite(block).all():
                raise ValueError('Audio contains invalid sample values.')
        return dict(id=ident, path=str(decoded.relative_to(config.data_dir)),
                    original_path=str(original.relative_to(config.data_dir)), name=filename[:250],
                    sample_rate=info.samplerate, frames=info.frames, duration=info.duration,
                    channels=info.channels)
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise


def waveform_peaks(path: Path, bins: int = 1200) -> list[list[float]]:
    with sf.SoundFile(path) as reader:
        blocksize = max(1, int(np.ceil(reader.frames / bins)))
        peaks = []
        for block in reader.blocks(blocksize=blocksize, dtype='float32', always_2d=True):
            peaks.append([float(np.min(block)), float(np.max(block))])
        return peaks
