"""Preview and final output use one deterministic settings adapter."""
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable
import math
import subprocess
import numpy as np
import soundfile as sf
from eightd_engine.audio_io import _find_ffmpeg
from eightd_engine.dsp import render_8d_file_to_wav, measure_loudness_file
from .models import MixSettings

class RenderCancelled(Exception):
    pass

@dataclass(frozen=True)
class RenderRequest:
    source: Path
    output: Path
    settings: MixSettings
    format: str
    start_seconds: float | None = None
    duration_seconds: float | None = None


def render(request: RenderRequest, progress: Callable[[float], None],
           cancelled: Callable[[], bool]) -> dict:
    def check():
        if cancelled():
            raise RenderCancelled('Render cancelled')
    check()
    if request.format not in ('wav', 'flac', 'mp3'):
        raise ValueError('Choose WAV, FLAC or MP3.')
    if request.output.exists():
        raise ValueError('Output already exists; create a new render.')
    request.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.render-', dir=request.output.parent) as temp:
        work = Path(temp)
        source = request.source
        info = sf.info(source)
        if not info.frames:
            raise ValueError('Empty audio cannot be rendered.')
        if request.start_seconds is not None:
            start = request.start_seconds
            length = request.duration_seconds if request.duration_seconds is not None else 15
            if not math.isfinite(start) or start < 0 or start >= info.duration or not math.isfinite(length) or length <= 0 or length > 30:
                raise ValueError('Preview must start inside the track and last at most 30 seconds.')
            source = work / 'excerpt.wav'
            with sf.SoundFile(request.source) as reader:
                reader.seek(int(start * info.samplerate))
                audio = reader.read(min(int(length * info.samplerate), info.frames-reader.tell()), dtype='float32', always_2d=True)
            sf.write(source, audio, info.samplerate, subtype='FLOAT')
        check()
        wav = work / 'processed.wav'
        def on_progress(done, total):
            check()
            progress(0.85 * done / max(total, 1))
            check()
        params = request.settings.model_dump()
        params.update(preserve_quality=True, youtube_master=False)
        render_8d_file_to_wav(source, wav, on_progress=on_progress, **params)
        check()
        result_file = wav
        if request.format == 'flac':
            result_file = work / 'processed.flac'
            with sf.SoundFile(wav) as reader, sf.SoundFile(result_file, 'w', samplerate=reader.samplerate, channels=2, subtype='PCM_24') as writer:
                for block in reader.blocks(blocksize=65536, dtype='float32', always_2d=True):
                    check()
                    writer.write(block)
        elif request.format == 'mp3':
            result_file = work / 'processed.mp3'
            subprocess.run([_find_ffmpeg(), '-nostdin', '-v', 'error', '-y', '-i', str(wav),
                            '-codec:a', 'libmp3lame', '-b:a', '320k', str(result_file)],
                           check=True, capture_output=True, timeout=180)
        check()
        progress(0.92)
        # SoundFile supports MP3 on supported packaged libsndfile versions.
        info = sf.info(result_file)
        peak = 0.0
        sums = np.zeros(5, dtype=np.float64)
        count = 0
        for block in sf.blocks(result_file, blocksize=65536, dtype='float64', always_2d=True):
            check()
            if not np.isfinite(block).all():
                raise ValueError('Rendered audio contains invalid samples.')
            peak = max(peak, float(np.max(np.abs(block))))
            l, r = block[:, 0], block[:, 1]
            sums += [l.sum(), r.sum(), np.dot(l,l), np.dot(r,r), np.dot(l,r)]
            count += len(l)
        if peak > 1.00001:
            raise ValueError('Encoded output exceeds full scale. Export WAV or lower source level.')
        lsum, rsum, ll, rr, lr = sums
        denominator = math.sqrt(max(0, ll-lsum**2/count) * max(0, rr-rsum**2/count))
        correlation = float(np.clip((lr-lsum*rsum/count) / denominator, -1, 1)) if denominator > 1e-12 else 0.0
        try:
            lufs, true_peak = measure_loudness_file(result_file)
        except Exception:
            lufs, true_peak = None, None
        clean = lambda x: float(x) if x is not None and math.isfinite(x) else None
        report = dict(sample_rate=info.samplerate, frames=info.frames, duration=info.duration,
                      channels=info.channels, sample_peak=peak, correlation=correlation,
                      loudness_lufs=clean(lufs), true_peak_db=clean(true_peak))
        check()
        result_file.replace(request.output)
        return report
