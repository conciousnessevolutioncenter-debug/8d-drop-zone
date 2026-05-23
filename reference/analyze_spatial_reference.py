import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt


def rms(a):
    a = np.asarray(a, dtype=np.float64)
    return float(np.sqrt(np.mean(a * a) + 1e-18))


def filt(data, sr, kind, hz):
    sos = butter(4, hz / (sr / 2), btype=kind, output="sos")
    return sosfiltfilt(sos, data)


def band(data, sr, lo, hi):
    if lo <= 20:
        return filt(data, sr, "lowpass", hi)
    sos = butter(4, [lo / (sr / 2), min(hi / (sr / 2), 0.999)], btype="bandpass", output="sos")
    return sosfiltfilt(sos, data)


def analyze(path):
    x, sr = sf.read(path, always_2d=True, dtype="float64")
    x = x[:, :2]
    L, R = x[:, 0], x[:, 1]
    mid = (L + R) * 0.5
    side = (L - R) * 0.5
    low_l = filt(L, sr, "lowpass", 150)
    low_r = filt(R, sr, "lowpass", 150)
    low_mid = (low_l + low_r) * 0.5
    low_side = (low_l - low_r) * 0.5

    win = int(0.5 * sr)
    hop = int(0.1 * sr)
    pans, corrs, ratios, low_ratios = [], [], [], []
    for start in range(0, max(0, len(x) - win), hop):
        l = L[start:start + win]
        r = R[start:start + win]
        le, re = np.mean(l * l), np.mean(r * r)
        pans.append((re - le) / (re + le + 1e-18))
        c = np.corrcoef(l, r)[0, 1] if rms(l) > 1e-9 and rms(r) > 1e-9 else 1.0
        corrs.append(float(c) if np.isfinite(c) else 1.0)
        m, s = (l + r) * 0.5, (l - r) * 0.5
        ratios.append(rms(s) / (rms(m) + 1e-12))
        ll, rr = low_l[start:start + win], low_r[start:start + win]
        lm, ls = (ll + rr) * 0.5, (ll - rr) * 0.5
        low_ratios.append(rms(ls) / (rms(lm) + 1e-12))

    pans = np.asarray(pans, dtype=np.float64)
    if len(pans) > 8:
        p = pans - np.mean(pans)
        freqs = np.fft.rfftfreq(len(p), d=hop / sr)
        spec = np.abs(np.fft.rfft(p * np.hanning(len(p))))
        mask = (freqs >= 0.02) & (freqs <= 0.3)
        dom = float(freqs[mask][np.argmax(spec[mask])]) if np.any(mask) else 0.0
    else:
        dom = 0.0

    mono = mid
    band_defs = [(20, 80, "sub"), (80, 150, "bass"), (150, 500, "lowmid"), (500, 2000, "mid"), (2000, 6000, "presence"), (6000, 16000, "air")]
    band_db = {}
    for lo, hi, name in band_defs:
        y = band(mono, sr, lo, hi)
        band_db[name] = float(20 * np.log10(rms(y) / (rms(mono) + 1e-12) + 1e-12))

    return {
        "path": str(path),
        "sample_rate": int(sr),
        "duration_sec": len(x) / sr,
        "overall_corr": float(np.corrcoef(L, R)[0, 1]),
        "overall_side_mid": rms(side) / (rms(mid) + 1e-12),
        "low_side_mid_below_150": rms(low_side) / (rms(low_mid) + 1e-12),
        "pan_min_max_std": [float(np.min(pans)), float(np.max(pans)), float(np.std(pans))] if len(pans) else [0, 0, 0],
        "dominant_pan_hz": dom,
        "seconds_per_rotation": (1 / dom) if dom else None,
        "cycles_per_min": dom * 60,
        "corr_p10_p50_p90": [float(v) for v in np.percentile(corrs, [10, 50, 90])] if corrs else [1, 1, 1],
        "side_mid_p10_p50_p90": [float(v) for v in np.percentile(ratios, [10, 50, 90])] if ratios else [0, 0, 0],
        "low_side_mid_p50_p90": [float(v) for v in np.percentile(low_ratios, [50, 90])] if low_ratios else [0, 0],
        "band_db_relative": band_db,
    }


if __name__ == "__main__":
    print(json.dumps([analyze(Path(p)) for p in sys.argv[1:]], indent=2))
