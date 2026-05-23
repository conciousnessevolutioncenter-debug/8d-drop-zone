from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np


PATH = Path("reference/fireflies_8d_reference.wav")


def read_wav(path: Path):
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        frames = w.readframes(w.getnframes())
    x = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    x = x.reshape((-1, ch))[:, :2]
    return x, sr


def rms(x):
    return np.sqrt(np.mean(x * x) + 1e-18)


def db(x):
    return 20 * np.log10(max(float(x), 1e-12))


def band_energy_fft(x, sr, bands):
    mono = x.mean(axis=1)
    n = len(mono)
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(mono * win)) ** 2
    freqs = np.fft.rfftfreq(n, 1 / sr)
    out = {}
    total = np.sum(spec) + 1e-18
    for name, lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        out[name] = float(np.sum(spec[m]) / total)
    centroid = float(np.sum(freqs * spec) / total)
    return out, centroid


def estimate_rotation_from_pan(x, sr, hop_s=0.1, win_s=0.5):
    hop = int(hop_s * sr)
    win = int(win_s * sr)
    pans=[]; times=[]; cors=[]; side_ratios=[]; bass_side=[]
    for start in range(0, len(x)-win, hop):
        seg=x[start:start+win]
        l=seg[:,0]; r=seg[:,1]
        el=np.mean(l*l)+1e-12; er=np.mean(r*r)+1e-12
        pans.append((er-el)/(er+el))
        times.append((start+win/2)/sr)
        cors.append(float(np.corrcoef(l,r)[0,1]))
        mid=(l+r)*0.5; side=(l-r)*0.5
        side_ratios.append(float(rms(side)/(rms(mid)+1e-12)))
        # crude low-pass via FFT under 150 Hz to estimate bass width by window
        freqs=np.fft.rfftfreq(win,1/sr)
        L=np.fft.rfft(l*np.hanning(win)); R=np.fft.rfft(r*np.hanning(win))
        m=freqs<150
        mid_low=(L[m]+R[m])*0.5; side_low=(L[m]-R[m])*0.5
        bass_side.append(float(np.sqrt(np.mean(np.abs(side_low)**2))/(np.sqrt(np.mean(np.abs(mid_low)**2))+1e-12)))
    pans=np.array(pans); times=np.array(times)
    # remove slow DC, find dominant modulation frequency 0.02-0.3 Hz
    y=pans-np.mean(pans)
    spec=np.abs(np.fft.rfft(y*np.hanning(len(y))))
    freqs=np.fft.rfftfreq(len(y),hop_s)
    m=(freqs>=0.02)&(freqs<=0.3)
    dom=float(freqs[m][np.argmax(spec[m])]) if np.any(m) else 0.0
    return {
      "pan_min": float(np.min(pans)),
      "pan_max": float(np.max(pans)),
      "pan_std": float(np.std(pans)),
      "dominant_pan_hz": dom,
      "dominant_rotation_seconds": float(1/dom) if dom else None,
      "mean_corr": float(np.mean(cors)),
      "min_corr": float(np.min(cors)),
      "p10_corr": float(np.percentile(cors,10)),
      "mean_side_mid_ratio": float(np.mean(side_ratios)),
      "p90_side_mid_ratio": float(np.percentile(side_ratios,90)),
      "mean_bass_side_mid_ratio_lt150": float(np.mean(bass_side)),
      "p90_bass_side_mid_ratio_lt150": float(np.percentile(bass_side,90)),
    }


def section_stats(x, sr):
    sections=[(0,30),(30,60),(60,90),(90,120),(120,150),(150,180),(180,210),(210,228.98)]
    out=[]
    bands=[('sub_20_80',20,80),('bass_80_150',80,150),('lowmid_150_500',150,500),('mid_500_2k',500,2000),('presence_2k_6k',2000,6000),('air_6k_16k',6000,16000)]
    for a,b in sections:
        seg=x[int(a*sr):int(b*sr)]
        if len(seg)==0: continue
        band, cent=band_energy_fft(seg,sr,bands)
        l=seg[:,0]; r=seg[:,1]
        mid=(l+r)*0.5; side=(l-r)*0.5
        out.append({
            'start':a,'end':b,
            'rms_db': db(rms(seg)),
            'peak_dbfs': db(np.max(np.abs(seg))),
            'crest_db': db(np.max(np.abs(seg))/(rms(seg)+1e-12)),
            'corr': float(np.corrcoef(l,r)[0,1]),
            'side_mid_ratio': float(rms(side)/(rms(mid)+1e-12)),
            'centroid_hz': cent,
            'bands': band,
        })
    return out


def onset_tempo_crude(x,sr):
    mono=x.mean(axis=1)
    hop=512
    win=2048
    vals=[]
    prev=None
    for s in range(0,len(mono)-win,hop):
        spec=np.abs(np.fft.rfft(mono[s:s+win]*np.hanning(win)))
        if prev is None:
            vals.append(0)
        else:
            vals.append(float(np.sum(np.maximum(spec-prev,0))))
        prev=spec
    env=np.array(vals); env=(env-env.mean())/(env.std()+1e-12)
    # autocorr tempo 60-180 bpm
    ac=np.correlate(env,env,mode='full')[len(env)-1:]
    times=np.arange(len(ac))*hop/sr
    m=(times>=60/180)&(times<=60/60)
    lag=times[m][np.argmax(ac[m])]
    bpm=60/lag if lag else None
    return float(bpm)

x,sr=read_wav(PATH)
summary={
 'file': str(PATH),
 'sample_rate': sr,
 'duration_s': len(x)/sr,
 'peak_dbfs': db(np.max(np.abs(x))),
 'rms_dbfs': db(rms(x)),
 'crest_db': db(np.max(np.abs(x))/(rms(x)+1e-12)),
 'rotation': estimate_rotation_from_pan(x,sr),
 'tempo_bpm_crude': onset_tempo_crude(x,sr),
 'sections': section_stats(x,sr),
}
print(json.dumps(summary, indent=2))
