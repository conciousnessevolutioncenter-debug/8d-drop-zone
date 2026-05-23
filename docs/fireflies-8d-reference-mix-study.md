# Fireflies 8D Reference Mix Study

Reference URL: https://youtu.be/uhjJqClcUkY

Video title: **Owl City - Fireflies (8D AUDIO)**

Downloaded/analyzed file: `reference/fireflies_8d_reference.wav`

## Technical measurements

- Duration: **228.98 s** / **3:49**
- Sample rate: **48 kHz**
- Channels: **stereo**
- Integrated loudness: **-12.36 LUFS**
- True peak from loudnorm scan: **+0.59 dBTP**
- Loudness range: **16.60 LU**
- Peak sample level after WAV extraction: approximately **0 dBFS**
- Overall RMS: approximately **-16.74 dBFS**
- Crude tempo estimate: **~90.7 BPM**

## Motion behavior

The defining trait is a slow, comfortable, full-song horizontal orbit.

Measured from sliding left/right energy:

- Dominant pan modulation: **0.09628 Hz**
- Rotation period: **~10.39 seconds per cycle**
- Equivalent app setting: **~5.78 cycles per minute**
- Pan energy range: about **-0.70 to +0.70**

Interpretation:

- The movement is obvious, but not frantic.
- It does not use a fast “spinning fan” effect.
- The track breathes because one complete rotation takes roughly ten seconds.
- This is close to a musical 4-bar orbit at ~92 BPM.

## Stereo image

Measured wide-image characteristics:

- Mean side/mid ratio: **~0.86**
- 90th percentile side/mid ratio: **~1.22**
- Correlation trends lower in bigger sections, indicating stronger spatial widening.

Mix interpretation:

- The whole track has a strong binaural/stereo manipulation layer.
- The motion is not just left/right autopan; there is significant side energy.
- The image is intentionally wide and headphone-first.

## Bass behavior

Measured low-band side/mid ratio below 150 Hz:

- Mean: **~0.20**
- 90th percentile: **~0.35**

Interpretation:

- The reference does allow some low-band stereo/phase movement.
- For our app, we should improve on this by keeping sub/kick more protected and mono.
- Use the Fireflies motion speed and wide image as the reference, but keep our professional bass-management rule.

App decision:

- Default crossover remains **150 Hz**.
- Bass below crossover remains **static mono center**.
- Mid/high motion follows the reference orbit speed.

## Frequency distribution by arrangement

The track changes spectral density through sections:

- Intro/early sections are more mid-focused and sparse.
- Bigger sections bring more upper-mid/presence energy.
- The outro becomes very side-heavy and mid/presence-focused, with minimal true sub.

Important reference lesson:

- The 8D effect works best when the listener has bright, harmonically rich material to track spatially.
- Motion should be strongest in the mid/high band, especially the presence/air/reverb/delay field.
- Do not rely on sub-bass to communicate spatial movement.

## App preset encoded

Created:

`eightd_engine/mix_profiles.py`

Preset:

```python
FIREFLIES_8D_REFERENCE = MixProfile(
    name="Fireflies 8D Reference",
    rotation_cpm=5.78,
    room_size=0.18,
    crossover_hz=150.0,
)
```

GUI changes:

- The app now defaults to the Fireflies reference speed.
- Added **Apply Fireflies 8D Reference Preset** button.
- The preset sets:
  - Rotation Speed: **5.78 cycles/minute**
  - Room Size: **0.18**
  - Bass Crossover: **150 Hz**

## Mix rules for The 8D Engine based on this reference

1. **Orbit speed:** default to ~10.4 seconds per full rotation.
2. **Motion curve:** smooth circular/sine-like movement, no hard jumps.
3. **Bass:** keep 20–150 Hz mono and centered, even if the reference has some low-band side movement.
4. **Motion band:** spatialize 150 Hz+ material.
5. **Room:** add subtle room support; do not drown the direct signal.
6. **Brightness:** upper mids and highs should carry much of the perceived spatial motion.
7. **Comfort:** avoid rotation speeds faster than ~8 seconds/cycle unless intentionally extreme.
8. **Loudness:** keep output peak-safe; the reference is hot, but app exports should avoid intersample clipping.

## Next DSP improvements suggested by the reference

- Add a **motion depth** control to scale pan/ITD intensity.
- Add a **high-frequency motion emphasis** option so 4 kHz+ can move more than low mids.
- Add **section automation** so intros move slower/subtler and choruses widen more.
- Add **phase/correlation meter** in the GUI.
- Add a **true peak/loudness normalization** export stage.
- Add optional “YouTube 8D Hot Master” export target around **-12 to -14 LUFS**, -1 dBTP ceiling.
