# Helmet Album Manifesto Notes

Captured from Christ's voice notes and translated into product requirements.

## Core insight

Generative music is not just AI making songs. It is music created from rules: probability, math, constraints, logic, gates, and controlled randomness. The point is not bland random output. The system must still create music that *affects* the listener emotionally.

## Max/MSP inspiration

Max/MSP works like modular logic:

- Objects have inputs/outputs.
- Some objects have attributes.
- Wires route control/audio data.
- Gates decide whether signals pass.
- Math transforms musical input.
- Small rules can create rich behavior, e.g. `+7` turns one key press into a fifth.

Future app implication: build a patch/rule layer where music can be manipulated as math and logic, not only as fixed audio files.

## Physical album idea

The album is a helmet:

- Built-in headphones.
- Small computer, likely Raspberry Pi.
- Front switches select song/scene.
- Different artist-designed helmet variants:
  - Viking helmet
  - Mohawk helmet
  - Alien helmet
- The album only fully works when worn, powered on, and moved.

## Motion interaction

Head movement should matter. The listener does not only press play; they perform the album by moving their head.

Possible mappings:

- Yaw -> spatial azimuth/orbit offset
- Pitch -> brightness/reverb/density
- Roll -> modulation depth/glitch/scene morph
- Stillness -> sparse/ambient behavior
- Fast motion -> percussion or intensity changes

## Swarm concept

If two or more helmets are in the same room, they should detect each other through Wi‑Fi and sync musically.

Design questions:

- Who is clock master?
- How much latency is acceptable?
- Should helmets play the same composition or complementary parts?
- How large can the swarm get while staying musical?
- How do devices quantize actions to shared beats/bars?

## Practical architecture

Desktop app now:

- Converts/imports/exports audio.
- Builds the 8D DSP core.
- Provides the first generative helper functions.

Helmet runtime later:

- Headless Python service on Raspberry Pi.
- Audio engine using pre-rendered loops, generated MIDI, or real-time synthesis.
- IMU sensor reader.
- Physical switch reader.
- OSC/UDP swarm sync.
- Local config per helmet scene.

## Principle

The goal is an emotionally powerful generative instrument-album, not a novelty gadget. The constraints must be musical enough that the output feels intentional, beautiful, and alive.
