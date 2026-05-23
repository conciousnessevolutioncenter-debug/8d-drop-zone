"""Generative music planning primitives for the helmet-album direction.

This module intentionally stays small for v0.1. The voice-note concept points
beyond static file conversion toward a musical system with rules:
- probability, constraints, and scale-aware note choices
- head-motion or switches selecting scenes/songs
- multiple helmets syncing clocks over a local network

The desktop app does not render full generative compositions yet, but these
helpers establish the musical-rule layer we can expand into a Raspberry Pi build.
"""

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class GenerativeRuleSet:
    """Musical constraints for rule-based phrase generation."""

    root_midi: int = 60  # C4
    scale_intervals: tuple[int, ...] = (0, 2, 3, 5, 7, 8, 10)  # natural minor
    octave_span: int = 2
    step_probability: float = 0.72
    rest_probability: float = 0.12


def generate_melodic_degrees(
    length: int, rules: GenerativeRuleSet, seed: int | None = None
) -> list[int | None]:
    """Generate scale-degree melody data with musical continuity.

    Returns MIDI notes or None for rests. The algorithm favors stepwise motion
    while occasionally allowing leaps, preventing pure-random musical blandness.
    """

    rng = random.Random(seed)
    scale_len = len(rules.scale_intervals)
    degree = rng.randrange(scale_len * rules.octave_span)
    melody: list[int | None] = []

    for _ in range(length):
        if rng.random() < rules.rest_probability:
            melody.append(None)
            continue

        octave, scale_index = divmod(degree, scale_len)
        midi = rules.root_midi + 12 * octave + rules.scale_intervals[scale_index]
        melody.append(midi)

        if rng.random() < rules.step_probability:
            degree += rng.choice([-1, 1])
        else:
            degree += rng.choice([-4, -3, 3, 4, 5])
        degree = max(0, min(scale_len * rules.octave_span - 1, degree))

    return melody


def tempo_sync_period_ms(bpm: float, bars: float = 1.0, beats_per_bar: int = 4) -> float:
    """Return milliseconds for a musically meaningful sync period."""

    if bpm <= 0:
        raise ValueError("bpm must be positive")
    return (60_000.0 / bpm) * beats_per_bar * bars
