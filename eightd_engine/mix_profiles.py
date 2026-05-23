"""Reference mix profiles for The 8D Engine.

Profiles translate analyzed reference tracks into reproducible app settings.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MixProfile:
    name: str
    rotation_cpm: float
    room_size: float
    crossover_hz: float
    description: str


FIREFLIES_8D_REFERENCE = MixProfile(
    name="Fireflies 8D Reference",
    # Measured dominant pan modulation: 0.09628 Hz = 10.386 s/cycle.
    rotation_cpm=5.78,
    room_size=0.18,
    crossover_hz=150.0,
    description=(
        "Smooth YouTube-style 8D orbit based on Owl City - Fireflies (8D AUDIO): "
        "~10.4 seconds per rotation, wide but not frantic motion, light room support, "
        "and static/protected low-end foundation."
    ),
)


PROFILES = {
    FIREFLIES_8D_REFERENCE.name: FIREFLIES_8D_REFERENCE,
}
