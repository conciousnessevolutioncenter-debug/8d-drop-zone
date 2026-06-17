"""Subscription tiers and feature gating — the single source of truth.

Maps the product spec (Free / Creator / Producer / Studio Master) onto the app's
*real* capabilities. Capabilities that don't exist yet (text->audio generation)
are intentionally omitted. Stripe (the payments phase) flips `user.tier` via a
webhook; everything here just reads it, so the gates work before billing is live.
"""
from __future__ import annotations

from .models import User

# Capability keys (only ones the app can actually do):
#   analyze   – Listen/Watch/Transcribe (media analysis)
#   download  – download processed/rendered audio
#   mix_basic – volume / fades
#   mix_full  – EQ, compression, reverb, multi-track
#   eight_d   – the 8D spatial editor (your dsp presets)
#   stems     – AI stem separation (Demucs / HF Space)
TIERS = {
    "free":     {"label": "Free",          "model": "capped",  "caps": {"analyze"}},
    "creator":  {"label": "Creator",       "model": "credits", "caps": {"analyze", "download", "mix_basic"}},
    "producer": {"label": "Producer",      "model": "credits", "caps": {"analyze", "download", "mix_basic", "mix_full"}},
    "studio":   {"label": "Studio Master", "model": "credits", "caps": {"analyze", "download", "mix_basic", "mix_full", "eight_d", "stems"}},
}
ORDER = ["free", "creator", "producer", "studio"]
FREE_GENERATION_CAP = 5  # spec: free users get a small monthly analysis/gen cap


def tier_of(user: User | None) -> str:
    return (user.tier if user and user.tier in TIERS else "free")


def label(user: User | None) -> str:
    return TIERS[tier_of(user)]["label"]


def is_paid(user: User | None) -> bool:
    return tier_of(user) != "free"


def can(user: User | None, capability: str) -> bool:
    """Does this user's tier grant `capability`?"""
    return capability in TIERS[tier_of(user)]["caps"]


def caps(user: User | None) -> set[str]:
    return set(TIERS[tier_of(user)]["caps"])


def set_tier(user: User, tier: str) -> None:
    """Apply a tier (called by the Stripe webhook in the payments phase, or the
    dev tool). Keeps the legacy `is_pro` flag in sync for the existing badge UI."""
    if tier not in TIERS:
        tier = "free"
    user.tier = tier
    user.is_pro = tier != "free"


def consume_credits(user: User, n: int = 1) -> bool:
    """Deduct credits for a billable task; returns False if insufficient.
    Free tier isn't credit-based, so it always passes here (gating is by `can`)."""
    if tier_of(user) == "free":
        return True
    if user.credits < n:
        return False
    user.credits -= n
    return True
