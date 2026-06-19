"""AI mix co-producer — turn a plain-English vibe into concrete mixer moves.

Env-gated on ANTHROPIC_API_KEY (like Stripe/SMTP): dormant + friendly 503 when
unset. Calls the Claude Messages API with a tool schema so the model returns
strict JSON directives the browser applies straight to the channel strips.
No new dependency — uses ``requests`` (already pulled in by the stack).
"""
from __future__ import annotations

import os

API_URL = "https://api.anthropic.com/v1/messages"
# Fast + cheap is plenty for structured mix directives; override via env.
DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


_TOOL = {
    "name": "set_mix",
    "description": "Apply a spatial mix: per-stem level, pan, 3-band EQ, and a global 8D orbit feel.",
    "input_schema": {
        "type": "object",
        "properties": {
            "channels": {
                "type": "array",
                "description": "One entry per stem you want to move. Omit stems you'd leave untouched.",
                "items": {
                    "type": "object",
                    "properties": {
                        "stem": {"type": "string", "description": "Stem/channel name — match one from the provided list exactly."},
                        "gain_db": {"type": "number", "description": "Level change in dB, roughly -12 to +6."},
                        "pan": {"type": "number", "description": "-1 hard left, 0 center, +1 hard right."},
                        "eq_low_db": {"type": "number", "description": "Low shelf ~200 Hz, -12 to +12 dB."},
                        "eq_mid_db": {"type": "number", "description": "Mid bell ~1 kHz, -12 to +12 dB."},
                        "eq_high_db": {"type": "number", "description": "High shelf ~4 kHz, -12 to +12 dB."},
                        "mute": {"type": "boolean"},
                    },
                    "required": ["stem"],
                },
            },
            "orbit": {"type": "string", "enum": ["still", "slow", "medium", "fast"],
                      "description": "How fast the 8D orbit should rotate for this vibe."},
            "notes": {"type": "string", "description": "One or two first-person sentences explaining the moves, like a producer talking to an artist."},
        },
        "required": ["channels", "notes"],
    },
}

_SYSTEM = (
    "You are a world-class spatial-audio mix engineer working inside The 8D Engine. "
    "Given a producer's vibe description and the list of available stems, return concrete, tasteful mix moves "
    "with the set_mix tool. Only address stems from the provided list, using their exact names. Keep everything "
    "musical and in range: prefer subtle EQ (a few dB), keep lead vocals near center, keep bass/kick mono and "
    "centered, and let pads, synths and FX go wider. Choose an orbit speed that matches the energy. "
    "Always include a short, warm producer's note."
)


def suggest_mix(prompt: str, stems: list[str], timeout: int = 45) -> dict:
    """Call Claude and return the structured ``set_mix`` directives dict."""
    import requests

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    stem_list = ", ".join(s for s in stems if s) or "vocals, drums, bass, other"
    body = {
        "model": DEFAULT_MODEL,
        "max_tokens": 1024,
        "system": _SYSTEM,
        "tools": [_TOOL],
        "tool_choice": {"type": "tool", "name": "set_mix"},
        "messages": [{"role": "user", "content": f"Available stems: {stem_list}.\n\nVibe: {prompt}"}],
    }
    r = requests.post(
        API_URL,
        headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json=body, timeout=timeout,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Claude API error {r.status_code}: {r.text[:300]}")
    for block in r.json().get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == "set_mix":
            return block.get("input", {})
    raise RuntimeError("No structured mix was returned.")
