"""AI mix co-producer — turn a plain-English vibe into concrete mixer moves.

Free by default via Groq (fast Llama models, free API key, no card). Falls back
to Claude if ANTHROPIC_API_KEY is set instead. Dormant + friendly 503 when no
provider key is present. Either way returns the same JSON the browser applies
straight to the channel strips. No new dependency — uses ``requests``.
"""
from __future__ import annotations

import json
import os

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")


def available() -> bool:
    return bool(os.environ.get("GROQ_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))


def provider() -> str:
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "none"


_SYSTEM = (
    "You are a world-class spatial-audio mix engineer working inside The 8D Engine. "
    "Given a producer's vibe description and the list of available stems, return concrete, tasteful mix moves. "
    "Only address stems from the provided list, using their exact names. Keep everything musical and in range: "
    "prefer subtle EQ (a few dB), keep lead vocals near center, keep bass/kick mono and centered, and let pads, "
    "synths and FX go wider. Choose an orbit speed that matches the energy. Always include a short, warm producer's note."
)

# Shared output contract. dB fields roughly -12..+12 (gain_db -12..+6), pan -1..+1.
_JSON_SHAPE = (
    '{"channels":[{"stem":"<exact name>","gain_db":<number>,"pan":<number -1..1>,'
    '"eq_low_db":<number>,"eq_mid_db":<number>,"eq_high_db":<number>,"mute":<true|false>}],'
    '"orbit":"still|slow|medium|fast","notes":"<one or two sentences>"}'
)

# Claude uses a tool to force structured output.
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
            "notes": {"type": "string", "description": "One or two first-person sentences explaining the moves."},
        },
        "required": ["channels", "notes"],
    },
}


def _normalize(data: dict) -> dict:
    """Coerce a model response into the directive shape the browser expects."""
    if not isinstance(data, dict):
        raise RuntimeError("Model did not return an object.")
    chans = data.get("channels")
    if not isinstance(chans, list):
        chans = []
    out = {"channels": [c for c in chans if isinstance(c, dict) and c.get("stem")],
           "notes": str(data.get("notes") or "Mix applied.")}
    if data.get("orbit") in ("still", "slow", "medium", "fast"):
        out["orbit"] = data["orbit"]
    return out


def _user_msg(prompt: str, stems: list[str]) -> str:
    stem_list = ", ".join(s for s in stems if s) or "vocals, drums, bass, other"
    return f"Available stems: {stem_list}.\n\nVibe: {prompt}"


def _groq_mix(prompt: str, stems: list[str], timeout: int) -> dict:
    import requests
    key = os.environ["GROQ_API_KEY"]
    body = {
        "model": GROQ_MODEL,
        "temperature": 0.6,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": _SYSTEM + " Respond ONLY with a JSON object of exactly this shape: " + _JSON_SHAPE},
            {"role": "user", "content": _user_msg(prompt, stems)},
        ],
    }
    r = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {key}", "content-type": "application/json"},
                      json=body, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Groq API error {r.status_code}: {r.text[:300]}")
    content = r.json()["choices"][0]["message"]["content"]
    return _normalize(json.loads(content))


def _anthropic_mix(prompt: str, stems: list[str], timeout: int) -> dict:
    import requests
    key = os.environ["ANTHROPIC_API_KEY"]
    body = {
        "model": ANTHROPIC_MODEL, "max_tokens": 1024, "system": _SYSTEM,
        "tools": [_TOOL], "tool_choice": {"type": "tool", "name": "set_mix"},
        "messages": [{"role": "user", "content": _user_msg(prompt, stems)}],
    }
    r = requests.post(ANTHROPIC_URL,
                      headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                      json=body, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Claude API error {r.status_code}: {r.text[:300]}")
    for block in r.json().get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == "set_mix":
            return _normalize(block.get("input", {}))
    raise RuntimeError("No structured mix was returned.")


def suggest_mix(prompt: str, stems: list[str], timeout: int = 45) -> dict:
    """Dispatch to the configured provider (Groq preferred, then Claude)."""
    p = provider()
    if p == "groq":
        return _groq_mix(prompt, stems, timeout)
    if p == "anthropic":
        return _anthropic_mix(prompt, stems, timeout)
    raise RuntimeError("No AI provider configured (set GROQ_API_KEY).")
