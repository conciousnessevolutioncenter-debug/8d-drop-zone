"""Single source of truth for the signing secret.

The same secret signs session cookies *and* the password-reset / email-verify
tokens, so a placeholder value in production is an account-takeover vector:
the default below is public in this repository, and anyone can use it to mint
a valid session for any account. Resolution therefore fails loudly instead of
falling back, and web_app resolves it *outside* its guarded social imports so
a misconfigured deploy stops the process rather than quietly serving a
signed-out app.
"""
from __future__ import annotations

import os

INSECURE_DEFAULT = "dev-insecure-change-me"


def session_secret() -> str:
    """Return the signing secret, or raise if it would be guessable.

    Set SOCIAL_DEV=1 to opt into the placeholder for local development.
    """
    secret = os.environ.get("SESSION_SECRET", "")
    if secret and secret != INSECURE_DEFAULT:
        return secret
    if os.environ.get("SOCIAL_DEV"):
        return INSECURE_DEFAULT
    raise RuntimeError(
        "SESSION_SECRET is unset or still the placeholder value. It signs "
        "session cookies and password-reset tokens, so leaving it at the "
        "default lets anyone forge a login for any account. Generate one with "
        "`python -c \"import secrets; print(secrets.token_urlsafe(48))\"` and "
        "set it in the deployment environment, or set SOCIAL_DEV=1 for local "
        "development."
    )
