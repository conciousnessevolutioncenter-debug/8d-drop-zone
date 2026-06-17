"""Email sending + signed password-reset tokens.

Tokens are stateless: signed with SESSION_SECRET via itsdangerous (no DB row to
store/expire). Email goes out over SMTP when configured (works with Resend/
SendGrid/Mailgun/Gmail SMTP); if SMTP isn't set, the message is logged so dev
(and you, pre-provider) can still grab the reset link from the server logs.
"""
from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

_RESET_SALT = "8d-password-reset"


def _serializer() -> URLSafeTimedSerializer:
    secret = os.environ.get("SESSION_SECRET", "dev-insecure-change-me")
    return URLSafeTimedSerializer(secret, salt=_RESET_SALT)


def make_reset_token(user_id: int) -> str:
    return _serializer().dumps({"uid": int(user_id)})


def read_reset_token(token: str, max_age_seconds: int = 3600) -> int | None:
    try:
        data = _serializer().loads(token, max_age=max_age_seconds)
        return int(data["uid"])
    except (BadSignature, SignatureExpired, KeyError, ValueError, TypeError):
        return None


def smtp_configured() -> bool:
    return bool(os.environ.get("SMTP_HOST"))


def send_email(to_addr: str, subject: str, body_text: str) -> bool:
    """Send an email via SMTP if configured; otherwise log it and return False."""
    host = os.environ.get("SMTP_HOST")
    if not host:
        print(f"[mailer] (no SMTP configured) email to {to_addr}: {subject}\n{body_text}", flush=True)
        return False
    msg = EmailMessage()
    msg["From"] = os.environ.get("SMTP_FROM", os.environ.get("SMTP_USER", "no-reply@8dengine.app"))
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body_text)
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASS")
    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=20) as s:
                if user:
                    s.login(user, password)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=20) as s:
                s.starttls()
                if user:
                    s.login(user, password)
                s.send_message(msg)
        return True
    except Exception as exc:  # pragma: no cover - network/provider specific
        print(f"[mailer] send failed to {to_addr}: {exc}", flush=True)
        return False
