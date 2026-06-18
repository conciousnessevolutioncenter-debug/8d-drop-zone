"""Tiny in-process rate limiter (fixed window per IP+bucket).

Single-instance; for multi-instance scale, back it with Redis. Good enough to
blunt brute-force logins, signup spam, and posting floods. Behind the Vercel→
Railway hop the real client IP is in X-Forwarded-For.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict

from fastapi import HTTPException, Request

_hits: dict[str, list[float]] = defaultdict(list)
_lock = threading.Lock()


def client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request, bucket: str, limit: int, per_seconds: int) -> None:
    """Allow `limit` requests per `per_seconds` for this IP+bucket, else 429."""
    now = time.time()
    cutoff = now - per_seconds
    key = f"{bucket}:{client_ip(request)}"
    with _lock:
        q = _hits[key]
        drop = 0
        while drop < len(q) and q[drop] < cutoff:
            drop += 1
        if drop:
            del q[:drop]
        if len(q) >= limit:
            raise HTTPException(status_code=429, detail="Too many requests — please slow down and try again shortly.")
        q.append(now)
