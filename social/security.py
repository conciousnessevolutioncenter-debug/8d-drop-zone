"""Password hashing + the session-based current-user dependency."""
from __future__ import annotations

import bcrypt
from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from .db import get_db
from .models import User


def hash_password(raw: str) -> str:
    # bcrypt hard-caps the input at 72 bytes; truncate so long passwords don't error.
    return bcrypt.hashpw(raw.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(raw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(raw.encode("utf-8")[:72], hashed.encode("utf-8"))
    except Exception:
        return False


def login_session(request: Request, user: User) -> None:
    request.session["uid"] = user.id


def logout_session(request: Request) -> None:
    request.session.pop("uid", None)


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    """Require a logged-in user; 401 otherwise."""
    uid = request.session.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="Not signed in")
    user = db.get(User, uid)
    if not user:
        request.session.pop("uid", None)
        raise HTTPException(status_code=401, detail="Session expired")
    return user


def optional_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    uid = request.session.get("uid")
    return db.get(User, uid) if uid else None
