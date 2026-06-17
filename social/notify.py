"""Notification helpers — kept dependency-light (models only) so any route or the
WebSocket handlers can create one without circular imports."""
from __future__ import annotations

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from .models import Notification


def create_notification(db: Session, user_id: int, kind: str, text: str, link: str = "") -> None:
    """Add a notification for `user_id`. Caller owns the commit."""
    db.add(Notification(user_id=user_id, kind=kind, text=text[:200], link=link[:200]))


def unread_count(db: Session, user_id: int) -> int:
    return db.scalar(
        select(func.count()).select_from(Notification).where(
            Notification.user_id == user_id, Notification.read.is_(False)
        )
    ) or 0
