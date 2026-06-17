"""ORM models for the social layer (Phase 1 + feed; chat/payments added later)."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship

from .db import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    handle = Column(String(40), unique=True, index=True, nullable=False)
    display_name = Column(String(80), nullable=False, default="")
    bio = Column(String(280), nullable=False, default="")
    avatar_url = Column(String(500), nullable=False, default="")
    is_pro = Column(Boolean, nullable=False, default=False)   # derived: tier != free
    tier = Column(String(20), nullable=False, default="free")  # free | creator | producer | studio
    credits = Column(Integer, nullable=False, default=0)
    stripe_customer_id = Column(String(80), nullable=True)  # reserved for payments phase
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)

    posts = relationship("Post", back_populates="author", cascade="all, delete-orphan", foreign_keys="Post.author_id")


class Follow(Base):
    __tablename__ = "follows"
    follower_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    followee_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)


class Post(Base):
    __tablename__ = "posts"
    id = Column(Integer, primary_key=True)
    author_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    body = Column(Text, nullable=False, default="")
    image_url = Column(String(500), nullable=True)
    track_job_id = Column(String(40), nullable=True)          # attach a rendered 8D master
    repost_of = Column(Integer, ForeignKey("posts.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now, index=True)

    author = relationship("User", back_populates="posts", foreign_keys=[author_id])
    comments = relationship("Comment", back_populates="post", cascade="all, delete-orphan")


class Like(Base):
    __tablename__ = "likes"
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    post_id = Column(Integer, ForeignKey("posts.id", ondelete="CASCADE"), primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)


class Comment(Base):
    __tablename__ = "comments"
    id = Column(Integer, primary_key=True)
    post_id = Column(Integer, ForeignKey("posts.id", ondelete="CASCADE"), nullable=False, index=True)
    author_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    body = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)

    post = relationship("Post", back_populates="comments")


Index("ix_follow_followee", Follow.followee_id)


# ---- Realtime: rooms, messages, DMs, read receipts, moderation ----

class Room(Base):
    __tablename__ = "rooms"
    id = Column(Integer, primary_key=True)
    slug = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(80), nullable=False)
    topic = Column(String(160), nullable=False, default="")
    is_private = Column(Boolean, nullable=False, default=False)
    pro_only = Column(Boolean, nullable=False, default=False)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)


class DmThread(Base):
    __tablename__ = "dm_threads"
    id = Column(Integer, primary_key=True)
    user_a = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)  # always the lower id
    user_b = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    __table_args__ = (UniqueConstraint("user_a", "user_b", name="uq_dm_pair"),)


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    room_id = Column(Integer, ForeignKey("rooms.id", ondelete="CASCADE"), nullable=True, index=True)
    thread_id = Column(Integer, ForeignKey("dm_threads.id", ondelete="CASCADE"), nullable=True, index=True)
    sender_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    body = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now, index=True)


class DmRead(Base):
    __tablename__ = "dm_reads"
    thread_id = Column(Integer, ForeignKey("dm_threads.id", ondelete="CASCADE"), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    last_read_message_id = Column(Integer, nullable=False, default=0)


class Block(Base):
    __tablename__ = "blocks"
    blocker_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    blocked_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)


class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True)
    reporter_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    target_type = Column(String(20), nullable=False)   # 'user' | 'message' | 'post'
    target_id = Column(Integer, nullable=False)
    reason = Column(String(280), nullable=False, default="")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
