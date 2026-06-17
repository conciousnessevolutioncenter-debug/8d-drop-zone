"""Social routes: auth, profiles, feed, follow. Server-rendered (PRG pattern),
mounted under /social so the audio app routes are untouched."""
from __future__ import annotations

import re

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select, func, delete
from sqlalchemy.orm import Session

from .db import get_db
from .models import User, Post, Like, Follow, Comment
from .security import hash_password, verify_password, login_session, logout_session, optional_user, current_user
from .ui import layout, esc

router = APIRouter(prefix="/social", tags=["social"])
_HANDLE_RE = re.compile(r"^[a-z0-9_]{3,30}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _counts(db: Session, user: User):
    followers = db.scalar(select(func.count()).select_from(Follow).where(Follow.followee_id == user.id))
    following = db.scalar(select(func.count()).select_from(Follow).where(Follow.follower_id == user.id))
    posts = db.scalar(select(func.count()).select_from(Post).where(Post.author_id == user.id))
    return followers or 0, following or 0, posts or 0


def _post_card(db: Session, post: Post, viewer: User | None) -> str:
    a = post.author
    likes = db.scalar(select(func.count()).select_from(Like).where(Like.post_id == post.id)) or 0
    ncomments = db.scalar(select(func.count()).select_from(Comment).where(Comment.post_id == post.id)) or 0
    pro = '<span class="pro" style="margin-left:6px">PRO</span>' if a.is_pro else ""
    track = ""
    if post.track_job_id:
        track = ('<div class="row card" style="border-color:rgba(98,224,255,.25);background:rgba(98,224,255,.05);margin-top:10px;padding:9px 11px">'
                 '<span class="tag" style="color:var(--cyan)">8D TRACK ATTACHED</span></div>')
    when = post.created_at.strftime("%b %d, %H:%M") if post.created_at else ""
    like_link = f'<a href="/social/posts/{post.id}/like" onclick="event.preventDefault();fetch(this.href,{{method:\'POST\'}}).then(()=>location.reload())">♥ {likes}</a>' if viewer else f'♥ {likes}'
    return f"""<div class="card">
      <div class="row"><span class="avatar"></span>
        <div><div style="font-size:13px;font-weight:500">{esc(a.display_name or a.handle)}{pro}</div>
        <a href="/social/u/{esc(a.handle)}" class="handle">@{esc(a.handle)} · {when}</a></div></div>
      <p style="margin:10px 0 0;font-size:14px;line-height:1.55;color:#cdd7ea">{esc(post.body)}</p>
      {track}
      <div class="post-actions">{like_link}<span>💬 {ncomments}</span><span>↻ repost</span></div>
    </div>"""


@router.get("/", response_class=HTMLResponse)
def feed(request: Request, db: Session = Depends(get_db)):
    user = optional_user(request, db)
    if not user:
        body = """<h1>The 8D Engine, now social.</h1>
        <p class="lede">Share spatial masters, join the lobby, follow other creators.</p>
        <div class="card"><a href="/social/register" class="btn">Create account</a>
        <a href="/social/login" class="btn ghost" style="margin-left:8px">Sign in</a></div>"""
        return HTMLResponse(layout("Welcome", body, None, "feed"))
    # timeline: own posts + people you follow, newest first
    followed = select(Follow.followee_id).where(Follow.follower_id == user.id)
    rows = db.scalars(
        select(Post).where((Post.author_id == user.id) | (Post.author_id.in_(followed)))
        .order_by(Post.created_at.desc()).limit(50)
    ).all()
    composer = """<div class="card"><form method="post" action="/social/posts">
      <textarea name="body" placeholder="Share a track or a thought…" maxlength="2000"></textarea>
      <button class="btn" type="submit">Post</button></form></div>"""
    feed_html = "".join(_post_card(db, p, user) for p in rows) or '<p class="muted" style="margin-top:16px">Your feed is quiet — follow some creators or post something.</p>'
    return HTMLResponse(layout("Feed", composer + feed_html, user, "feed"))


@router.get("/register", response_class=HTMLResponse)
def register_page(request: Request, db: Session = Depends(get_db)):
    if optional_user(request, db):
        return RedirectResponse("/social", status_code=303)
    body = """<h1>Create your account</h1>
    <div class="card"><form method="post" action="/social/register">
      <label>Display name</label><input name="display_name" maxlength="80" required>
      <label>Handle</label><input name="handle" placeholder="lowercase, 3–30" required>
      <label>Email</label><input name="email" type="email" required>
      <label>Password</label><input name="password" type="password" minlength="8" required>
      <button class="btn" type="submit">Join</button>
    </form></div><p class="muted" style="margin-top:12px">Already have one? <a href="/social/login" style="color:var(--cyan)">Sign in</a></p>"""
    return HTMLResponse(layout("Register", body, None))


@router.post("/register")
def register(request: Request, db: Session = Depends(get_db),
             display_name: str = Form(...), handle: str = Form(...),
             email: str = Form(...), password: str = Form(...)):
    handle = handle.strip().lower(); email = email.strip().lower()
    err = None
    if not _HANDLE_RE.match(handle): err = "Handle must be 3–30 chars: a–z, 0–9, underscore."
    elif not _EMAIL_RE.match(email): err = "Enter a valid email."
    elif len(password) < 8: err = "Password must be at least 8 characters."
    elif db.scalar(select(User).where(User.email == email)): err = "That email is already registered."
    elif db.scalar(select(User).where(User.handle == handle)): err = "That handle is taken."
    if err:
        body = f'<h1>Create your account</h1><div class="card"><p class="err">{esc(err)}</p>' \
               f'<a href="/social/register" class="btn ghost">Back</a></div>'
        return HTMLResponse(layout("Register", body, None), status_code=400)
    user = User(email=email, handle=handle, display_name=display_name.strip() or handle,
                password_hash=hash_password(password))
    db.add(user); db.commit(); db.refresh(user)
    login_session(request, user)
    return RedirectResponse("/social", status_code=303)


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request, db: Session = Depends(get_db)):
    if optional_user(request, db):
        return RedirectResponse("/social", status_code=303)
    body = """<h1>Sign in</h1><div class="card"><form method="post" action="/social/login">
      <label>Email</label><input name="email" type="email" required>
      <label>Password</label><input name="password" type="password" required>
      <button class="btn" type="submit">Sign in</button></form></div>
    <p class="muted" style="margin-top:12px">New here? <a href="/social/register" style="color:var(--cyan)">Create an account</a></p>"""
    return HTMLResponse(layout("Sign in", body, None))


@router.post("/login")
def login(request: Request, db: Session = Depends(get_db),
          email: str = Form(...), password: str = Form(...)):
    user = db.scalar(select(User).where(User.email == email.strip().lower()))
    if not user or not verify_password(password, user.password_hash):
        body = '<h1>Sign in</h1><div class="card"><p class="err">Wrong email or password.</p>' \
               '<a href="/social/login" class="btn ghost">Try again</a></div>'
        return HTMLResponse(layout("Sign in", body, None), status_code=401)
    login_session(request, user)
    return RedirectResponse("/social", status_code=303)


@router.get("/logout")
def logout(request: Request):
    logout_session(request)
    return RedirectResponse("/social", status_code=303)


@router.post("/posts")
def create_post(request: Request, db: Session = Depends(get_db),
                body: str = Form(""), track_job_id: str = Form(None),
                user: User = Depends(current_user)):
    text = (body or "").strip()
    if text or track_job_id:
        db.add(Post(author_id=user.id, body=text[:2000], track_job_id=track_job_id or None))
        db.commit()
    return RedirectResponse("/social", status_code=303)


@router.post("/posts/{post_id}/like")
def toggle_like(post_id: int, request: Request, db: Session = Depends(get_db),
                user: User = Depends(current_user)):
    existing = db.get(Like, {"user_id": user.id, "post_id": post_id})
    if existing:
        db.execute(delete(Like).where(Like.user_id == user.id, Like.post_id == post_id))
    elif db.get(Post, post_id):
        db.add(Like(user_id=user.id, post_id=post_id))
    db.commit()
    return {"ok": True}


@router.get("/u/{handle}", response_class=HTMLResponse)
def profile(handle: str, request: Request, db: Session = Depends(get_db)):
    viewer = optional_user(request, db)
    u = db.scalar(select(User).where(User.handle == handle.lower()))
    if not u:
        return HTMLResponse(layout("Not found", '<h1>No such user</h1>', viewer), status_code=404)
    followers, following, nposts = _counts(db, u)
    is_me = viewer and viewer.id == u.id
    is_following = bool(viewer and db.get(Follow, {"follower_id": viewer.id, "followee_id": u.id}))
    if is_me:
        action = '<a href="/social/settings" class="btn ghost">Edit profile</a>'
    elif viewer:
        verb, path = ("Unfollow", "unfollow") if is_following else ("Follow", "follow")
        action = (f'<form method="post" action="/social/{path}/{esc(u.handle)}" style="display:inline">'
                  f'<button class="btn" type="submit">{verb}</button></form>'
                  f'<a href="/social/dm/{esc(u.handle)}" class="btn ghost" style="margin-left:8px">Message</a>')
    else:
        action = '<a href="/social/login" class="btn">Sign in to follow</a>'
    pro = '<span class="pro" style="margin-left:8px">PRO</span>' if u.is_pro else ""
    head = f"""<div class="card"><div class="row"><span class="big"></span>
      <div><div style="font-size:18px;font-weight:600">{esc(u.display_name or u.handle)}{pro}</div>
      <span class="handle">@{esc(u.handle)}</span></div>
      <span style="margin-left:auto">{action}</span></div>
      <p style="margin:12px 0 0;color:#cdd7ea;font-size:13.5px">{esc(u.bio) or '<span class="muted">No bio yet.</span>'}</p>
      <div class="split"><span><b>{nposts}</b> posts</span><span><b>{followers}</b> followers</span><span><b>{following}</b> following</span></div>
    </div>"""
    posts = db.scalars(select(Post).where(Post.author_id == u.id).order_by(Post.created_at.desc()).limit(50)).all()
    body_html = head + ("".join(_post_card(db, p, viewer) for p in posts) or '<p class="muted" style="margin-top:14px">No posts yet.</p>')
    return HTMLResponse(layout(u.display_name or u.handle, body_html, viewer))


@router.post("/follow/{handle}")
def follow(handle: str, request: Request, db: Session = Depends(get_db), user: User = Depends(current_user)):
    target = db.scalar(select(User).where(User.handle == handle.lower()))
    if target and target.id != user.id and not db.get(Follow, {"follower_id": user.id, "followee_id": target.id}):
        db.add(Follow(follower_id=user.id, followee_id=target.id)); db.commit()
    return RedirectResponse(f"/social/u/{handle.lower()}", status_code=303)


@router.post("/unfollow/{handle}")
def unfollow(handle: str, request: Request, db: Session = Depends(get_db), user: User = Depends(current_user)):
    target = db.scalar(select(User).where(User.handle == handle.lower()))
    if target:
        db.execute(delete(Follow).where(Follow.follower_id == user.id, Follow.followee_id == target.id)); db.commit()
    return RedirectResponse(f"/social/u/{handle.lower()}", status_code=303)


@router.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, db: Session = Depends(get_db), user: User = Depends(current_user)):
    body = f"""<h1>Edit profile</h1><div class="card"><form method="post" action="/social/settings">
      <label>Display name</label><input name="display_name" maxlength="80" value="{esc(user.display_name)}">
      <label>Bio</label><textarea name="bio" maxlength="280">{esc(user.bio)}</textarea>
      <button class="btn" type="submit">Save</button></form></div>"""
    return HTMLResponse(layout("Settings", body, user))


@router.post("/settings")
def save_settings(request: Request, db: Session = Depends(get_db), user: User = Depends(current_user),
                  display_name: str = Form(""), bio: str = Form("")):
    user.display_name = display_name.strip()[:80] or user.handle
    user.bio = bio.strip()[:280]
    db.commit()
    return RedirectResponse(f"/social/u/{user.handle}", status_code=303)


