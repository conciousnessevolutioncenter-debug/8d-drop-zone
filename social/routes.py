"""Social routes: auth, profiles, feed, follow. Server-rendered (PRG pattern),
mounted under /social so the audio app routes are untouched."""
from __future__ import annotations

import re
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from sqlalchemy import select, func, delete
from sqlalchemy.orm import Session

from .db import get_db
from .models import User, Post, Like, Follow, Comment
from .security import hash_password, verify_password, login_session, logout_session, optional_user, current_user
from .ui import layout, esc

router = APIRouter(prefix="/social", tags=["social"])
_HANDLE_RE = re.compile(r"^[a-z0-9_]{3,30}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Post-image storage. Dev: a temp dir served by the route below. Prod: point at a
# persistent volume or swap to S3/R2 (the container's temp dir is ephemeral).
MEDIA_DIR = Path(tempfile.gettempdir()) / "8d_social_media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
_IMG_EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp"}
_MAX_IMG = 8 * 1024 * 1024


def _counts(db: Session, user: User):
    followers = db.scalar(select(func.count()).select_from(Follow).where(Follow.followee_id == user.id))
    following = db.scalar(select(func.count()).select_from(Follow).where(Follow.follower_id == user.id))
    posts = db.scalar(select(func.count()).select_from(Post).where(Post.author_id == user.id))
    return followers or 0, following or 0, posts or 0


def _post_body(db: Session, post: Post, viewer: User | None) -> str:
    """The inner content of a post (author header, text, image, track)."""
    a = post.author
    pro = '<span class="pro" style="margin-left:6px">PRO</span>' if a.is_pro else ""
    when = post.created_at.strftime("%b %d, %H:%M") if post.created_at else ""
    text = f'<p style="margin:10px 0 0;font-size:14px;line-height:1.55;color:#cdd7ea">{esc(post.body)}</p>' if post.body else ""
    img = (f'<img src="{esc(post.image_url)}" alt="post image" '
           f'style="margin-top:10px;max-width:100%;border-radius:11px;border:1px solid var(--hair)">') if post.image_url else ""
    track = ""
    if post.track_job_id:
        track = ('<div class="row" style="border:1px solid rgba(98,224,255,.25);background:rgba(98,224,255,.05);'
                 'border-radius:11px;margin-top:10px;padding:9px 11px">'
                 '<span class="tag" style="color:var(--cyan)">8D TRACK ATTACHED</span></div>')
    return (f'<div class="row"><span class="avatar"></span>'
            f'<div><div style="font-size:13px;font-weight:500">{esc(a.display_name or a.handle)}{pro}</div>'
            f'<a href="/social/u/{esc(a.handle)}" class="handle">@{esc(a.handle)} · {when}</a></div></div>'
            f'{text}{img}{track}')


def _post_card(db: Session, post: Post, viewer: User | None) -> str:
    # Render a repost as a thin "reposted by" wrapper around the original.
    repost_header = ""
    target = post
    if post.repost_of:
        original = db.get(Post, post.repost_of)
        repost_header = f'<div class="handle" style="margin-bottom:8px">↻ {esc(post.author.handle)} reposted</div>'
        if not original:
            return f'<div class="card">{repost_header}<p class="muted">The original post was removed.</p></div>'
        target = original

    likes = db.scalar(select(func.count()).select_from(Like).where(Like.post_id == target.id)) or 0
    ncomments = db.scalar(select(func.count()).select_from(Comment).where(Comment.post_id == target.id)) or 0
    like = (f'<a href="/social/posts/{target.id}/like" onclick="event.preventDefault();'
            f"fetch(this.href,{{method:'POST'}}).then(()=>location.reload())\">♥ {likes}</a>") if viewer else f"♥ {likes}"
    comment = f'<a href="/social/p/{target.id}">💬 {ncomments}</a>'
    if viewer:
        repost = (f'<a href="/social/posts/{target.id}/repost" onclick="event.preventDefault();'
                  f"fetch(this.href,{{method:'POST'}}).then(()=>location.reload())\">↻ repost</a>")
    else:
        repost = "↻ repost"
    return (f'<div class="card">{repost_header}{_post_body(db, target, viewer)}'
            f'<div class="post-actions">{like}{comment}{repost}</div></div>')


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
    composer = """<div class="card"><form method="post" action="/social/posts" enctype="multipart/form-data">
      <textarea name="body" placeholder="Share a track or a thought…" maxlength="2000"></textarea>
      <label style="margin-top:10px">Image (optional)</label>
      <input type="file" name="image" accept="image/png,image/jpeg,image/gif,image/webp">
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


async def _save_image(image: UploadFile | None) -> str | None:
    """Persist an uploaded image (validated) and return its served URL, or None."""
    if not image or not image.filename:
        return None
    ext = _IMG_EXT.get((image.content_type or "").lower())
    if not ext:
        return None  # silently ignore non-images rather than failing the post
    data = await image.read(_MAX_IMG + 1)
    if not data or len(data) > _MAX_IMG:
        return None
    name = f"{uuid.uuid4().hex}{ext}"
    (MEDIA_DIR / name).write_bytes(data)
    return f"/social/media/{name}"


@router.post("/posts")
async def create_post(request: Request, db: Session = Depends(get_db),
                      body: str = Form(""), track_job_id: str = Form(None),
                      image: UploadFile = File(None),
                      user: User = Depends(current_user)):
    text = (body or "").strip()
    image_url = await _save_image(image)
    if text or track_job_id or image_url:
        db.add(Post(author_id=user.id, body=text[:2000], track_job_id=track_job_id or None, image_url=image_url))
        db.commit()
    return RedirectResponse("/social", status_code=303)


@router.get("/media/{name}")
def media(name: str):
    # Path-safe: only a bare filename, must exist in the media dir.
    if "/" in name or "\\" in name or ".." in name:
        return HTMLResponse("bad path", status_code=400)
    path = MEDIA_DIR / name
    if not path.is_file():
        return HTMLResponse("not found", status_code=404)
    return FileResponse(str(path))


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


@router.post("/posts/{post_id}/repost")
def repost(post_id: int, request: Request, db: Session = Depends(get_db),
           user: User = Depends(current_user)):
    target = db.get(Post, post_id)
    if target:
        # Collapse repost-of-repost to the original so embeds don't nest.
        original_id = target.repost_of or target.id
        if not db.scalar(select(Post).where(Post.author_id == user.id, Post.repost_of == original_id)):
            db.add(Post(author_id=user.id, body="", repost_of=original_id))
            db.commit()
    return {"ok": True}


@router.post("/posts/{post_id}/comment")
def add_comment(post_id: int, request: Request, db: Session = Depends(get_db),
                body: str = Form(""), user: User = Depends(current_user)):
    text = (body or "").strip()
    if text and db.get(Post, post_id):
        db.add(Comment(post_id=post_id, author_id=user.id, body=text[:2000]))
        db.commit()
    return RedirectResponse(f"/social/p/{post_id}", status_code=303)


@router.get("/p/{post_id}", response_class=HTMLResponse)
def post_detail(post_id: int, request: Request, db: Session = Depends(get_db)):
    viewer = optional_user(request, db)
    post = db.get(Post, post_id)
    if not post:
        return HTMLResponse(layout("Not found", "<h1>Post not found</h1>", viewer), status_code=404)
    card = _post_card(db, post, viewer)
    comments = db.scalars(select(Comment).where(Comment.post_id == post_id).order_by(Comment.created_at)).all()
    clist = ""
    for cm in comments:
        ca = db.get(User, cm.author_id)
        when = cm.created_at.strftime("%b %d, %H:%M") if cm.created_at else ""
        clist += (f'<div class="card" style="margin-top:10px;padding:11px 14px"><div class="row"><span class="avatar"></span>'
                  f'<a href="/social/u/{esc(ca.handle)}" class="handle">@{esc(ca.handle)} · {when}</a></div>'
                  f'<p style="margin:8px 0 0;font-size:13.5px;color:#cdd7ea">{esc(cm.body)}</p></div>')
    if viewer:
        composer = (f'<div class="card"><form method="post" action="/social/posts/{post_id}/comment">'
                    f'<textarea name="body" placeholder="Add a comment…" maxlength="2000"></textarea>'
                    f'<button class="btn" type="submit">Comment</button></form></div>')
    else:
        composer = '<p class="muted" style="margin-top:14px"><a href="/social/login" style="color:var(--cyan)">Sign in</a> to comment.</p>'
    head = '<a href="/social" class="tag" style="color:var(--cyan)">← BACK TO FEED</a>'
    body_html = head + card + f'<div class="tag" style="margin-top:18px">{len(comments)} COMMENTS</div>' + clist + composer
    return HTMLResponse(layout("Post", body_html, viewer, "feed"))


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


