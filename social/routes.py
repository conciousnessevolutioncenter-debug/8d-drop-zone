"""Social routes: auth, profiles, feed, follow. Server-rendered (PRG pattern),
mounted under /social so the audio app routes are untouched."""
from __future__ import annotations

import os
import re
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from sqlalchemy import select, func, delete
from sqlalchemy.orm import Session

from .db import get_db
from .models import User, Post, Like, Follow, Comment, Notification
from .security import hash_password, verify_password, login_session, logout_session, optional_user, current_user
from .ui import layout, esc, avatar_html
from .notify import create_notification
from .mailer import (make_reset_token, read_reset_token, send_email, smtp_configured,
                     make_verify_token, read_verify_token)
from .ratelimit import rate_limit

router = APIRouter(prefix="/social", tags=["social"])
_HANDLE_RE = re.compile(r"^[a-z0-9_]{3,30}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Post-image storage. Dev: a temp dir served by the route below. Prod: point at a
# persistent volume or swap to S3/R2 (the container's temp dir is ephemeral).
# Persistent in prod: set SOCIAL_MEDIA_DIR to a mounted volume path (the temp
# dir is wiped on redeploy). Dev falls back to a temp folder.
MEDIA_DIR = Path(os.environ.get("SOCIAL_MEDIA_DIR") or (Path(tempfile.gettempdir()) / "8d_social_media"))
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
# content-type -> extension, plus an extension allowlist for fallback when the
# browser sends a generic content-type (e.g. application/octet-stream).
_IMG_EXT = {"image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png",
            "image/gif": ".gif", "image/webp": ".webp", "image/avif": ".avif", "image/bmp": ".bmp"}
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif", ".bmp"}
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
    return (f'<div class="row">{avatar_html(a)}'
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
    owner = ""
    if viewer and viewer.id == target.author_id:
        owner = (f'<a href="/social/p/{target.id}/edit" style="margin-left:auto">edit</a>'
                 f'<a href="/social/posts/{target.id}/delete" onclick="event.preventDefault();'
                 f"if(confirm('Delete this post?'))fetch(this.href,{{method:'POST'}}).then(()=>location.href='/social')\">delete</a>")
    return (f'<div class="card">{repost_header}{_post_body(db, target, viewer)}'
            f'<div class="post-actions">{like}{comment}{repost}{owner}</div></div>')


@router.get("/", response_class=HTMLResponse)
def feed(request: Request, db: Session = Depends(get_db), img: str = ""):
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
      <input type="file" name="image" accept="image/png,image/jpeg,image/gif,image/webp,image/avif,image/heic,image/heif,.heic,.heif">
      <button class="btn" type="submit">Post</button></form></div>"""
    banner = ('<div class="card" style="border-color:rgba(255,138,138,.4);background:rgba(255,138,138,.06);margin-bottom:4px">'
              '<span class="err" style="margin:0">Couldn\'t add that image — supported types are JPG, PNG, GIF, WebP, AVIF, BMP and HEIC, up to 8 MB.</span></div>') if img == "err" else ""
    feed_html = "".join(_post_card(db, p, user) for p in rows) or '<p class="muted" style="margin-top:16px">Your feed is quiet — follow some creators or post something.</p>'
    return HTMLResponse(layout("Feed", banner + composer + feed_html, user, "feed"))


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
    rate_limit(request, "register", 6, 300)
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
    # Send a verification link (logged if SMTP unset). Soft gate: we don't block
    # login, just nudge with a banner until verified.
    token = make_verify_token(user.id)
    link = f"{str(request.base_url).rstrip('/')}/social/verify/{token}"
    send_email(user.email, "Verify your 8D Engine email",
               f"Welcome! Confirm your email to finish setting up your account:\n\n{link}")
    login_session(request, user)
    return RedirectResponse("/social", status_code=303)


@router.get("/verify/{token}", response_class=HTMLResponse)
def verify_email(token: str, request: Request, db: Session = Depends(get_db)):
    uid = read_verify_token(token)
    viewer = optional_user(request, db)
    if uid is None:
        return HTMLResponse(layout("Verify", '<h1>Link expired</h1><p class="lede">That verification link is invalid or has expired.</p><a href="/social/verify/resend" class="btn">Resend</a>', viewer), status_code=400)
    u = db.get(User, uid)
    if u and not u.email_verified:
        u.email_verified = True
        db.commit()
    return HTMLResponse(layout("Verified", '<h1>Email verified ✓</h1><p class="lede">Thanks — your account is confirmed.</p><a href="/social" class="btn">Go to feed</a>', viewer), status_code=200)


@router.get("/verify/resend")
def verify_resend(request: Request, db: Session = Depends(get_db), user: User = Depends(current_user)):
    rate_limit(request, "verify_resend", 4, 600)
    if not user.email_verified:
        token = make_verify_token(user.id)
        link = f"{str(request.base_url).rstrip('/')}/social/verify/{token}"
        send_email(user.email, "Verify your 8D Engine email", f"Confirm your email:\n\n{link}")
    return RedirectResponse("/social", status_code=303)


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request, db: Session = Depends(get_db)):
    if optional_user(request, db):
        return RedirectResponse("/social", status_code=303)
    body = """<h1>Sign in</h1><div class="card"><form method="post" action="/social/login">
      <label>Email</label><input name="email" type="email" required>
      <label>Password</label><input name="password" type="password" required>
      <button class="btn" type="submit">Sign in</button></form></div>
    <p class="muted" style="margin-top:12px">New here? <a href="/social/register" style="color:var(--cyan)">Create an account</a>
    &nbsp;·&nbsp; <a href="/social/reset-request" style="color:var(--cyan)">Forgot password?</a></p>"""
    return HTMLResponse(layout("Sign in", body, None))


@router.post("/login")
def login(request: Request, db: Session = Depends(get_db),
          email: str = Form(...), password: str = Form(...)):
    rate_limit(request, "login", 10, 300)
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


@router.get("/reset-request", response_class=HTMLResponse)
def reset_request_page(request: Request):
    body = """<h1>Reset your password</h1>
    <p class="lede">Enter your email and we'll send a reset link.</p>
    <div class="card"><form method="post" action="/social/reset-request">
      <label>Email</label><input name="email" type="email" required>
      <button class="btn" type="submit">Send reset link</button></form></div>"""
    return HTMLResponse(layout("Reset password", body, None))


@router.post("/reset-request", response_class=HTMLResponse)
def reset_request(request: Request, db: Session = Depends(get_db), email: str = Form(...)):
    rate_limit(request, "reset", 5, 600)
    user = db.scalar(select(User).where(User.email == email.strip().lower()))
    if user:
        token = make_reset_token(user.id)
        link = f"{str(request.base_url).rstrip('/')}/social/reset/{token}"
        send_email(user.email, "Reset your 8D Engine password",
                   f"Tap to reset your password (valid 1 hour):\n\n{link}\n\nIf you didn't request this, ignore it.")
    # Always the same response — never reveal whether an email exists.
    note = "" if smtp_configured() else '<p class="muted" style="margin-top:10px">(Email isn\'t configured yet — the reset link is in the server logs.)</p>'
    body = ('<h1>Check your email</h1><p class="lede">If an account exists for that address, '
            'a reset link is on its way.</p>' + note +
            '<a href="/social/login" class="btn ghost" style="margin-top:12px">Back to sign in</a>')
    return HTMLResponse(layout("Reset password", body, None))


@router.get("/reset/{token}", response_class=HTMLResponse)
def reset_page(token: str, request: Request):
    if read_reset_token(token) is None:
        body = '<h1>Link expired</h1><p class="lede">That reset link is invalid or has expired.</p><a href="/social/reset-request" class="btn">Request a new one</a>'
        return HTMLResponse(layout("Reset password", body, None), status_code=400)
    body = f"""<h1>Set a new password</h1><div class="card"><form method="post" action="/social/reset/{esc(token)}">
      <label>New password</label><input name="password" type="password" minlength="8" required>
      <button class="btn" type="submit">Update password</button></form></div>"""
    return HTMLResponse(layout("Reset password", body, None))


@router.post("/reset/{token}", response_class=HTMLResponse)
def reset_submit(token: str, request: Request, db: Session = Depends(get_db), password: str = Form(...)):
    uid = read_reset_token(token)
    if uid is None:
        return HTMLResponse(layout("Reset password", '<h1>Link expired</h1><a href="/social/reset-request" class="btn">Request a new one</a>', None), status_code=400)
    if len(password) < 8:
        return HTMLResponse(layout("Reset password", f'<h1>Set a new password</h1><div class="card"><p class="err">Password must be at least 8 characters.</p><a href="/social/reset/{esc(token)}" class="btn ghost">Back</a></div>', None), status_code=400)
    user = db.get(User, uid)
    if user:
        user.password_hash = hash_password(password)
        db.commit()
        login_session(request, user)
        return RedirectResponse("/social", status_code=303)
    return HTMLResponse(layout("Reset password", "<h1>Account not found</h1>", None), status_code=404)


def _heic_to_jpeg(data: bytes) -> bytes | None:
    """Convert HEIC/HEIF (iPhone photos) to JPEG so it renders in browsers."""
    try:
        import io
        import pillow_heif
        from PIL import Image
        pillow_heif.register_heif_opener()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=90)
        return out.getvalue()
    except Exception as exc:
        print(f"[social] HEIC->JPEG conversion failed: {exc}", flush=True)
        return None


async def _save_image(image: UploadFile | None) -> str | None:
    """Persist an uploaded image and return its served URL, or None if the file
    isn't a supported/renderable image or is too large. HEIC/HEIF is converted
    to JPEG; generic content-types fall back to the file extension."""
    if not image or not image.filename:
        return None
    data = await image.read(_MAX_IMG + 1)
    if not data or len(data) > _MAX_IMG:
        return None
    ct = (image.content_type or "").lower()
    suf = Path(image.filename).suffix.lower()
    if ct in ("image/heic", "image/heif") or suf in (".heic", ".heif"):
        jpg = _heic_to_jpeg(data)
        if not jpg:
            return None
        data, ext = jpg, ".jpg"
    else:
        ext = _IMG_EXT.get(ct)
        if not ext and suf in _IMG_EXTS:
            ext = ".jpg" if suf == ".jpeg" else suf
        if not ext:
            return None
    name = f"{uuid.uuid4().hex}{ext}"
    (MEDIA_DIR / name).write_bytes(data)
    return f"/social/media/{name}"


@router.post("/posts")
async def create_post(request: Request, db: Session = Depends(get_db),
                      body: str = Form(""), track_job_id: str = Form(None),
                      image: UploadFile = File(None),
                      user: User = Depends(current_user)):
    rate_limit(request, "post", 20, 60)
    text = (body or "").strip()
    image_provided = bool(image and image.filename)
    image_url = await _save_image(image)
    if text or track_job_id or image_url:
        db.add(Post(author_id=user.id, body=text[:2000], track_job_id=track_job_id or None, image_url=image_url))
        db.commit()
    # If the user attached a file we couldn't use, tell them (instead of silently dropping it).
    if image_provided and not image_url:
        return RedirectResponse("/social?img=err", status_code=303)
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
    post = db.get(Post, post_id)
    if text and post:
        db.add(Comment(post_id=post_id, author_id=user.id, body=text[:2000]))
        if post.author_id != user.id:
            create_notification(db, post.author_id, "comment", f"@{user.handle} commented on your post", f"/social/p/{post_id}")
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
        clist += (f'<div class="card" style="margin-top:10px;padding:11px 14px"><div class="row">{avatar_html(ca)}'
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


@router.get("/notifications", response_class=HTMLResponse)
def notifications(request: Request, db: Session = Depends(get_db), user: User = Depends(current_user)):
    rows = db.scalars(
        select(Notification).where(Notification.user_id == user.id).order_by(Notification.created_at.desc()).limit(60)
    ).all()
    items = ""
    for n in rows:
        when = n.created_at.strftime("%b %d, %H:%M") if n.created_at else ""
        dot = '<span style="color:var(--cyan)">●</span> ' if not n.read else ""
        icon = {"follow": "+", "comment": "💬", "dm": "✉"}.get(n.kind, "•")
        items += (f'<a href="{esc(n.link) or "/social"}" class="card" style="display:block;margin-top:10px;padding:11px 14px">'
                  f'<div class="row" style="gap:10px">{dot}<span>{icon}</span>'
                  f'<span style="font-size:13.5px">{esc(n.text)}</span>'
                  f'<span class="handle" style="margin-left:auto">{when}</span></div></a>')
    # mark all read once viewed
    for n in rows:
        n.read = True
    db.commit()
    body = '<h1>Notifications</h1>' + (items or '<p class="muted" style="margin-top:14px">Nothing yet.</p>')
    return HTMLResponse(layout("Notifications", body, user, "notifs"))


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
    head = f"""<div class="card"><div class="row">{avatar_html(u, 'big')}
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
        db.add(Follow(follower_id=user.id, followee_id=target.id))
        create_notification(db, target.id, "follow", f"@{user.handle} followed you", f"/social/u/{user.handle}")
        db.commit()
    return RedirectResponse(f"/social/u/{handle.lower()}", status_code=303)


@router.post("/unfollow/{handle}")
def unfollow(handle: str, request: Request, db: Session = Depends(get_db), user: User = Depends(current_user)):
    target = db.scalar(select(User).where(User.handle == handle.lower()))
    if target:
        db.execute(delete(Follow).where(Follow.follower_id == user.id, Follow.followee_id == target.id)); db.commit()
    return RedirectResponse(f"/social/u/{handle.lower()}", status_code=303)


@router.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, db: Session = Depends(get_db), user: User = Depends(current_user)):
    body = f"""<h1>Edit profile</h1><div class="card"><form method="post" action="/social/settings" enctype="multipart/form-data">
      <div class="row" style="gap:12px">{avatar_html(user, 'big')}<div class="muted">Your avatar</div></div>
      <label>Avatar image (optional)</label>
      <input type="file" name="avatar" accept="image/png,image/jpeg,image/gif,image/webp">
      <label>Display name</label><input name="display_name" maxlength="80" value="{esc(user.display_name)}">
      <label>Bio</label><textarea name="bio" maxlength="280">{esc(user.bio)}</textarea>
      <button class="btn" type="submit">Save</button></form></div>"""
    return HTMLResponse(layout("Settings", body, user))


@router.post("/settings")
async def save_settings(request: Request, db: Session = Depends(get_db), user: User = Depends(current_user),
                        display_name: str = Form(""), bio: str = Form(""), avatar: UploadFile = File(None)):
    user.display_name = display_name.strip()[:80] or user.handle
    user.bio = bio.strip()[:280]
    avatar_url = await _save_image(avatar)
    if avatar_url:
        user.avatar_url = avatar_url
    db.commit()
    return RedirectResponse(f"/social/u/{user.handle}", status_code=303)


@router.post("/posts/{post_id}/delete")
def delete_post(post_id: int, request: Request, db: Session = Depends(get_db), user: User = Depends(current_user)):
    post = db.get(Post, post_id)
    if post and post.author_id == user.id:
        db.delete(post); db.commit()
    return {"ok": True}


@router.get("/p/{post_id}/edit", response_class=HTMLResponse)
def edit_post_page(post_id: int, request: Request, db: Session = Depends(get_db), user: User = Depends(current_user)):
    post = db.get(Post, post_id)
    if not post or post.author_id != user.id:
        return HTMLResponse(layout("Edit", "<h1>Can't edit that post</h1>", user), status_code=403)
    body = (f'<h1>Edit post</h1><div class="card"><form method="post" action="/social/p/{post_id}/edit">'
            f'<textarea name="body" maxlength="2000">{esc(post.body)}</textarea>'
            f'<button class="btn" type="submit">Save</button> '
            f'<a href="/social/p/{post_id}" class="btn ghost" style="margin-left:8px">Cancel</a></form></div>')
    return HTMLResponse(layout("Edit post", body, user, "feed"))


@router.post("/p/{post_id}/edit")
def edit_post(post_id: int, request: Request, db: Session = Depends(get_db), body: str = Form(""), user: User = Depends(current_user)):
    post = db.get(Post, post_id)
    if post and post.author_id == user.id:
        post.body = (body or "").strip()[:2000]
        db.commit()
    return RedirectResponse(f"/social/p/{post_id}", status_code=303)


@router.get("/people", response_class=HTMLResponse)
def people(request: Request, db: Session = Depends(get_db), q: str = ""):
    viewer = optional_user(request, db)
    query = select(User)
    term = (q or "").strip()
    if term:
        like = f"%{term.lower()}%"
        query = query.where(func.lower(User.handle).like(like) | func.lower(User.display_name).like(like))
    if viewer:
        query = query.where(User.id != viewer.id)
    users = db.scalars(query.order_by(User.created_at.desc()).limit(50)).all()
    following = set()
    if viewer:
        following = set(db.scalars(select(Follow.followee_id).where(Follow.follower_id == viewer.id)).all())
    cards = ""
    for u in users:
        pro = '<span class="pro" style="margin-left:6px">' + ("PRO") + '</span>' if u.is_pro else ""
        if viewer and u.id in following:
            act = f'<form method="post" action="/social/unfollow/{esc(u.handle)}" style="margin-left:auto"><button class="btn ghost" type="submit">Following</button></form>'
        elif viewer:
            act = f'<form method="post" action="/social/follow/{esc(u.handle)}" style="margin-left:auto"><button class="btn" type="submit">Follow</button></form>'
        else:
            act = ''
        cards += (f'<div class="card" style="margin-top:10px"><div class="row">{avatar_html(u)}'
                  f'<div><a href="/social/u/{esc(u.handle)}" style="font-size:14px;font-weight:500">{esc(u.display_name or u.handle)}{pro}</a>'
                  f'<div class="handle">@{esc(u.handle)}</div></div>{act}</div>'
                  + (f'<p class="muted" style="margin:8px 0 0">{esc(u.bio)}</p>' if u.bio else '') + '</div>')
    search = (f'<form method="get" action="/social/people"><input name="q" placeholder="Search people…" value="{esc(term)}">'
              f'<button class="btn" type="submit">Search</button></form>')
    body = '<h1>People</h1><p class="lede">Find creators to follow.</p><div class="card">' + search + '</div>' + (cards or '<p class="muted" style="margin-top:14px">No matches.</p>')
    return HTMLResponse(layout("People", body, viewer, "people"))


