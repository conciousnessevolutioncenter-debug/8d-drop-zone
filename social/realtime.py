"""Realtime chat + DMs over WebSockets.

Single-instance in-process fan-out (a dict of channel -> sockets). To scale to
multiple Railway instances later, swap `Hub.broadcast` for Redis pub/sub — the
rest of the code stays the same. Messages persist to Postgres/SQLite so history
survives restarts. WebSocket auth reuses the same signed session cookie.
"""
from __future__ import annotations

import asyncio
import html
from collections import defaultdict

from fastapi import APIRouter, Depends, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select, func, or_, and_
from sqlalchemy.orm import Session

from .db import get_db, SessionLocal
from .models import User, Room, Message, DmThread, DmRead, Block, Report
from .security import current_user, optional_user
from .ui import layout, esc, CSS
from . import entitlements as ent

rt_router = APIRouter(prefix="/social", tags=["social-realtime"])


class Hub:
    def __init__(self):
        self._chans: dict[str, set[WebSocket]] = defaultdict(set)

    def join(self, channel: str, ws: WebSocket):
        self._chans[channel].add(ws)

    def leave(self, channel: str, ws: WebSocket):
        self._chans[channel].discard(ws)

    async def broadcast(self, channel: str, payload: dict):
        for ws in list(self._chans.get(channel, ())):
            try:
                await ws.send_json(payload)
            except Exception:
                self._chans[channel].discard(ws)


hub = Hub()


def _blocked_ids(db: Session, uid: int) -> set[int]:
    rows = db.scalars(select(Block.blocked_id).where(Block.blocker_id == uid)).all()
    return set(rows)


# ---------- Rooms ----------

@rt_router.get("/rooms", response_class=HTMLResponse)
def rooms(request: Request, db: Session = Depends(get_db)):
    user = optional_user(request, db)
    rows = db.scalars(select(Room).order_by(Room.slug == "lobby", Room.created_at.desc())).all()
    cards = ""
    for r in rows:
        lock = ' <span class="tag" style="color:var(--violet)">PRO</span>' if r.pro_only else ""
        cards += (f'<a href="/social/rooms/{esc(r.slug)}" class="card" style="display:block;margin-top:10px">'
                  f'<div class="row"><span style="font-size:14px;font-weight:500"># {esc(r.name)}{lock}</span></div>'
                  f'<div class="muted" style="margin-top:4px">{esc(r.topic)}</div></a>')
    create = ""
    if user:
        create = """<div class="card"><form method="post" action="/social/rooms">
          <label>New room name</label><input name="name" maxlength="80" required>
          <label>Topic</label><input name="topic" maxlength="160">
          <button class="btn" type="submit">Create room</button></form></div>"""
    body = '<h1>Rooms</h1><p class="lede">Real-time public chat. The lobby is open to everyone.</p>' + create + cards
    return HTMLResponse(layout("Rooms", body, user, "rooms"))


@rt_router.post("/rooms")
def create_room(request: Request, db: Session = Depends(get_db),
                name: str = Form(...), topic: str = Form(""), user: User = Depends(current_user)):
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")[:50] or "room"
    base, i = slug, 2
    while db.scalar(select(Room).where(Room.slug == slug)):
        slug = f"{base}-{i}"; i += 1
    db.add(Room(slug=slug, name=name.strip()[:80], topic=topic.strip()[:160], created_by=user.id))
    db.commit()
    return RedirectResponse(f"/social/rooms/{slug}", status_code=303)


@rt_router.get("/rooms/{slug}", response_class=HTMLResponse)
def room_page(slug: str, request: Request, db: Session = Depends(get_db)):
    user = optional_user(request, db)
    if not user:
        return RedirectResponse("/social/login", status_code=303)
    room = db.scalar(select(Room).where(Room.slug == slug))
    if not room:
        return HTMLResponse(layout("Not found", "<h1>No such room</h1>", user, "rooms"), status_code=404)
    if room.pro_only and not ent.is_paid(user):
        return HTMLResponse(layout(room.name, '<h1># ' + esc(room.name) + '</h1><p class="lede">This room is for paid members. <a href="/social/billing" style="color:var(--cyan)">See plans</a>.</p>', user, "rooms"), status_code=403)
    blocked = _blocked_ids(db, user.id)
    history = db.scalars(select(Message).where(Message.room_id == room.id).order_by(Message.created_at.desc()).limit(50)).all()
    history = list(reversed(history))
    msgs_html = ""
    for m in history:
        if m.sender_id in blocked:
            continue
        sender = db.get(User, m.sender_id)
        msgs_html += _msg_line(sender, m.body, m.created_at.strftime("%H:%M"))
    body = _chat_shell(title=f"# {esc(room.name)}", subtitle=esc(room.topic),
                       ws_path=f"/social/ws/rooms/{esc(slug)}", history=msgs_html)
    return HTMLResponse(_chat_doc(f"# {room.name}", body, user, "rooms"))


@rt_router.websocket("/ws/rooms/{slug}")
async def ws_room(ws: WebSocket, slug: str):
    try:
        uid = ws.session.get("uid")
    except Exception:
        uid = None
    if not uid:
        await ws.close(code=4401); return
    db = SessionLocal()
    try:
        user = db.get(User, uid)
        room = db.scalar(select(Room).where(Room.slug == slug))
        if not user or not room or (room.pro_only and not ent.is_paid(user)):
            await ws.close(code=4403); return
        await ws.accept()
        channel = f"room:{slug}"
        hub.join(channel, ws)
        try:
            while True:
                data = await ws.receive_json()
                body = (data.get("body") or "").strip()[:2000]
                if not body:
                    continue
                msg = Message(room_id=room.id, sender_id=user.id, body=body)
                db.add(msg); db.commit(); db.refresh(msg)
                await hub.broadcast(channel, {
                    "handle": user.handle, "display": user.display_name or user.handle,
                    "pro": user.is_pro, "body": body, "ts": msg.created_at.strftime("%H:%M"),
                })
        except WebSocketDisconnect:
            pass
        finally:
            hub.leave(channel, ws)
    finally:
        db.close()


# ---------- Direct messages ----------

def _thread_for(db: Session, a: int, b: int, create: bool = True) -> DmThread | None:
    lo, hi = sorted((a, b))
    t = db.scalar(select(DmThread).where(DmThread.user_a == lo, DmThread.user_b == hi))
    if not t and create:
        t = DmThread(user_a=lo, user_b=hi); db.add(t); db.commit(); db.refresh(t)
    return t


@rt_router.get("/inbox", response_class=HTMLResponse)
def inbox(request: Request, db: Session = Depends(get_db)):
    user = optional_user(request, db)
    if not user:
        return RedirectResponse("/social/login", status_code=303)
    threads = db.scalars(select(DmThread).where(or_(DmThread.user_a == user.id, DmThread.user_b == user.id))).all()
    items = ""
    for t in threads:
        other_id = t.user_b if t.user_a == user.id else t.user_a
        other = db.get(User, other_id)
        last = db.scalar(select(Message).where(Message.thread_id == t.id).order_by(Message.created_at.desc()).limit(1))
        read = db.get(DmRead, {"thread_id": t.id, "user_id": user.id})
        unread = last and last.sender_id != user.id and (not read or read.last_read_message_id < last.id)
        dot = ' <span style="color:var(--cyan)">●</span>' if unread else ""
        preview = esc(last.body[:60]) if last else '<span class="muted">No messages yet</span>'
        items += (f'<a href="/social/dm/{esc(other.handle)}" class="card" style="display:block;margin-top:10px">'
                  f'<div class="row"><span class="avatar"></span><b>{esc(other.display_name or other.handle)}</b>{dot}</div>'
                  f'<div class="muted" style="margin-top:4px">{preview}</div></a>')
    body = '<h1>Inbox</h1><p class="lede">Private one-on-one messages.</p>' + (items or '<p class="muted" style="margin-top:14px">No conversations yet — open someone\'s profile to message them.</p>')
    return HTMLResponse(layout("Inbox", body, user, "inbox"))


@rt_router.get("/dm/{handle}", response_class=HTMLResponse)
def dm_page(handle: str, request: Request, db: Session = Depends(get_db)):
    user = optional_user(request, db)
    if not user:
        return RedirectResponse("/social/login", status_code=303)
    other = db.scalar(select(User).where(User.handle == handle.lower()))
    if not other or other.id == user.id:
        return HTMLResponse(layout("Inbox", "<h1>Can't open that conversation</h1>", user, "inbox"), status_code=404)
    t = _thread_for(db, user.id, other.id)
    history = db.scalars(select(Message).where(Message.thread_id == t.id).order_by(Message.created_at.desc()).limit(50)).all()
    history = list(reversed(history))
    # mark read up to latest
    last_id = history[-1].id if history else 0
    rr = db.get(DmRead, {"thread_id": t.id, "user_id": user.id})
    if rr:
        rr.last_read_message_id = max(rr.last_read_message_id, last_id)
    else:
        db.add(DmRead(thread_id=t.id, user_id=user.id, last_read_message_id=last_id))
    db.commit()
    msgs_html = "".join(
        _msg_line(db.get(User, m.sender_id), m.body, m.created_at.strftime("%H:%M"),
                  mine=(m.sender_id == user.id))
        for m in history
    )
    body = _chat_shell(title=esc(other.display_name or other.handle), subtitle=f"@{esc(other.handle)}",
                       ws_path=f"/social/ws/dm/{t.id}", history=msgs_html, dm=True)
    return HTMLResponse(_chat_doc(other.display_name or other.handle, body, user, "inbox"))


@rt_router.websocket("/ws/dm/{thread_id}")
async def ws_dm(ws: WebSocket, thread_id: int):
    try:
        uid = ws.session.get("uid")
    except Exception:
        uid = None
    if not uid:
        await ws.close(code=4401); return
    db = SessionLocal()
    try:
        t = db.get(DmThread, thread_id)
        if not t or uid not in (t.user_a, t.user_b):
            await ws.close(code=4403); return
        user = db.get(User, uid)
        await ws.accept()
        channel = f"dm:{thread_id}"
        hub.join(channel, ws)
        try:
            while True:
                data = await ws.receive_json()
                if data.get("type") == "read":
                    rr = db.get(DmRead, {"thread_id": thread_id, "user_id": uid})
                    top = db.scalar(select(func.max(Message.id)).where(Message.thread_id == thread_id)) or 0
                    if rr: rr.last_read_message_id = top
                    else: db.add(DmRead(thread_id=thread_id, user_id=uid, last_read_message_id=top))
                    db.commit()
                    await hub.broadcast(channel, {"type": "read", "by": user.handle})
                    continue
                body = (data.get("body") or "").strip()[:2000]
                if not body:
                    continue
                msg = Message(thread_id=thread_id, sender_id=uid, body=body)
                db.add(msg)
                other_id = t.user_b if t.user_a == uid else t.user_a
                from .notify import create_notification
                create_notification(db, other_id, "dm", f"New message from @{user.handle}", f"/social/dm/{user.handle}")
                db.commit(); db.refresh(msg)
                await hub.broadcast(channel, {
                    "type": "msg", "handle": user.handle, "display": user.display_name or user.handle,
                    "body": body, "ts": msg.created_at.strftime("%H:%M"), "sender_id": uid,
                })
        except WebSocketDisconnect:
            pass
        finally:
            hub.leave(channel, ws)
    finally:
        db.close()


# ---------- Moderation ----------

@rt_router.post("/block/{handle}")
def block(handle: str, request: Request, db: Session = Depends(get_db), user: User = Depends(current_user)):
    target = db.scalar(select(User).where(User.handle == handle.lower()))
    if target and target.id != user.id and not db.get(Block, {"blocker_id": user.id, "blocked_id": target.id}):
        db.add(Block(blocker_id=user.id, blocked_id=target.id)); db.commit()
    return RedirectResponse(f"/social/u/{handle.lower()}", status_code=303)


@rt_router.post("/report")
def report(request: Request, db: Session = Depends(get_db), user: User = Depends(current_user),
           target_type: str = Form(...), target_id: int = Form(...), reason: str = Form("")):
    db.add(Report(reporter_id=user.id, target_type=target_type[:20], target_id=target_id, reason=reason[:280]))
    db.commit()
    return {"ok": True}


# ---------- HTML helpers (8D design) ----------

def _msg_line(sender: User | None, body: str, ts: str, mine: bool = False) -> str:
    name = esc(sender.display_name or sender.handle) if sender else "unknown"
    color = "var(--cyan)" if mine else "var(--violet)"
    return (f'<div style="margin-bottom:9px;font-size:13px;line-height:1.4">'
            f'<span style="color:{color};font-weight:500">{name}</span> '
            f'<span class="handle">{ts}</span>'
            f'<div style="color:#cdd7ea">{html.escape(body)}</div></div>')


def _chat_shell(title: str, subtitle: str, ws_path: str, history: str, dm: bool = False) -> str:
    return f"""<h1 style="margin-bottom:2px">{title}</h1><p class="lede">{subtitle}</p>
    <div class="card" style="display:flex;flex-direction:column;height:60vh">
      <div id="log" style="flex:1;overflow-y:auto;padding-right:6px">{history}</div>
      <div class="row" style="margin-top:10px;gap:8px">
        <input id="msg" placeholder="Message…" autocomplete="off" style="flex:1" maxlength="2000">
        <button id="send" class="btn" style="margin-top:0">Send</button>
      </div>
      <div id="seen" class="handle" style="margin-top:6px;height:14px;text-align:right"></div>
    </div>
    <script>
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(proto + '://' + location.host + '{ws_path}');
    const log = document.getElementById('log'), inp = document.getElementById('msg');
    const esc = s => s.replace(/[&<>]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c]));
    const DM = {str(dm).lower()};
    function line(d){{
      const div = document.createElement('div'); div.style.cssText='margin-bottom:9px;font-size:13px;line-height:1.4';
      div.innerHTML = '<span style="color:var(--violet);font-weight:500">'+esc(d.display||d.handle)+'</span> <span class="handle">'+esc(d.ts)+'</span><div style="color:#cdd7ea">'+esc(d.body)+'</div>';
      log.appendChild(div); log.scrollTop = log.scrollHeight;
    }}
    ws.onmessage = e => {{ const d = JSON.parse(e.data);
      if (d.type === 'read') {{ document.getElementById('seen').textContent = 'Seen'; return; }}
      line(d);
      if (DM) ws.send(JSON.stringify({{type:'read'}}));
    }};
    function send(){{ const v = inp.value.trim(); if(!v) return; ws.send(JSON.stringify({{body:v}})); inp.value=''; }}
    document.getElementById('send').onclick = send;
    inp.addEventListener('keydown', e => {{ if(e.key==='Enter') send(); }});
    ws.onopen = () => {{ log.scrollTop = log.scrollHeight; if (DM) ws.send(JSON.stringify({{type:'read'}})); }};
    </script>"""


def _chat_doc(title: str, body: str, user, active: str) -> str:
    return layout(title, body, user, active)
