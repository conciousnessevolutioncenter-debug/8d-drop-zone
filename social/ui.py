"""Shared HTML shell for the social pages — reuses the 8D Engine design tokens
(dark void background, cyan/violet accents, mono uppercase labels, hairline
cards) so the social section looks like the same product as the audio tool."""
from __future__ import annotations

import html

from .models import User

CSS = """
:root{
  --void:#04060f; --panel:rgba(13,20,38,.5); --hair:rgba(255,255,255,.09);
  --hair2:rgba(255,255,255,.16); --ink:#e9f1ff; --soft:#8a93a8; --mut:#6b7488;
  --cyan:#62e0ff; --violet:#9d8bff; --mono:ui-monospace,"SFMono-Regular",Menlo,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:linear-gradient(180deg,#02040c 0%,#04060f 45%,#060912 100%);
  color:var(--ink);font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;min-height:100vh;}
a{color:inherit;text-decoration:none}
.wrap{width:min(900px,calc(100vw - 32px));margin:0 auto;padding:18px 0 60px}
.topbar{display:flex;align-items:center;gap:14px;padding:8px 4px 16px;border-bottom:1px solid var(--hair)}
.word{font-family:var(--mono);letter-spacing:.22em;font-size:13px}
.tag{font-family:var(--mono);letter-spacing:.16em;font-size:10px;color:var(--soft)}
.nav{margin-left:auto;display:flex;gap:16px;font-family:var(--mono);font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--soft)}
.nav a.active{color:var(--cyan)}
.pro{font-family:var(--mono);font-size:10px;letter-spacing:.14em;color:#0a0e18;background:var(--cyan);border-radius:999px;padding:4px 9px}
.avatar{width:28px;height:28px;border-radius:50%;background:linear-gradient(135deg,var(--cyan),var(--violet));display:inline-block;vertical-align:middle}
.card{border:1px solid var(--hair);border-radius:14px;background:var(--panel);padding:15px 17px;margin-top:14px}
h1{font-size:clamp(20px,2.4vw,26px);font-weight:600;letter-spacing:-.02em;margin:18px 0 4px}
.lede{color:var(--soft);font-size:13.5px;margin:0 0 6px}
label{display:block;font-family:var(--mono);font-size:10px;letter-spacing:.16em;text-transform:uppercase;color:var(--soft);margin:12px 0 6px}
input,textarea{width:100%;background:rgba(4,7,15,.7);border:1px solid var(--hair);border-radius:11px;color:var(--ink);padding:11px 13px;font-size:14px;outline:none}
input:focus,textarea:focus{border-color:var(--cyan)}
textarea{resize:vertical;min-height:70px;font-family:inherit}
.btn{display:inline-block;border:none;cursor:pointer;font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;
  color:#06101c;background:linear-gradient(135deg,var(--cyan),var(--violet));border-radius:999px;padding:11px 20px;margin-top:14px}
.btn.ghost{background:transparent;border:1px solid var(--hair2);color:var(--ink)}
.muted{color:var(--soft);font-size:12.5px}
.handle{font-family:var(--mono);font-size:11px;letter-spacing:.06em;color:var(--soft)}
.err{color:#ff8a8a;font-size:12.5px;margin-top:10px}
.row{display:flex;align-items:center;gap:10px}
.post-actions{display:flex;gap:22px;margin-top:11px;font-family:var(--mono);font-size:11px;color:var(--soft)}
.post-actions a:hover{color:var(--cyan)}
.big{width:54px;height:54px;border-radius:50%;background:linear-gradient(135deg,var(--violet),var(--cyan))}
.split{display:flex;gap:18px;font-family:var(--mono);font-size:11px;color:var(--soft);margin-top:8px}
.split b{color:var(--ink)}
"""


def avatar_html(user, cls: str = "avatar") -> str:
    """Either the user's uploaded avatar image or the gradient placeholder."""
    if user is not None and getattr(user, "avatar_url", None):
        return f'<img class="{cls}" src="{html.escape(user.avatar_url)}" alt="" style="object-fit:cover">'
    return f'<span class="{cls}"></span>'


def layout(title: str, body: str, user: User | None = None, active: str = "") -> str:
    nav_items = [("feed", "/social", "Feed"), ("rooms", "/social/rooms", "Rooms"),
                 ("inbox", "/social/inbox", "Inbox"), ("people", "/social/people", "People")]
    if user:
        from . import entitlements as _ent
        nav_items.append(("billing", "/social/billing", "Plans"))
        nav_items.append(("me", f"/social/u/{user.handle}", "Profile"))
        badge = f'<span class="pro">{html.escape(_ent.label(user)).upper()}</span>' if _ent.is_paid(user) else ""
        bell = ""
        try:
            from .db import SessionLocal
            from .notify import unread_count
            _db = SessionLocal()
            try:
                n = unread_count(_db, user.id)
            finally:
                _db.close()
            count = f'<span style="color:#06101c;background:var(--cyan);border-radius:999px;font-size:9px;padding:1px 5px;margin-left:3px">{n}</span>' if n else ""
            bell = f'<a href="/social/notifications" class="tag" title="Notifications">◉{count}</a>'
        except Exception:
            bell = ""
        right = bell + badge + f'<a href="/social/logout" class="tag">LOGOUT</a>' + avatar_html(user)
    else:
        right = '<a href="/social/login" class="tag">LOGIN</a><a href="/social/register" class="pro">JOIN</a>'
    nav = "".join(
        f'<a href="{href}" class="{"active" if key == active else ""}">{label}</a>'
        for key, href, label in nav_items
    )
    verify_banner = ""
    if user is not None and not getattr(user, "email_verified", True):
        verify_banner = ('<div class="card" style="border-color:rgba(98,224,255,.3);background:rgba(98,224,255,.06);margin-top:12px;padding:10px 14px">'
                         '<span class="muted">Please verify your email to secure your account. </span>'
                         '<a href="/social/verify/resend" style="color:var(--cyan)">Resend link</a></div>')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · 8D Engine</title><style>{CSS}</style></head><body>
<div class="wrap">
  <div class="topbar">
    <a href="/social"><span class="word">THE 8D ENGINE</span></a>
    <span class="tag">// SOCIAL</span>
    <nav class="nav">{nav}</nav>
    <span class="row">{right}</span>
  </div>
  {verify_banner}
  {body}
</div></body></html>"""


def esc(s) -> str:
    return html.escape(str(s or ""))
