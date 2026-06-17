"""Billing / upgrade UI + tier management.

Shows the four tiers in the 8D design and the user's current entitlements.
Real Stripe checkout lands in the payments phase; until then the Upgrade
buttons either explain that (prod) or switch tiers for testing (when SOCIAL_DEV
is set). The webhook will call entitlements.set_tier — the same path the dev
switch uses — so nothing here changes when billing goes live.
"""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from .db import get_db
from .models import User
from .security import current_user, optional_user
from .ui import layout, esc
from . import entitlements as ent

billing_router = APIRouter(prefix="/social", tags=["social-billing"])

_PERKS = {
    "free":     ["Listen / watch / transcribe", "5 analyses / month", "Preview only — no downloads"],
    "creator":  ["Everything in Free", "Unlimited analysis (credits)", "Downloads (standard quality)", "Basic mixing — levels & fades"],
    "producer": ["Everything in Creator", "Full mixing — EQ, comp, reverb, multitrack", "High-quality downloads"],
    "studio":   ["Everything in Producer", "8D spatial editor (exclusive)", "AI stem separation", "Studio-fidelity downloads"],
}
_PRICE = {"free": "Free", "creator": "$6/mo", "producer": "$14/mo", "studio": "$29/mo"}


@billing_router.get("/billing", response_class=HTMLResponse)
def billing(request: Request, db: Session = Depends(get_db)):
    user = optional_user(request, db)
    if not user:
        return RedirectResponse("/social/login", status_code=303)
    dev = bool(os.environ.get("SOCIAL_DEV"))
    cur = ent.tier_of(user)
    cards = ""
    for t in ent.ORDER:
        info = TIERS_meta = ent.TIERS[t]
        active = (t == cur)
        perks = "".join(f'<li>{esc(p)}</li>' for p in _PERKS[t])
        if active:
            btn = '<span class="pro" style="display:inline-block;margin-top:12px">CURRENT PLAN</span>'
        elif dev:
            btn = (f'<form method="post" action="/social/dev/set-tier"><input type="hidden" name="tier" value="{t}">'
                   f'<button class="btn" type="submit">Switch to {esc(info["label"])}</button></form>')
        elif t == "free":
            btn = ''
        else:
            btn = '<form method="post" action="/social/billing/checkout"><input type="hidden" name="tier" value="%s">' \
                  '<button class="btn" type="submit">Upgrade</button></form>' % t
        border = "border:2px solid var(--cyan)" if active else "border:1px solid var(--hair)"
        cards += (f'<div class="card" style="{border};margin-top:0">'
                  f'<div class="tag" style="color:var(--cyan)">{esc(info["label"]).upper()}</div>'
                  f'<div style="font-size:22px;font-weight:600;margin:6px 0 2px">{_PRICE[t]}</div>'
                  f'<ul class="muted" style="padding-left:16px;margin:8px 0 0;font-size:12.5px;line-height:1.7">{perks}</ul>'
                  f'{btn}</div>')
    grid = f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin-top:14px">{cards}</div>'
    capline = ", ".join(sorted(ent.caps(user))) or "analyze"
    status = (f'<div class="card"><div class="row"><div><div style="font-size:15px;font-weight:600">'
              f'Current plan: {esc(ent.label(user))}</div>'
              f'<div class="handle">credits: {user.credits} · unlocks: {esc(capline)}</div></div>'
              f'<a href="/social/studio" class="btn ghost" style="margin-left:auto">Open studio</a></div></div>')
    note = '' if os.environ.get("SOCIAL_DEV") else '<p class="muted" style="margin-top:12px">Secure checkout (Stripe) activates in the payments phase — these plans are wired and ready.</p>'
    return HTMLResponse(layout("Plans", '<h1>Plans &amp; billing</h1><p class="lede">Upgrade for downloads, mixing, and the 8D editor.</p>' + status + grid + note, user, "billing"))


@billing_router.post("/billing/checkout")
def checkout(request: Request, db: Session = Depends(get_db), tier: str = Form(...), user: User = Depends(current_user)):
    # Placeholder until Stripe is wired: real flow creates a Checkout Session and
    # redirects to Stripe; the webhook then calls entitlements.set_tier.
    body = ('<h1>Almost there</h1><p class="lede">Stripe checkout isn\'t connected yet — '
            'the payments phase wires it to this exact button.</p>'
            '<a href="/social/billing" class="btn ghost">Back to plans</a>')
    return HTMLResponse(layout("Checkout", body, user, "billing"))


@billing_router.post("/dev/set-tier")
def dev_set_tier(request: Request, db: Session = Depends(get_db), tier: str = Form(...), user: User = Depends(current_user)):
    if not os.environ.get("SOCIAL_DEV"):
        return RedirectResponse("/social/billing", status_code=303)
    ent.set_tier(user, tier)
    if ent.is_paid(user) and user.credits == 0:
        user.credits = 500  # seed some credits for testing the credit model
    db.commit()
    return RedirectResponse("/social/billing", status_code=303)


@billing_router.get("/studio", response_class=HTMLResponse)
def studio(request: Request, db: Session = Depends(get_db)):
    user = optional_user(request, db)
    if not user:
        return RedirectResponse("/social/login", status_code=303)
    rows = ""
    feature_caps = [("download", "Download renders"), ("mix_basic", "Basic mixing"),
                    ("mix_full", "Full mixing (EQ/comp/reverb)"), ("eight_d", "8D spatial editor"),
                    ("stems", "AI stem separation")]
    for cap, name in feature_caps:
        ok = ent.can(user, cap)
        mark = '<span style="color:#5DCAA5">● unlocked</span>' if ok else '<span class="muted">○ locked</span>'
        link = f'<a href="/" class="tag" style="color:var(--cyan)">OPEN</a>' if (ok and cap == "eight_d") else ""
        rows += f'<div class="row" style="justify-content:space-between;padding:9px 0;border-bottom:1px solid var(--hair)"><span>{esc(name)}</span><span class="row" style="gap:10px">{mark}{link}</span></div>'
    body = (f'<h1>Studio</h1><p class="lede">Your unlocked tools on the <b>{esc(ent.label(user))}</b> plan.</p>'
            f'<div class="card">{rows}</div>'
            f'<a href="/social/billing" class="btn" style="margin-top:14px">Manage plan</a>')
    return HTMLResponse(layout("Studio", body, user, "billing"))
