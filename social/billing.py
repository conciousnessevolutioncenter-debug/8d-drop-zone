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
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import get_db
from .models import User, WebhookEvent
from .security import current_user, optional_user
from .ui import layout, esc
from . import entitlements as ent

billing_router = APIRouter(prefix="/social", tags=["social-billing"])

# Stripe is optional: everything below stays dormant until STRIPE_SECRET_KEY is
# set. Tier <-> Stripe Price ID mapping comes from env so prices live in your
# Stripe dashboard, not the code.
_TIER_PRICE_ENV = {"creator": "STRIPE_PRICE_CREATOR", "producer": "STRIPE_PRICE_PRODUCER", "studio": "STRIPE_PRICE_STUDIO"}


def _stripe():
    key = os.environ.get("STRIPE_SECRET_KEY")
    if not key:
        return None
    try:
        import stripe
    except Exception:
        return None
    stripe.api_key = key
    return stripe


def stripe_enabled() -> bool:
    return bool(os.environ.get("STRIPE_SECRET_KEY"))


def _price_for(tier: str) -> str | None:
    return os.environ.get(_TIER_PRICE_ENV.get(tier, ""), None)


def _tier_for_price(price_id: str) -> str | None:
    for tier, env in _TIER_PRICE_ENV.items():
        if os.environ.get(env) == price_id:
            return tier
    return None

_PERKS = {
    "free":     ["Listen / watch / transcribe", "5 analyses / month", "Preview only — no downloads"],
    "creator":  ["Everything in Free", "Watermark-free shares (clean video & card)", "Unlimited analysis (credits)", "Downloads (standard quality)", "Basic mixing — levels & fades"],
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
    manage = ""
    if ent.is_paid(user) and stripe_enabled() and user.stripe_customer_id:
        manage = '<form method="post" action="/social/billing/portal" style="margin-left:8px"><button class="btn ghost" type="submit">Manage subscription</button></form>'
    status = (f'<div class="card"><div class="row"><div><div style="font-size:15px;font-weight:600">'
              f'Current plan: {esc(ent.label(user))}</div>'
              f'<div class="handle">credits: {user.credits} · unlocks: {esc(capline)}</div></div>'
              f'<a href="/social/studio" class="btn ghost" style="margin-left:auto">Open studio</a>{manage}</div></div>')
    if os.environ.get("SOCIAL_DEV"):
        note = ''
    elif stripe_enabled():
        note = '<p class="muted" style="margin-top:12px">Secure checkout by Stripe. Cancel anytime from Manage subscription.</p>'
    else:
        note = '<p class="muted" style="margin-top:12px">Secure checkout (Stripe) activates once billing keys are set — these plans are wired and ready.</p>'
    return HTMLResponse(layout("Plans", '<h1>Plans &amp; billing</h1><p class="lede">Upgrade for downloads, mixing, and the 8D editor.</p>' + status + grid + note, user, "billing"))


@billing_router.post("/billing/checkout")
def checkout(request: Request, db: Session = Depends(get_db), tier: str = Form(...), user: User = Depends(current_user)):
    stripe = _stripe()
    price = _price_for(tier)
    if not stripe or not price:
        body = ('<h1>Almost there</h1><p class="lede">Card checkout isn\'t connected yet. '
                'Once Stripe keys + price IDs are set, this button opens secure checkout.</p>'
                '<a href="/social/billing" class="btn ghost">Back to plans</a>')
        return HTMLResponse(layout("Checkout", body, user, "billing"))
    # Reuse or create the Stripe customer (so the portal + webhooks can find the user).
    if not user.stripe_customer_id:
        cust = stripe.Customer.create(email=user.email, metadata={"uid": str(user.id)})
        user.stripe_customer_id = cust["id"]; db.commit()
    base = str(request.base_url).rstrip("/")
    session = stripe.checkout.Session.create(
        mode="subscription",
        customer=user.stripe_customer_id,
        line_items=[{"price": price, "quantity": 1}],
        success_url=f"{base}/social/billing?upgraded=1",
        cancel_url=f"{base}/social/billing",
        client_reference_id=str(user.id),
        metadata={"uid": str(user.id), "tier": tier},
        subscription_data={"metadata": {"uid": str(user.id), "tier": tier}},
    )
    return RedirectResponse(session["url"], status_code=303)


@billing_router.post("/billing/portal")
def portal(request: Request, db: Session = Depends(get_db), user: User = Depends(current_user)):
    stripe = _stripe()
    if not stripe or not user.stripe_customer_id:
        return RedirectResponse("/social/billing", status_code=303)
    sess = stripe.billing_portal.Session.create(
        customer=user.stripe_customer_id,
        return_url=f"{str(request.base_url).rstrip('/')}/social/billing",
    )
    return RedirectResponse(sess["url"], status_code=303)


@billing_router.post("/stripe/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    stripe = _stripe()
    secret = os.environ.get("STRIPE_WEBHOOK_SECRET")
    payload = await request.body()
    if not stripe or not secret:
        return JSONResponse({"error": "stripe not configured"}, status_code=503)
    try:
        event = stripe.Webhook.construct_event(payload, request.headers.get("stripe-signature", ""), secret)
    except Exception as exc:
        return JSONResponse({"error": f"bad signature: {exc}"}, status_code=400)

    # Idempotency: skip events we've already handled.
    eid = event["id"]
    if db.get(WebhookEvent, eid):
        return JSONResponse({"ok": True, "duplicate": True})

    etype = event["type"]
    obj = event["data"]["object"]
    if etype == "checkout.session.completed":
        uid = (obj.get("metadata") or {}).get("uid") or obj.get("client_reference_id")
        tier = (obj.get("metadata") or {}).get("tier")
        cust = obj.get("customer")
        if uid and tier:
            u = db.get(User, int(uid))
            if u:
                ent.set_tier(u, tier)
                if cust:
                    u.stripe_customer_id = cust
    elif etype in ("customer.subscription.deleted", "customer.subscription.paused"):
        cust = obj.get("customer")
        u = db.scalar(select(User).where(User.stripe_customer_id == cust)) if cust else None
        if u:
            ent.set_tier(u, "free")
    elif etype == "customer.subscription.updated":
        # Reflect plan changes (upgrade/downgrade) from the subscription's price.
        cust = obj.get("customer")
        u = db.scalar(select(User).where(User.stripe_customer_id == cust)) if cust else None
        items = (((obj.get("items") or {}).get("data")) or [{}])
        price_id = ((items[0].get("price") or {}).get("id")) if items else None
        status = obj.get("status")
        if u and price_id and status in ("active", "trialing"):
            t = _tier_for_price(price_id)
            if t:
                ent.set_tier(u, t)

    db.add(WebhookEvent(stripe_event_id=eid))
    db.commit()
    return JSONResponse({"ok": True})


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
