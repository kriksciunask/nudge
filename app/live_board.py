"""Live audience board for the Nudge demo shop (presenter page + the shopper side of the shared state).

Shoppers: app.py calls publish() after every click; it writes one small record into the process-wide
LiveBoard (nudge/live.py), kept in st.cache_resource so every session in the server sees the same object.

Presenter: open  <app url>/?board=1  on the projector. It refreshes itself every 2 s (st.fragment).
Add  &key=<board_key>&menu=1  to get the presenter menu (hide a shopper, reset the board) and  &key=...&demo=12  to fill
the board with fake shoppers for rehearsal. board_key comes from st.secrets["board_key"] (fallback below).
"""

import hmac
import html
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

import streamlit as st

from nudge.decision import DISCOUNT_RATE
from nudge.live import LiveBoard, fill_demo, summary

FALLBACK_KEY = "mercato"      # used only when no board_key is set in secrets; set a real one on the cloud
QR_JS = (Path(__file__).resolve().parent / "static" / "qrcode.min.js").read_text()   # vendored qrcodejs 1.0.0 (MIT), no CDN
JUST_LEFT_SECONDS = 8         # how long a card stays highlighted after Leave
MAX_CARDS = 20
LEFT_NAME = ["No message", f"+{DISCOUNT_RATE:.0%}", "Reminder"]       # short, so a leaver label never wraps
ZONE_NAME = ["No message", f"Reminder + {DISCOUNT_RATE:.0%}", "Reminder only"]


@st.cache_resource
def get_store() -> LiveBoard:
    """One LiveBoard per server process, shared by every shopper session and the board page."""
    return LiveBoard()


def publish(sid: str, hist: list[float], zone: int, cart_value: float, n_cart: int, status: str,
            words: list[str], nick: str | None) -> None:
    """Shopper side: write this session's state. Never lets a board problem break the shop."""
    try:
        get_store().update(sid, hist=hist, zone=zone, cart_value=cart_value, n_cart=n_cart, status=status,
                           words=words, nick=nick)
    except Exception:  # noqa: BLE001
        pass


def _secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:  # noqa: BLE001  (no secrets file locally)
        return default


def presenter_ok() -> bool:
    given = str(st.query_params.get("key", ""))
    return bool(given) and hmac.compare_digest(given, _secret("board_key", FALLBACK_KEY))


def _manual_join_url() -> str:
    return _secret("join_url") or os.environ.get("NUDGE_JOIN_URL", "")


def is_local(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return host in ("localhost", "::1") or host.startswith("127.") or host == "0.0.0.0"


def join_url() -> str:
    """Where phones should go: a secret / env value when set, else this app's own address (no query, no key)."""
    manual = _manual_join_url()
    if manual:
        return manual
    try:
        parts = urlsplit(st.context.url)
        return f"{parts.scheme}://{parts.netloc}{parts.path}".rstrip("/") + "/"
    except Exception:  # noqa: BLE001
        return "http://localhost:8501/"


def _spark(hist: list[float], zone: int) -> str:
    w, h, pad = 92, 38, 4
    top = max(0.3, max(hist) * 1.1) if hist else 0.3
    pts = [(pad + (w - 2 * pad) * (i / max(len(hist) - 1, 1)), h - pad - (h - 2 * pad) * min(v, top) / top)
           for i, v in enumerate(hist)]
    if len(pts) == 1:                      # one value so far: a short flat stub, not a floating dot
        pts = [(pad, pts[0][1]), (w - pad, pts[0][1])]
    line = f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>' if len(pts) > 1 else ""
    x, y = pts[-1] if pts else (pad, h - pad)
    return (f'<svg class="sp" viewBox="0 0 {w} {h}" width="{w}" height="{h}" aria-hidden="true">{line}'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6"/></svg>')


BAG = ('<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.75" '
       'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 7h12l1 13H5L6 7Z"/>'
       '<path d="M9 7V6a3 3 0 0 1 6 0v1"/></svg>')


def _card(r: dict, now: float) -> str:
    name = ZONE_NAME[r["zone"]]
    just = r["status"] == "left" and r["left_at"] and now - r["left_at"] < JUST_LEFT_SECONDS
    if r["status"] == "bought":
        cls, line = "bought", "Bought, no message"
    elif just:
        cls, line = "left just" + (" fresh" if now - r["left_at"] < 3 else ""), f"Left → {LEFT_NAME[r['zone']]}"
    elif r["status"] == "left":
        cls, line = "left", f"Left → {LEFT_NAME[r['zone']]}"
    else:
        cls, line = "", name
    cart = f'<span class="cart" title="Has a cart">{BAG}€{r["cart_value"]:.0f}</span>' if r["n_cart"] and r["cart_value"] else ""
    return (f'<div class="bd-card z{r["zone"]} {cls}"><div class="r1"><b>{html.escape(r["nick"])}</b>{cart}</div>'
            f'<div class="r2"><span class="pct">{round(r["chance"] * 100)}<sup>%</sup></span>{_spark(r["hist"], r["zone"])}</div>'
            f'<div class="r3"><i></i>{html.escape(line)}</div></div>')


def _card_small(r: dict, now: float, grp: str = "") -> str:
    """Compact card for a crowded room: name, chance, decision (no sparkline)."""
    left = r["status"] == "left"
    just = left and r["left_at"] and now - r["left_at"] < JUST_LEFT_SECONDS
    cls = ("left just" + (" fresh" if now - r["left_at"] < 3 else "")) if just else ("left" if left else "")
    line = "Bought" if r["status"] == "bought" else (f"Left → {LEFT_NAME[r['zone']]}" if left else ZONE_NAME[r["zone"]])
    if r["status"] == "bought":
        cls = "bought"
    return (f'<div class="bd-card c {grp} z{r["zone"]} {cls}"><div class="r1"><b>{html.escape(r["nick"])}</b>'
            f'<span class="pct">{round(r["chance"] * 100)}<sup>%</sup></span></div>'
            f'<div class="r3"><i></i>{html.escape(line)}</div></div>')


def _body(rows: list[dict], now: float) -> str:
    if not rows:
        return ('<div class="bd-eyebrow"><i></i>Nudge · live audience board</div>'
                '<div class="bd-empty"><div class="big">Scan to join</div>'
                '<p>Open your phone camera on the code, then shop like you normally would. '
                'Your card appears here and Nudge\'s guess moves with every click.</p></div>')
    s = summary(rows)
    segs = "".join(f'<span class="seg z{z}" style="flex:{s[k]}"></span><span class="seg z{z} left" style="flex:{s[kl]}"></span>'
                   for z, k, kl in ((1, "discount", "left_discount"), (2, "reminder", "left_reminder"), (0, "none", "left_none")))
    cells = "".join(f'<div class="cell z{z}"><b>{s[k]}</b><span>{ZONE_NAME[z]}</span>'
                    + (f'<em>+{s[kl]} left</em>' if s[kl] else "") + '</div>'
                    for z, k, kl in ((1, "discount", "left_discount"), (2, "reminder", "left_reminder"), (0, "none", "left_none")))
    # shopping now first (most recent activity), then the "Already left" row, which always shows the newest leavers.
    # Above 15 active shoppers the cards turn compact (no sparkline) and more fit.
    now_rows = [r for r in rows if r["status"] != "left"]
    left_rows = [r for r in rows if r["status"] == "left"]
    compact = len(now_rows) > 9
    cap_now, cap_left = (24, 6) if compact else (9, 5)
    draw = (lambda r, n, g: _card_small(r, n, g)) if compact else (lambda r, n, g: _card(r, n))
    cards = "".join(draw(r, now, "act") for r in now_rows[:cap_now])
    if left_rows:
        cards += '<div class="bd-sep">Already left</div>' + "".join(draw(r, now, "lft") for r in left_rows[:cap_left])
    def more_text(cn: int, cl: int) -> str:
        extra = [f"+{len(now_rows) - cn} more shopping"] if len(now_rows) > cn else []
        extra += [f"+{len(left_rows) - cl} more left"] if len(left_rows) > cl else []
        return " · ".join(extra)
    # on narrower screens CSS shows fewer compact cards (12 shopping, 4 left), so that text has its own counts
    wide, narrow = more_text(cap_now, cap_left), more_text(12 if compact else cap_now, 4 if compact else cap_left)
    more = (f'<div class="bd-more"><span class="wide">{wide}</span><span class="narrow">{narrow}</span></div>'
            if (wide or narrow) else "")
    if s["left_carts"] >= 5:
        held = DISCOUNT_RATE * s["left_held_value"]
        none_given = s["left_carts"] - s["left_carts_discounted"]
        room = (f'<div class="bd-held"><div class="h"><b>€{held:,.0f}</b><span>held back</span></div>'
                f'<div class="h"><b>{none_given}</b><span>cart leavers got no discount: whether they return is unknown</span></div></div>')
        total = (f'<div class="bd-total">Leavers with a cart: <b>{s["left_carts"]}</b> · discount everyone gives <b>{s["left_carts"]}</b> · '
                 f'Nudge gives <b>{s["left_carts_discounted"]}</b> · €{held:,.0f} = {DISCOUNT_RATE:.0%} of the carts it did not discount</div>')
    else:
        room = ""
        total = '<div class="bd-total">Room total appears after 5 shoppers leave with a cart</div>'
    return (f'<div class="bd-eyebrow"><i></i>Nudge · live audience board</div>'
            f'<div class="bd-herorow"><div class="bd-hero"><span class="num">{s["shoppers"]}</span><span class="lab">joined · '
            f'<b>{s["shopping"]}</b> shopping now</span></div>{room}</div>'
            f'<div class="bd-sum"><div class="t">If they left this second, Nudge would send <span><span class="sw"></span>hatched = already left, final</span></div>'
            f'<div class="bar" role="img" aria-label="Nudge decisions across the room">{segs}</div>'
            f'<div class="cells">{cells}</div>'
            f'<div class="bd-total">{total}</div></div>'
            f'<div class="bd-grid{" c" if compact else ""}">{cards}</div>{more}')


@st.fragment(run_every=2)
def _live() -> None:
    rows = get_store().snapshot()
    now = time.time()
    # the layout changes between the empty state (big QR) and the room view (QR in a side column)
    cols = "minmax(0,1fr) minmax(0,1fr)" if not rows else "minmax(0,1fr) clamp(340px, 23vw, 460px)"
    st.markdown(f'<style>.st-key-bd {{ grid-template-columns: {cols}; }}</style>', unsafe_allow_html=True)  # (hidden by app.py)
    st.markdown(f'<div class="bd">{_body(rows, now)}</div>', unsafe_allow_html=True)


def _qr_box(url: str) -> None:
    short = url.replace("https://", "").replace("http://", "").rstrip("/")
    st.iframe(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@600;800&display=swap');
  html,body {{ margin:0; background:transparent; font-family:'Plus Jakarta Sans',-apple-system,'Segoe UI',Arial,sans-serif; color:#14161A; }}
  #w {{ box-sizing:border-box; display:flex; flex-direction:column; align-items:center; gap:14px; padding:22px 16px 18px;
        border:1px solid #E6E5E1; border-radius:20px; background:#fff; }}
  #qr {{ line-height:0; }} #qr img, #qr canvas {{ display:block; }}
  .t {{ font-size:26px; font-weight:800; letter-spacing:-.02em; }}
  .u {{ font-size:22px; font-weight:600; color:#5D636C; word-break:break-all; text-align:center; }}
</style>
<div id="w"><div id="qr"></div><div class="t">Scan to join</div><div class="u">{html.escape(short)}</div></div>
<script>{QR_JS}</script>
<script>
  var size = Math.max(160, Math.min(window.innerWidth - 56, window.innerHeight - 200, 540));
  try {{
    new QRCode(document.getElementById('qr'), {{ text: {url!r}.replace(/'/g, ''), width: size, height: size,
      colorDark: '#14161A', colorLight: '#FFFFFF', correctLevel: QRCode.CorrectLevel.M }});
  }} catch (e) {{ document.getElementById('qr').innerHTML = '<div class="u" style="font-size:26px">' + {html.escape(short)!r} + '</div>'; }}
</script>""", height=740)


CSS = """
<style>
.block-container { max-width: none !important; padding: 26px 44px 20px 44px !important; }
.stMain { overflow-x: hidden; }
.st-key-bd { display: grid; gap: 40px; align-items: start; }
.st-key-bd > *:nth-child(1) { grid-column: 1; min-width: 0; }
.st-key-bd > *:nth-child(2) { grid-column: 2; padding-top: 4px; }
.bd-eyebrow { display:flex; align-items:center; gap:10px; font-size:18px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color: var(--muted); }
.bd-eyebrow i { width:10px; height:10px; border-radius:50%; background: var(--ok); box-shadow: 0 0 0 4px var(--ok-soft); }
.bd-hero { display:flex; align-items:baseline; gap:22px; margin: 6px 0 14px; }
.bd-hero .num { font-size: clamp(84px, 13vh, 136px); font-weight: 800; letter-spacing:-.045em; line-height:1; color: var(--ink); }
.bd-hero .lab b { font-weight: 800; color: var(--ink); }
.bd-hero .lab { font-size: clamp(24px, 3.4vh, 34px); font-weight: 600; color: var(--ink-2); letter-spacing:-.01em; }
.bd-sum { margin-bottom: 22px; }
.bd-sum .t { font-size: 20px; font-weight: 600; color: var(--muted); margin-bottom: 12px; }
.bd-sum .bar { display:flex; height: 18px; border-radius: 9px; overflow:hidden; background: var(--soft-2); gap:3px; }
.bd-sum .t span { font-weight: 500; margin-left: 14px; }
.bd-sum .t .sw { display: inline-block; width: 38px; height: 14px; margin: 0 8px 0 0; vertical-align: -1px; border-radius: 4px; background-color: var(--ink-2);
  background-image: repeating-linear-gradient(135deg, rgba(255,255,255,.8) 0 3px, transparent 3px 8px); }
.bd-herorow { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 12px 48px; }
.bd-held { display: flex; gap: 40px; align-items: flex-start; padding-bottom: 8px; }
.bd-held .h { display: flex; align-items: center; gap: 14px; max-width: 330px; }
.bd-held b { font-size: clamp(56px, 8vh, 84px); font-weight: 800; letter-spacing: -.04em; line-height: 1; color: var(--ink); }
.bd-held span { font-size: 20px; font-weight: 700; line-height: 1.25; color: var(--ink-2); }
.bd-grid.c { grid-template-columns: repeat(auto-fill, minmax(214px, 1fr)); gap: 10px; }
.bd-card.c { min-height: 0; padding: 8px 12px; gap: 2px; }
.bd-card.c .r1 { align-items: baseline; } .bd-card.c .pct { font-size: 28px; } .bd-card.c .pct sup { font-size: 16px; }
.bd-card.c.just { padding: 7px 11px; }
.bd-more .narrow { display: none; }
@media (max-width: 1600px) {
  .bd-grid.c > .act:nth-child(n+13) { display: none; }
  .bd-grid.c > .bd-sep + .lft + .lft + .lft + .lft ~ .lft { display: none; }
  .bd-more .wide { display: none; } .bd-more .narrow { display: inline; }
}
.cell em { font-style: normal; font-size: 20px; font-weight: 600; color: var(--faint); }
.bd-sep { grid-column: 1 / -1; margin-top: 6px; font-size: 20px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
.seg.z0 { background: var(--faint); } .seg.z1 { background: var(--accent); } .seg.z2 { background: var(--ink); }
.seg.left { background-image: repeating-linear-gradient(135deg, rgba(255,255,255,.8) 0 3px, transparent 3px 8px); }

.bd-sum .cells { display:grid; grid-template-columns: repeat(3, 1fr); gap: 18px; margin-top: 14px; }
.cell { display:flex; align-items:baseline; gap:12px; }
.cell b { font-size: clamp(38px, 5.2vh, 54px); font-weight: 800; letter-spacing:-.03em; line-height:1; }
.cell span { font-size: 22px; font-weight: 600; color: var(--muted); }
.cell.z1 b { color: var(--accent); } .cell.z2 b { color: var(--ink); } .cell.z0 b { color: var(--muted); }
.bd-grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(224px, 1fr)); gap: 14px; }
.bd-card { border: 1px solid var(--line); border-radius: 16px; padding: 12px 16px 12px; background: var(--paper); min-height: 112px;
  display:flex; flex-direction:column; justify-content:space-between; gap:4px; }
.bd-card .r1 { display:flex; justify-content:space-between; align-items:center; gap:8px; }
.bd-card .r1 b { font-size: 19px; font-weight: 700; color: var(--ink); white-space: nowrap; overflow:hidden; text-overflow: ellipsis; }
.bd-card .cart { display:flex; align-items:center; gap:4px; font-size: 18px; font-weight:600; color: var(--muted); flex: none; }
.bd-card .r2 { display:flex; justify-content:space-between; align-items:center; }
.bd-card .pct { font-size: 40px; font-weight: 800; letter-spacing:-.04em; line-height:1; }
.bd-card .pct sup { font-size: 20px; font-weight: 600; letter-spacing:0; margin-left:1px; vertical-align: baseline; position: relative; top: -.55em; }
.bd-card .sp { flex:none; overflow: visible; }
.bd-card .sp polyline { fill:none; stroke-width:2.5; stroke-linecap:round; stroke-linejoin:round; }
.bd-card .r3 { white-space: nowrap; display:flex; align-items:center; gap:8px; font-size: 18px; font-weight:700; line-height:1.2; }
.bd-card .r3 i { width:10px; height:10px; border-radius:50%; flex:none; }
.bd-card.z0 .pct, .bd-card.z0 .r3 { color: var(--muted); } .bd-card.z0 .sp polyline { stroke: var(--faint); } .bd-card.z0 .sp circle, .bd-card.z0 .r3 i { fill: var(--faint); background: var(--faint); }
.bd-card.z1 .pct, .bd-card.z1 .r3 { color: var(--accent); } .bd-card.z1 .sp polyline { stroke: var(--accent); } .bd-card.z1 .sp circle, .bd-card.z1 .r3 i { fill: var(--accent); background: var(--accent); }
.bd-card.z2 .pct, .bd-card.z2 .r3 { color: var(--ink); } .bd-card.z2 .sp polyline { stroke: var(--ink); } .bd-card.z2 .sp circle, .bd-card.z2 .r3 i { fill: var(--ink); background: var(--ink); }
.bd-card.left { background: var(--soft); border-color: var(--line-2); }
.bd-card.bought { background: var(--ok-soft); border-color: transparent; } .bd-card.bought .pct, .bd-card.bought .r3 { color: var(--ok); }
.bd-card.bought .sp polyline { stroke: var(--ok); } .bd-card.bought .sp circle, .bd-card.bought .r3 i { fill: var(--ok); background: var(--ok); }
.bd-card.just { border-width: 2px; padding: 11px 15px 11px; }
.bd-card.fresh { animation: bd-flash 1s ease-out 1; }
.bd-card.just.z0 { border-color: var(--muted); } .bd-card.just.z1 { border-color: var(--accent); background: var(--accent-soft); } .bd-card.just.z2 { border-color: var(--ink); }
@keyframes bd-flash { 0% { box-shadow: 0 0 0 0 rgba(20,22,26,.28); } 100% { box-shadow: 0 0 0 16px rgba(20,22,26,0); } }
.bd-total { margin-top: 16px; font-size: 22px; font-weight: 500; line-height: 1.4; color: var(--ink-2); } .bd-total b { font-weight: 800; color: var(--ink); }
.bd-warn { margin: 0 0 14px; padding: 14px 16px; border-radius: 14px; background: #FBEDEA; border: 1.5px solid var(--accent); color: #8E2616; font-size: 20px; font-weight: 700; line-height: 1.35; }
.bd-more { margin-top: 14px; font-size: 20px; font-weight: 600; color: var(--muted); }
.bd-empty { padding-top: 14vh; max-width: 760px; }
.bd-empty .big { font-size: clamp(72px, 12vh, 128px); font-weight: 800; letter-spacing:-.045em; line-height:1; }
.bd-empty p { margin-top: 22px; font-size: 26px; line-height: 1.45; color: var(--ink-2); }
.st-key-bd_ctl { margin-top: 18px; }
.st-key-bd_ctl button { min-height: 36px; font-size: 16px; }
@media (max-height: 960px) {
  .block-container { padding-top: 18px !important; }
  .bd-hero { margin: 2px 0 8px; } .bd-hero .num { font-size: 84px; }
  .bd-sum { margin-bottom: 14px; } .bd-sum .t { margin-bottom: 8px; } .bd-sum .cells { margin-top: 8px; }
  .cell b { font-size: 36px; } .bd-total { margin-top: 10px; font-size: 20px; }
  .bd-grid { gap: 10px; } .bd-card { min-height: 92px; padding: 8px 14px; } .bd-card.just { padding: 7px 13px; }
  .bd-card .pct { font-size: 32px; }
}
@media (prefers-reduced-motion: reduce) { .bd-card.fresh { animation: none; } }
@media (max-width: 1100px) { .st-key-bd { grid-template-columns: minmax(0,1fr) !important; } .st-key-bd > *:nth-child(n) { grid-column: 1; } }
</style>
"""


def render() -> None:
    """The presenter page. Called by app.py when the URL has ?board=1; the caller stops the script after."""
    st.markdown(CSS, unsafe_allow_html=True)
    ok = presenter_ok()
    store = get_store()
    menu = ok and str(st.query_params.get("menu", "")) == "1"      # the menu stays off the projector unless asked for
    if ok and str(st.query_params.get("demo", "")).isdigit() and not any(r["demo"] for r in store.snapshot(True)):
        fill_demo(store, min(int(st.query_params["demo"]), 40))
    with st.container(key="bd"):
        with st.container(key="bd_main"):
            _live()
            if menu:
                with st.container(key="bd_ctl"):
                    with st.popover("Presenter", icon=":material/tune:"):
                        who = st.text_input("Hide a shopper (their nickname)", key="bd_hide_in")
                        if st.button("Hide", key="bd_hide") and who:
                            st.toast(f"Hidden: {store.hide(who)}")
                        if st.button("Reset board", key="bd_reset", type="primary"):
                            store.reset()
                            st.toast("Board cleared")
        with st.container(key="bd_side"):
            url = join_url()
            if is_local(url) and not _manual_join_url():
                st.markdown('<div class="bd-warn">Phones can\'t reach localhost. Set NUDGE_JOIN_URL.</div>', unsafe_allow_html=True)
            _qr_box(url)
