"""Live audience board for the Nudge demo shop (presenter page + the shopper side of the shared state).

Shoppers: app.py calls publish() after every click; it writes one small record into the process-wide
LiveBoard (nudge/live.py), kept in st.cache_resource so every session in the server sees the same object.

Presenter: open  <app url>/?board=1  on the projector. It refreshes itself every 2 s (st.fragment).
Add  &key=<board_key>  to get the presenter menu (hide a shopper, reset the board) and  &demo=12  to fill
the board with fake shoppers for rehearsal. board_key comes from st.secrets["board_key"] (fallback below).
"""

import hmac
import html
import os
import time
from urllib.parse import urlsplit

import streamlit as st

from nudge.decision import DISCOUNT_RATE
from nudge.live import LiveBoard, fill_demo, summary

FALLBACK_KEY = "mercato"      # used only when no board_key is set in secrets; set a real one on the cloud
QR_JS = "https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js"
JUST_LEFT_SECONDS = 8         # how long a card stays highlighted after Leave
MAX_CARDS = 24
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


def join_url() -> str:
    """Where phones should go: a secret / env value when set, else this app's own address (no query, no key)."""
    manual = _secret("join_url") or os.environ.get("NUDGE_JOIN_URL", "")
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
        cls, line = "left just", f"Just left → {name}"
    elif r["status"] == "left":
        cls, line = "left", f"Left · {name}"
    else:
        cls, line = "", name
    cart = f'<span class="cart" title="Has a cart">{BAG}€{r["cart_value"]:.0f}</span>' if r["n_cart"] and r["cart_value"] else ""
    return (f'<div class="bd-card z{r["zone"]} {cls}"><div class="r1"><b>{html.escape(r["nick"])}</b>{cart}</div>'
            f'<div class="r2"><span class="pct">{round(r["chance"] * 100)}<sup>%</sup></span>{_spark(r["hist"], r["zone"])}</div>'
            f'<div class="r3"><i></i>{html.escape(line)}</div></div>')


def _body(rows: list[dict], now: float) -> str:
    if not rows:
        return ('<div class="bd-eyebrow"><i></i>Nudge · live audience board</div>'
                '<div class="bd-empty"><div class="big">Scan to join</div>'
                '<p>Open your phone camera on the code, then shop like you normally would. '
                'Your card appears here and Nudge\'s guess moves with every click.</p></div>')
    s = summary(rows)
    segs = "".join(f'<span class="seg z{z}" style="flex:{s[k]}"></span>'
                   for z, k in ((1, "discount"), (2, "reminder"), (0, "none")))
    cells = "".join(f'<div class="cell z{z}"><b>{s[k]}</b><span>{ZONE_NAME[z]}</span></div>'
                    for z, k in ((1, "discount"), (2, "reminder"), (0, "none")))
    cards = "".join(_card(r, now) for r in rows[:MAX_CARDS])
    more = f'<div class="bd-more">+{len(rows) - MAX_CARDS} more shoppers</div>' if len(rows) > MAX_CARDS else ""
    n = s["shoppers"]
    return (f'<div class="bd-eyebrow"><i></i>Nudge · live audience board</div>'
            f'<div class="bd-hero"><span class="num">{n}</span><span class="lab">{"shopper" if n == 1 else "shoppers"} '
            f'in the shop right now</span></div>'
            f'<div class="bd-sum"><div class="t">If they left this second, Nudge would send</div>'
            f'<div class="bar" role="img" aria-label="Nudge decisions across the room">{segs}</div>'
            f'<div class="cells">{cells}</div></div>'
            f'<div class="bd-grid">{cards}</div>{more}')


@st.fragment(run_every=2)
def _live() -> None:
    rows = get_store().snapshot()
    now = time.time()
    # the layout changes between the empty state (big QR) and the room view (QR in a side column)
    cols = "minmax(0,1fr) minmax(0,1fr)" if not rows else "minmax(0,1fr) 340px"
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
  .u {{ font-size:19px; font-weight:600; color:#5D636C; word-break:break-all; text-align:center; }}
</style>
<div id="w"><div id="qr"></div><div class="t">Scan to join</div><div class="u">{html.escape(short)}</div></div>
<script src="{QR_JS}"></script>
<script>
  var size = Math.max(160, Math.min(window.innerWidth - 64, window.innerHeight - 200, 540));
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
.bd-hero .lab { font-size: clamp(24px, 3.4vh, 34px); font-weight: 600; color: var(--ink-2); letter-spacing:-.01em; }
.bd-sum { margin-bottom: 22px; }
.bd-sum .t { font-size: 20px; font-weight: 600; color: var(--muted); margin-bottom: 12px; }
.bd-sum .bar { display:flex; height: 18px; border-radius: 9px; overflow:hidden; background: var(--soft-2); gap:3px; }
.seg.z0 { background: var(--faint); } .seg.z1 { background: var(--accent); } .seg.z2 { background: var(--ink); }
.bd-sum .cells { display:grid; grid-template-columns: repeat(3, 1fr); gap: 18px; margin-top: 14px; }
.cell { display:flex; align-items:baseline; gap:12px; }
.cell b { font-size: clamp(38px, 5.2vh, 54px); font-weight: 800; letter-spacing:-.03em; line-height:1; }
.cell span { font-size: 22px; font-weight: 600; color: var(--muted); }
.cell.z1 b { color: var(--accent); } .cell.z2 b { color: var(--ink); } .cell.z0 b { color: var(--muted); }
.bd-grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(236px, 1fr)); gap: 14px; }
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
.bd-card .r3 { display:flex; align-items:center; gap:8px; font-size: 18px; font-weight:700; line-height:1.2; }
.bd-card .r3 i { width:10px; height:10px; border-radius:50%; flex:none; }
.bd-card.z0 .pct, .bd-card.z0 .r3 { color: var(--muted); } .bd-card.z0 .sp polyline { stroke: var(--faint); } .bd-card.z0 .sp circle, .bd-card.z0 .r3 i { fill: var(--faint); background: var(--faint); }
.bd-card.z1 .pct, .bd-card.z1 .r3 { color: var(--accent); } .bd-card.z1 .sp polyline { stroke: var(--accent); } .bd-card.z1 .sp circle, .bd-card.z1 .r3 i { fill: var(--accent); background: var(--accent); }
.bd-card.z2 .pct, .bd-card.z2 .r3 { color: var(--ink); } .bd-card.z2 .sp polyline { stroke: var(--ink); } .bd-card.z2 .sp circle, .bd-card.z2 .r3 i { fill: var(--ink); background: var(--ink); }
.bd-card.left { background: var(--soft); border-color: var(--line-2); }
.bd-card.bought { background: var(--ok-soft); border-color: transparent; } .bd-card.bought .pct, .bd-card.bought .r3 { color: var(--ok); }
.bd-card.bought .sp polyline { stroke: var(--ok); } .bd-card.bought .sp circle, .bd-card.bought .r3 i { fill: var(--ok); background: var(--ok); }
.bd-card.just { border-width: 2px; padding: 11px 15px 11px; animation: bd-ring 1.6s ease-out infinite; }
.bd-card.just.z0 { border-color: var(--muted); } .bd-card.just.z1 { border-color: var(--accent); background: var(--accent-soft); } .bd-card.just.z2 { border-color: var(--ink); }
@keyframes bd-ring { 0% { box-shadow: 0 0 0 0 rgba(201,58,37,.35); } 100% { box-shadow: 0 0 0 14px rgba(201,58,37,0); } }
.bd-more { margin-top: 14px; font-size: 20px; font-weight: 600; color: var(--muted); }
.bd-empty { padding-top: 14vh; max-width: 760px; }
.bd-empty .big { font-size: clamp(72px, 12vh, 128px); font-weight: 800; letter-spacing:-.045em; line-height:1; }
.bd-empty p { margin-top: 22px; font-size: 26px; line-height: 1.45; color: var(--ink-2); }
.st-key-bd_ctl { margin-top: 18px; }
.st-key-bd_ctl button { min-height: 36px; font-size: 16px; }
@media (prefers-reduced-motion: reduce) { .bd-card.just { animation: none; } }
@media (max-width: 1100px) { .st-key-bd { grid-template-columns: minmax(0,1fr) !important; } .st-key-bd > *:nth-child(n) { grid-column: 1; } }
</style>
"""


def render() -> None:
    """The presenter page. Called by app.py when the URL has ?board=1; the caller stops the script after."""
    st.markdown(CSS, unsafe_allow_html=True)
    ok = presenter_ok()
    store = get_store()
    if ok and str(st.query_params.get("demo", "")).isdigit() and not any(r["demo"] for r in store.snapshot(True)):
        fill_demo(store, min(int(st.query_params["demo"]), 40))
    with st.container(key="bd"):
        with st.container(key="bd_main"):
            _live()
            if ok:
                with st.container(key="bd_ctl"):
                    with st.popover("Presenter", icon=":material/tune:"):
                        who = st.text_input("Hide a shopper (their nickname)", key="bd_hide_in")
                        if st.button("Hide", key="bd_hide") and who:
                            st.toast(f"Hidden: {store.hide(who)}")
                        if st.button("Reset board", key="bd_reset", type="primary"):
                            store.reset()
                            st.toast("Board cleared")
        with st.container(key="bd_side"):
            _qr_box(join_url())
