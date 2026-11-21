"""Nudge demo: "Mercato", a small online shop with ~100 funny products.

Run from the repo root:  python -m streamlit run app/app.py

A visitor browses (each product page opened = a view click), adds to cart, and finally leaves.
Their clicks become words (nudge/contract.py), predict() gives the chance they come back and buy
on their own within 7 days, and the decision layer (nudge/decision.py) picks:
no message / reminder only / reminder + 10%.

Screenshot shortcuts (only read when a new visitor opens the page):
  ?views=p047,p014  start having viewed these products (25 s apart)
  ?cart=p024,p023   start with these products in the cart
  ?view=p024        start on that product page
  ?visits=5&last=0.5&bought=1   start as a returning shopper with this history
  ?page=left        start on the "shopper left" result page
"""

import base64
import html
import io
import math
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from nudge.contract import START_WORDS, predict, session_words  # noqa: E402
START_WORDS_SET = set(START_WORDS)
from nudge.decision import DISCOUNT, DISCOUNT_RATE, NOTHING, REMINDER, cutoffs, decide  # noqa: E402

SHOP = "Mercato"
PAGE_SIZE = 20

st.set_page_config(page_title=f"{SHOP} · demo shop", page_icon=":material/shopping_bag:", layout="wide",
                   initial_sidebar_state="collapsed")


# --- Catalogue: 16 classmate products (shown first, with a friend's photo as an easter egg) + 90 funny products ---
@st.cache_data
def load_catalog() -> pd.DataFrame:
    shop = pd.read_csv(ROOT / "shop" / "catalog.csv")
    shop["image"] = shop["id"].map(lambda i: ROOT / "shop" / "images" / f"{i}.jpg")
    custom = pd.read_csv(ROOT / "shop" / "custom" / "products.csv", dtype=str).fillna("")
    custom = custom[custom["name"].str.strip() != ""].copy()
    custom["price"] = pd.to_numeric(custom["price"], errors="coerce").fillna(0)
    custom["image"] = custom["photo"].map(lambda f: ROOT / "shop" / "custom" / f)
    if "friend_photo" not in custom:
        custom["friend_photo"] = ""
    custom["friend"] = custom["friend_photo"].map(lambda f: str(ROOT / "shop" / "custom" / f) if f and (ROOT / "shop" / "custom" / f).exists() else None)
    custom["featured"] = True
    shop["friend"], shop["featured"] = None, False
    both = pd.concat([custom.drop(columns=["photo", "friend_photo"]), shop.drop(columns=["photo_query"])], ignore_index=True)
    both = both[both["price"] > 0]
    both["image"] = both["image"].map(lambda p: str(p) if Path(p).exists() else None)
    return both.set_index("id", drop=False)


@st.cache_data
def thumb(path: str | None, size: int = 160) -> str:
    """Small square JPEG as a data URI, for images inside HTML (cart rows, message preview)."""
    if not path:
        return ""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


CATALOG = load_catalog()

# --- Look: Mercato's own shop style (not a copy of any real brand) ---------------------------------
# White page, photos first, ink-black actions, one tomato accent. Plus Jakarta Sans when online,
# the system UI font (SF Pro / Segoe UI) when offline.
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');
:root {
  --ink:#14161A; --ink-2:#2E3238; --muted:#5D636C; --faint:#8A9099; --line:#E6E5E1; --line-2:#D6D4CE;
  --paper:#FFFFFF; --soft:#F5F4F1; --soft-2:#EEEDE8; --accent:#C93A25; --accent-soft:#FBEDEA;
  --ok:#1E6B4F; --ok-soft:#E7F2EC;
  --sans:'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
  --mono:'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
}
.stApp, .stApp *:not([data-testid="stIconMaterial"]):not(.code) { font-family: var(--sans) !important; }
.stApp .bz-bubble .code { font-family: var(--mono) !important; }

[data-testid="stElementContainer"]:has(style) { display:none; }
.stApp { background: var(--paper); color: var(--ink); }
header[data-testid="stHeader"], footer, #MainMenu, [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
.stMain { overflow-x: hidden; }
.block-container { padding: 0 32px 64px 32px !important; max-width: 1360px; }
@media (min-width: 1700px) { .block-container { max-width: 1600px; } }
[data-testid="stImage"] button, [data-testid="stElementToolbar"] { display:none !important; }

/* Keyboard focus: always visible, one style */
button:focus-visible, input:focus-visible, a:focus-visible, [role="button"]:focus-visible {
  outline: 2px solid var(--ink) !important; outline-offset: 2px !important; box-shadow:none !important; }

/* --- Buttons ------------------------------------------------------------------------------- */
.stButton button { border-radius: 10px; min-height: 40px; font-weight: 600; font-size: 14px; letter-spacing: .005em;
  transition: background-color .15s ease, border-color .15s ease, color .15s ease, transform .1s ease; }
.stButton button:active { transform: translateY(1px); }
button[data-testid="stBaseButton-primary"] { background: var(--ink); border: 1px solid var(--ink); color:#fff; }
button[data-testid="stBaseButton-primary"]:hover { background: var(--ink-2); border-color: var(--ink-2); color:#fff; }
button[data-testid="stBaseButton-secondary"] { background:#fff; border:1px solid var(--line-2); color: var(--ink); }
button[data-testid="stBaseButton-secondary"]:hover { border-color: var(--ink); color: var(--ink); background:#fff; }
button[data-testid="stBaseButton-primary"]:disabled, button[data-testid="stBaseButton-secondary"]:disabled {
  background: var(--soft); border-color: var(--soft); color: var(--faint); opacity:1; cursor: default; }
div[class*="st-key-gc_"] button:disabled, div[class*="st-key-sc_"] button:disabled, .st-key-detail_cart button:disabled {
  background: var(--ok-soft); border-color: var(--ok-soft); color: var(--ok); }
div[class*="st-key-gc_"] button, div[class*="st-key-sc_"] button { min-height: 36px; padding: 0 12px; background:#fff;
  border: 1px solid var(--line-2); color: var(--ink); font-size: 14px; }
div[class*="st-key-gc_"] button:not(:disabled):hover, div[class*="st-key-sc_"] button:not(:disabled):hover {
  background: var(--ink); border-color: var(--ink); color:#fff; }
button[data-testid="stBaseButton-tertiary"] { color: var(--muted); font-weight:500; min-height: 28px; padding: 0; }
button[data-testid="stBaseButton-tertiary"]:hover { color: var(--ink); text-decoration: underline; text-underline-offset: 3px; }

/* --- Header ------------------------------------------------------------------------------- */
.bz-util { margin: 0 calc(50% - 50vw); padding: 8px calc(50vw - 50%); background: var(--ink); color:#D9DBDF;
  font-size: 13px; display:flex; justify-content:space-between; gap:24px; letter-spacing:.01em; }
.bz-util b { color:#fff; font-weight:600; }
.bz-util .sep { color:#6B717A; margin: 0 10px; }
.st-key-hdr { padding: 16px 0 0 0; gap: 8px; }
.bz-logo { display:flex; align-items:center; gap:10px; text-decoration:none; height:48px; }
.bz-logo .mark { width:30px; height:30px; border-radius:9px; background: var(--ink); position:relative; flex:none; }
.bz-logo .mark span { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; color:#fff; font-weight:800; font-size:17px; }
.bz-logo .word { font-weight:800; font-size:26px; letter-spacing:-.04em; color: var(--ink); }
.st-key-hdr [data-testid="stTextInput"] > div > div { border-radius: 999px; background: var(--soft); border: 1px solid var(--soft); min-height:48px; }
.st-key-hdr [data-testid="stTextInput"] > div > div:focus-within { background:#fff; border-color: var(--ink); }
.st-key-hdr [data-testid="stTextInput"] input { font-size: 15px; padding-left: 4px; color: var(--ink); }
.st-key-hdr [data-testid="stTextInput"] input::placeholder { color: var(--muted); opacity:1; }
.bz-hcart { display:flex; align-items:center; justify-content:flex-end; gap:12px; height:48px; color: var(--ink); }
.bz-hcart .ico { position:relative; width:28px; height:28px; }
.bz-hcart .badge { position:absolute; top:-6px; right:-9px; min-width:18px; height:18px; padding:0 5px; border-radius:9px; background: var(--ink);
  color:#fff; font-size:11px; font-weight:700; display:flex; align-items:center; justify-content:center; box-shadow:0 0 0 2px #fff; }
/* Category navigation: text tabs with an underline */
.st-key-nav { border-bottom: 1px solid var(--line); margin: 4px 0 8px 0; padding: 0; }
.st-key-nav [role="radiogroup"] { gap: 0 32px; }
.st-key-nav button[data-variant="pills"] { background: transparent !important; border: none !important; border-radius: 0 !important;
  padding: 12px 0 14px 0 !important; min-height: 0; color: var(--muted) !important; box-shadow: inset 0 -2px 0 transparent; }
.st-key-nav button[data-variant="pills"] p { font-size: 15px; font-weight: 600; }
.st-key-nav button[data-variant="pills"]:hover { color: var(--ink) !important; box-shadow: inset 0 -2px 0 var(--line-2); }
.st-key-nav button[data-variant="pills"][data-selected="true"] { color: var(--ink) !important; box-shadow: inset 0 -2px 0 var(--ink); }

/* --- Listing ------------------------------------------------------------------------------ */
.bz-listhead { display:flex; align-items:baseline; justify-content:space-between; margin: 8px 0 32px 0; }
.bz-listhead .h { font-size: 28px; font-weight: 800; letter-spacing:-.025em; margin:0; padding:0; color: var(--ink); line-height:1.2; }
.bz-listhead span { color: var(--muted); font-size: 15px; }
.bz-section { font-size: 22px; font-weight: 800; letter-spacing:-.02em; color: var(--ink); margin: 48px 0 16px 0; }

/* Product card: the photo is the hero; the whole card opens the product page */
[class*="st-key-card_"] { position: relative; gap: 8px; padding-bottom: 24px; }
[class*="st-key-card_"] [data-testid="stImage"] { border-radius: 12px; overflow:hidden; background: var(--soft); }
[class*="st-key-card_"] [data-testid="stImage"] img { border-radius: 12px; aspect-ratio: 1/1; object-fit: cover; width:100%;
  transition: transform .35s cubic-bezier(.2,.7,.2,1); }
[class*="st-key-card_"]:hover [data-testid="stImage"] img { transform: scale(1.035); }
[class*="st-key-card_"]:hover .bz-name { text-decoration: underline; text-decoration-color: var(--line-2); text-underline-offset: 3px; }
div[class*="st-key-gv_"], div[class*="st-key-sv_"] { position:absolute !important; top:0; left:0; right:0; bottom: 68px; z-index: 3; margin:0; }
div[class*="st-key-gv_"] .stButton, div[class*="st-key-sv_"] .stButton,
div[class*="st-key-gv_"] button, div[class*="st-key-sv_"] button { width:100% !important; height:100% !important; }
div[class*="st-key-gv_"] button, div[class*="st-key-sv_"] button { background: transparent !important; border: none !important;
  color: transparent !important; border-radius: 12px; cursor: pointer; }
.bz-meta { padding: 4px 0 12px 0; }
.bz-cat { color: var(--muted); font-size: 11.5px; font-weight:600; text-transform: uppercase; letter-spacing: .08em; }
.bz-name { font-weight: 600; color: var(--ink); line-height: 1.35; font-size: 15px; margin: 4px 0 8px 0; height: 2.7em;
  display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient: vertical; overflow:hidden; }
.bz-price { font-weight: 800; font-size: 18px; color: var(--ink); letter-spacing:-.01em; font-variant-numeric: tabular-nums; }
.bz-noimg { aspect-ratio:1/1; display:flex; align-items:center; justify-content:center; background: var(--soft); color: var(--muted);
  border-radius:12px; font-size:13px; }
.bz-empty { padding: 64px 0; text-align:center; color: var(--muted); }
.bz-empty b { display:block; color: var(--ink); font-size: 18px; margin-bottom: 4px; }

/* --- Product page ------------------------------------------------------------------------- */
.bz-crumb { color: var(--muted); font-size: 13px; }
.bz-crumb b { color: var(--ink); font-weight:600; }
.st-key-pdp_img [data-testid="stImage"] { background: var(--soft); border-radius: 16px; overflow:hidden; }
.st-key-pdp_img [data-testid="stImage"] img { border-radius: 16px; aspect-ratio:1/1; object-fit:cover; }
.bz-pdp-cat { color: var(--muted); font-size: 12px; font-weight:700; text-transform: uppercase; letter-spacing:.1em; margin-top: 8px; }
.bz-pdp-name { font-size: 34px; line-height: 1.15; font-weight: 800; letter-spacing: -.03em; color: var(--ink); margin: 8px 0 16px 0; }
.bz-big-price { font-size: 30px; font-weight: 800; color: var(--ink); letter-spacing:-.02em; font-variant-numeric: tabular-nums; }
.bz-vat { color: var(--muted); font-size: 13px; margin-top: 2px; }
.bz-desc { color: var(--ink-2); font-size: 17px; line-height: 1.6; margin: 24px 0 24px 0; max-width: 34em; }
.st-key-detail_cart button { min-height: 52px; font-size: 16px; padding: 0 40px; min-width: 260px; }
.bz-perks { list-style:none; padding:0; margin: 16px 0 0 0; border-top: 1px solid var(--line); }
.bz-perks li { display:flex; gap:12px; align-items:center; padding: 12px 0; border-bottom: 1px solid var(--line); color: var(--ink-2); font-size: 14px; margin:0; }
.bz-perks svg { flex:none; color: var(--ink); }

/* --- Side column: cart + what Nudge sees -------------------------------------------------- */
div[data-testid="stColumn"]:has(.st-key-cartbox) { position: sticky; top: 16px; align-self: flex-start; min-width: 280px;
  max-height: calc(100vh - 32px); overflow-y: auto; scrollbar-width: thin; }
.st-key-cartbox { border: 1px solid var(--line); border-radius: 16px; padding: 20px; gap: 12px; background:#fff; }
.bz-panel-title { font-weight: 800; color: var(--ink); font-size: 17px; letter-spacing:-.01em; display:flex; justify-content:space-between; align-items:baseline; }
.bz-panel-title span { color: var(--muted); font-size: 13px; font-weight: 500; }
.bz-citem { display:flex; gap:12px; align-items:center; padding: 4px 0; }
div[class*="st-key-rm_"] button { min-height: 32px; width: 32px; padding: 0; border-radius: 8px; justify-content:center; }
div[class*="st-key-rm_"] button:hover { background: var(--soft); text-decoration: none; }
div[class*="st-key-rm_"] button p { position:absolute; width:1px; height:1px; overflow:hidden; clip: rect(0 0 0 0); white-space: nowrap; }
div[class*="st-key-rm_"] { display:flex; justify-content:flex-end; }
.bz-citem img { width:48px; height:48px; border-radius:8px; object-fit:cover; background: var(--soft); flex:none; }
.bz-citem .t { font-size: 13.5px; line-height:1.3; color: var(--ink); font-weight:500;
  display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient: vertical; overflow:hidden; }
.bz-citem .p { font-size: 13.5px; font-weight: 700; margin-top: 2px; }
.bz-cempty { color: var(--muted); font-size: 14px; padding: 8px 0; }
.bz-total { display:flex; justify-content:space-between; align-items:baseline; border-top: 1px solid var(--line); padding-top: 12px; margin: 4px 0 8px 0; }
.bz-total span { color: var(--muted); font-size: 14px; }
.bz-total b { font-size: 20px; font-weight: 800; font-variant-numeric: tabular-nums; }
.st-key-nudgebox { background: var(--soft); border-radius: 16px; padding: 20px 20px 16px 20px; gap: 8px; }
.bz-nudge-title { display:flex; align-items:center; gap:8px; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing:.1em; color: var(--muted); }
.bz-nudge-title i { width:8px; height:8px; border-radius:50%; background: var(--ok); box-shadow: 0 0 0 3px rgba(30,107,79,.15); }
.bz-chips { display:flex; flex-wrap:wrap; gap:6px; }
.st-key-nudgebox .bz-chips { flex-direction: column; align-items: flex-start; gap: 6px; margin: 4px 0; }
.bz-chip { display:inline-block; font-size: 15px; font-weight: 500; line-height: 1.3; padding: 6px 10px; border-radius: 8px;
  background:#fff; border:1px solid var(--line); color: var(--ink); }
.bz-chip.cart { border-color: #BFDCCB; background: var(--ok-soft); color: var(--ok); font-weight: 600; }
.bz-chip.ctx { background: var(--ink); color:#fff; border-color: var(--ink); }
.bz-more { color: var(--muted); font-size: 13px; padding: 2px 0; }
.bz-nudge-note { color: var(--muted); font-size: 13px; line-height: 1.5; margin: 4px 0 8px 0; }

/* --- Live meter: the chance after every click (every part has a fixed height, so nothing jumps) --- */
.nm-label { font-size: 15px; font-weight: 600; color: var(--ink-2); margin-top: 10px; }
.nm-sub { font-size: 14px; line-height: 1.4; color: var(--muted); min-height: 2.8em; margin-top: 2px; }
.nm-hero-row { display:flex; align-items:baseline; gap: 10px; height: 62px; }
.nm-hero { font-size: 58px; font-weight: 800; line-height: 1; letter-spacing: -.05em; color: var(--ink); font-variant-numeric: tabular-nums; }
.nm-hero .pc { font-size: .5em; margin-left: 2px; letter-spacing: 0; color: var(--muted); }
.nm-delta { font-size: 15px; font-weight: 600; color: var(--muted); font-variant-numeric: tabular-nums; white-space: nowrap;
  min-width: 0; animation: nm-fade .4s ease-out both; }
.nm-chart { --y: 42px; --lab: 128px; display:grid; grid-template-columns: var(--y) minmax(0,1fr); grid-template-rows: var(--h) 24px; margin-top: 12px; }
.nm-chart.big { --y: 46px; --lab: 162px; }
.nm-y { position: relative; }
.nm-y span { position:absolute; right: 7px; transform: translateY(50%); font-size: 14px; line-height: 1; color: var(--muted); font-variant-numeric: tabular-nums; }
.nm-y span.cut { font-weight: 700; color: var(--ink-2); }
.nm-plot { position: relative; }
.nm-zones { position:absolute; inset:0; border-radius: 8px; overflow:hidden; background: var(--paper); box-shadow: inset 0 0 0 1px var(--line); }
.nm-zone { position:absolute; left:0; right:0; transition: background-color .2s ease; }
.nm-zone.z0 { background: color-mix(in srgb, var(--faint) 12%, var(--paper)); }
.nm-zone.z1 { background: color-mix(in srgb, var(--accent) 8%, var(--paper)); }
.nm-zone.z2 { background: color-mix(in srgb, var(--ink-2) 3%, var(--paper)); }
.nm-zone.z0.on { background: color-mix(in srgb, var(--faint) 24%, var(--paper)); }
.nm-zone.z1.on { background: color-mix(in srgb, var(--accent) 16%, var(--paper)); }
.nm-zone.z2.on { background: color-mix(in srgb, var(--ink-2) 10%, var(--paper)); }
.nm-cut { position:absolute; left:0; right:0; height:0; border-top: 1.5px dashed color-mix(in srgb, var(--ink-2) 45%, transparent); }
.nm-zl { position:absolute; right: 10px; transform: translateY(-50%); font-size: 14px; line-height: 1.2; font-weight: 600; color: var(--muted); white-space: nowrap; }
.nm-zl.on { font-weight: 800; }
.nm-zl.z0.on { color: var(--ink-2); } .nm-zl.z1.on { color: var(--accent); } .nm-zl.z2.on { color: var(--ink); }
.nm-line { position:absolute; top:0; bottom:0; left:0; right: var(--lab); }
.nm-svg { position:absolute; inset:0; width:100%; height:100%; overflow: visible; color: var(--ink); }
.nm-svg path { fill:none; stroke: currentColor; stroke-width: 2.5; stroke-linejoin: round; stroke-linecap: round; vector-effect: non-scaling-stroke; }
.nm-new { animation: nm-draw .45s cubic-bezier(.2,.7,.2,1) both; }
.nm-dot { position:absolute; width:7px; height:7px; margin: 0 0 -3.5px -3.5px; border-radius:50%; background: var(--ink); }
.nm-dot.now { width:14px; height:14px; margin: 0 0 -7px -7px; box-shadow: 0 0 0 4px var(--paper); animation: nm-pop .35s .25s cubic-bezier(.2,.7,.2,1.3) both; }
.nm-val { position:absolute; transform: translateX(-50%); margin-bottom: 13px; font-size: 14px; font-weight: 800; line-height: 1.2;
  padding: 1px 6px; border-radius: 6px; background: var(--paper); box-shadow: 0 0 0 1px var(--line); font-variant-numeric: tabular-nums;
  white-space: nowrap; animation: nm-pop .35s .25s cubic-bezier(.2,.7,.2,1.3) both; }
.nm-val.below { margin-bottom: -36px; }
.nm-val.z0 { color: var(--ink-2); } .nm-val.z1 { color: var(--accent); } .nm-val.z2 { color: var(--ink); }
.nm-chart.big .nm-val { font-size: 16px; margin-bottom: 16px; }
.nm-chart.big .nm-val.below { margin-bottom: -42px; }
.nm-dot.now.z0 { background: var(--muted); } .nm-dot.now.z1 { background: var(--accent); } .nm-dot.now.z2 { background: var(--ink); }
.nm-x { grid-column: 2; display:flex; justify-content:space-between; align-items:flex-end; font-size: 14px; color: var(--muted); padding: 0 var(--lab) 0 0; gap: 12px; white-space: nowrap; }
.nm-chart.big .nm-x { padding-right: calc(var(--lab) - 110px); }
.nm-chart.big .nm-zl { font-size: 15px; }
.nm-chart.big .nm-y span, .nm-chart.big .nm-x { font-size: 15px; }
.nm-chart.big .nm-dot { width:9px; height:9px; margin: 0 0 -4.5px -4.5px; }
.nm-chart.big .nm-dot.now { width:18px; height:18px; margin: 0 0 -9px -9px; }
.nm-chart.big .nm-svg path { stroke-width: 3; }
.nm-act { margin-top: 12px; }
.nm-act small { display:block; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); }
.nm-act b { display:flex; align-items:center; gap: 9px; font-size: 19px; font-weight: 800; letter-spacing: -.01em; line-height: 1.3; white-space: nowrap; }
.nm-act i { width: 10px; height: 10px; border-radius: 3px; flex: none; }
.nm-act.z0 b { color: var(--ink-2); } .nm-act.z0 i { background: var(--muted); }
.nm-act.z1 b { color: var(--accent); } .nm-act.z1 i { background: var(--accent); }
.nm-act.z2 b { color: var(--ink); } .nm-act.z2 i { background: var(--ink); }
.nm-why { font-size: 13.5px; line-height: 1.4; color: var(--muted); min-height: 2.8em; margin-top: 2px; }
.nm-words { margin-top: 12px; padding: 12px 0 10px 0; border-top: 1px solid var(--line-2); }
.nm-words-head { display:flex; justify-content:space-between; align-items:baseline; margin-bottom: 8px; }
.nm-words-head b { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); }
.nm-words-head span { font-size: 12px; color: var(--muted); }
.st-key-nudgebox .nm-words .bz-chips { min-height: 150px; flex-wrap: nowrap; margin: 0; gap: 6px; }
.st-key-nudgebox .bz-chip { font-size: 13px; padding: 4px 9px; max-width: 100%; white-space: normal; overflow-wrap: break-word; text-wrap: balance; }
@keyframes nm-draw { from { clip-path: var(--clip-from); } to { clip-path: inset(0 0 0 0); } }
@keyframes nm-pop { from { transform: scale(.4); opacity: 0; } to { transform: scale(1); opacity: 1; } }
@keyframes nm-fade { from { opacity: 0; transform: translateY(3px); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) { .nm-new, .nm-dot.now, .nm-val, .nm-delta { animation: none; } .nm-zone { transition: none; } }

/* --- Result page ("the shopper left") ----------------------------------------------------- */
.bz-eyebrow { font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing:.12em; color: var(--muted); margin-top: 16px; }
.bz-left-title { font-size: 34px; font-weight: 800; letter-spacing: -.03em; color: var(--ink); margin: 6px 0 0 0; line-height:1.15; }
.bz-result { display:grid; grid-template-columns: 7fr 5fr; gap: 32px; align-items: start; }
.bz-card { border: 1px solid var(--line); border-radius: 20px; padding: 28px 32px; }
.bz-hero-label { color: var(--muted); font-size: 16px; line-height:1.5; }
.bz-hero-label b { color: var(--ink); }
.bz-hero { font-size: 104px; font-weight: 800; color: var(--ink); line-height: .95; letter-spacing: -.06em; font-variant-numeric: tabular-nums;
  margin: 4px 0 8px -4px; }
.bz-hero sup { font-size: .42em; vertical-align: top; position: relative; top: .18em; margin-left: 4px; letter-spacing:0; color: var(--muted); }
.bz-top { display:grid; grid-template-columns: minmax(0,1fr) minmax(0,1.15fr); gap: 28px; align-items: start; }
.bz-top .bz-decision { margin: 0; padding: 0 0 0 28px; border-top: none; border-left: 1px solid var(--line); min-height: 100%; }
.bz-story { margin-top: 24px; padding-top: 18px; border-top: 1px solid var(--line); }
.bz-story small { display:block; font-size: 13px; font-weight:700; text-transform: uppercase; letter-spacing:.1em; color: var(--muted); margin-bottom: 12px; }
.bz-decision { margin-top: 20px; padding-top: 16px; border-top: 1px solid var(--line); }
.bz-decision small, .bz-seen small { display:block; font-size: 13px; font-weight:700; text-transform: uppercase; letter-spacing:.1em; color: var(--muted); margin-bottom: 6px; }
.bz-decision .d { font-size: 28px; font-weight: 800; letter-spacing: -.02em; color: var(--ink); line-height:1.25; }
.bz-decision .why { color: var(--muted); font-size: 16px; margin-top: 4px; }
.bz-money { margin-top: 14px; display:inline-flex; gap:8px; align-items:center; padding: 8px 12px; border-radius: 8px; font-size: 15px; font-weight:600; }
.bz-money.keep { background: var(--ok-soft); color: var(--ok); }
.bz-money.give { background: var(--accent-soft); color: #9C2C1B; }
.bz-seen { margin-top: 16px; padding-top: 14px; border-top: 1px solid var(--line); }
.bz-seen .bz-chip { font-size: 13px; padding: 4px 8px; color: var(--muted); }
.bz-seen .bz-chip.ctx { color:#fff; }
.bz-seen .bz-chip.cart { color: var(--ok); }
/* Message preview: a generic chat app on a phone (not a copy of any real app's branding) */
.bz-phonecol { display:flex; flex-direction:column; gap: 12px; }
.bz-phone { flex: none; border-radius: 28px; border: 1px solid var(--line-2); background: #fff; overflow:hidden; display:flex; flex-direction:column;
  box-shadow: 0 1px 2px rgba(20,22,26,.04), 0 24px 48px -24px rgba(20,22,26,.18); }
.bz-phone-top { display:flex; align-items:center; gap:12px; padding: 14px 20px; border-bottom: 1px solid var(--line); background:#fff; }
.bz-ava { width:36px; height:36px; border-radius:50%; background: var(--ink); color:#fff; font-weight:800; display:flex; align-items:center; justify-content:center; font-size:16px; }
.bz-phone-top b { font-size: 15px; display:block; color: var(--ink); }
.bz-phone-top small { font-size: 13px; color: var(--muted); }
.bz-chat { flex: none; min-height: 240px; background: #ECE7DF; padding: 20px 16px 24px 16px; display:flex; flex-direction:column; gap:8px; }
.bz-day { align-self:center; background: rgba(255,255,255,.75); color: var(--muted); font-size: 12px; padding: 4px 10px; border-radius: 8px; margin-bottom: 4px; }
.bz-bubble { background:#fff; color:#111; padding: 8px 8px 6px 8px; border-radius: 4px 14px 14px 14px; max-width: 88%; font-size: 15px; line-height: 1.45;
  box-shadow: 0 1px 1px rgba(0,0,0,.08); }
.bz-bubble img { width: 100%; aspect-ratio: 4/3; object-fit: cover; object-position: center; border-radius: 10px; display:block; margin-bottom: 8px; }
.bz-bubble p { margin: 0 4px; }
.bz-bubble .code { font-family: var(--mono); background: var(--accent-soft); color:#9C2C1B; padding: 1px 6px; border-radius: 4px; font-weight: 700; }
.bz-bubble small { display:block; text-align:right; color:#6B7178; font-size:12px; margin: 4px 4px 0 0; }
.bz-bubble.ghost { background: transparent; box-shadow:none; border: 1.5px dashed #B4AEA4; color: #8F887D; padding: 10px 12px; }
.bz-bubble.ghost p { text-decoration: line-through; text-decoration-color: #A39C91; }
.bz-quiet { margin: auto 0; display:flex; flex-direction:column; gap: 16px; align-items:center; }
.bz-silent { text-align:center; color: var(--muted); font-size: 15px; line-height:1.5; }
.bz-silent b { display:block; color: var(--ink); font-size: 17px; margin-bottom: 2px; }
.bz-cap { text-align:center; color: var(--muted); font-size: 13px; }
.st-key-lefthead { margin-bottom: 8px; }
.bz-bubble.msg { max-width: 330px; }

/* --- Thank-you page ----------------------------------------------------------------------- */
.bz-thanks { max-width: 560px; margin: 64px auto 24px auto; text-align:center; }
.bz-thanks .tick { width:56px; height:56px; border-radius:50%; background: var(--ok-soft); color: var(--ok); display:flex; align-items:center; justify-content:center; margin: 0 auto 24px auto; }
.bz-thanks h1 { font-size: 34px; font-weight: 800; letter-spacing: -.03em; margin: 0 0 8px 0; padding:0; }
.bz-thanks p { color: var(--muted); font-size: 16px; line-height: 1.6; }

/* Narrow laptops: keep everything inside the screen */
@media (max-width: 1360px) {
  .block-container { padding: 0 24px 48px 24px !important; }
  .bz-name { font-size: 14px; }
  .bz-hero { font-size: 96px; }
  .bz-card { padding: 24px 28px; }
}
/* Phones: the live meter and cart come before the products (columns stack), and nothing is sticky */
@media (max-width: 640px) {
  div[data-testid="stColumn"]:has(.st-key-cartbox) { order: -1; position: static; max-height: none; overflow: visible; min-width: 0; }
  .st-key-nudgebox { margin-top: 8px !important; }
  .bz-result { grid-template-columns: 1fr; gap: 16px; }
  .bz-top { grid-template-columns: 1fr; gap: 16px; }
  .bz-top .bz-decision { padding: 16px 0 0 0; border-left: none; border-top: 1px solid var(--line); }
  .bz-card { padding: 20px; }
  .bz-hero { font-size: 72px; }
  .bz-left-title { font-size: 28px; }
  .nm-chart.big { --lab: 132px; }
  .nm-chart.big .nm-zl { font-size: 14px; }
}
.bz-wild { margin-top: 24px; border-top: 1px solid var(--line, #e7e5e1); padding-top: 16px; }
.bz-wild-title { font-weight: 600; font-size: 15px; margin-bottom: 10px; }
.bz-wild img { width: 100%; max-height: 420px; object-fit: cover; border-radius: 12px; display: block; }
.bz-wild-cap { font-size: 13px; color: #6b6b6b; margin-top: 6px; }
</style>
""",
    unsafe_allow_html=True,
)

ICON = {  # one line-icon style (24px grid, 1.75 stroke), inline so it works offline
    "bag": '<path d="M6 7h12l1 13H5L6 7Z"/><path d="M9 7V6a3 3 0 0 1 6 0v1"/>',
    "truck": '<path d="M3 6h11v10H3z"/><path d="M14 10h4l3 3v3h-7"/><circle cx="7" cy="17.5" r="1.5"/><circle cx="17" cy="17.5" r="1.5"/>',
    "return": '<path d="M9 14 4 9l5-5"/><path d="M4 9h11a5 5 0 0 1 0 10h-3"/>',
    "check": '<path d="m5 12.5 4.5 4.5L19 7.5"/>',
    "clock": '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
}


def icon(name: str, size: int = 20) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICON[name]}</svg>')


def esc(text) -> str:
    return html.escape(str(text))


def euro(x: float) -> str:
    return f"€{x:,.0f}"


WORD_PARTS = {"view": "viewed", "cart": "added to cart", "start": "first click", "quick": "quick",
              "normal": "normal pace", "long": "long pause", "away": "came back later",
              "new": "new visitor", "returning": "returning visitor"}
WORD_FULL = {"bought_recently": "bought in the last 14 days", "visits_1": "1 visit in 14 days",
             "visits_2_4": "2–4 visits in 14 days", "visits_5plus": "5+ visits in 14 days",
             "last_today": "last visit today", "last_1_3d": "last visit 1–3 days ago",
             "last_4_14d": "last visit 4–14 days ago"}


def word_label(word: str) -> str:
    """Model word -> plain words for the audience: "view_expensive_again_quick" -> "viewed · expensive · again · quick"."""
    if word in WORD_FULL:
        return WORD_FULL[word]
    return " · ".join(WORD_PARTS.get(part, part) for part in word.split("_"))


def chips(words: list[str], last: int | None = None, count_earlier: bool = True) -> str:
    """Words as chips (raw model word in the tooltip). With `last`, only the latest ones plus a count of earlier ones."""
    shown = list(enumerate(words))
    more = ""
    if last and len(shown) > last:
        more = f'<span class="bz-more">+ {len(shown) - last} earlier word{"" if len(shown) - last == 1 else "s"}</span>' if count_earlier else ""
        shown = shown[-last:]
    return more + "".join(
        f'<span class="bz-chip{" cart" if w.startswith("cart") else ""}{" ctx" if i == 0 or w in WORD_FULL else ""}" '
        f'title="{esc(w)}">'
        f'{esc(word_label(w))}</span>' for i, w in shown)


# --- Live chance: the model's score after every click, and the line chart of it ---------------------
ZONE_OF = {NOTHING: 0, DISCOUNT: 1, REMINDER: 2}
ZONE_NAME = ["No message", f"Reminder + {DISCOUNT_RATE:.0%}", "Reminder only"]
ACTION_LINE = ["No message", f"Reminder + {DISCOUNT_RATE:.0%} off", "Reminder, no discount"]
BAND_WORDS = {"cheap": "a cheap", "mid": "a mid-priced", "expensive": "an expensive"}
PACE_WORDS = {"start": "first click", "quick": "quick pace", "normal": "normal pace", "long": "after a long pause",
              "away": "after a 30+ min break"}


def why_line(words: list[str]) -> str:
    """Plain facts from the last words the model read (no claim about which word caused the score)."""
    hist = [WORD_FULL[w] for w in words if w in WORD_FULL and not w.startswith("last_")]
    who = ("Returning" + (", " + ", ".join(h.replace(" in 14 days", "").replace(" in the last 14 days", " recently")
                                          for h in hist) if hist else "")) if words[0] == "returning" else "New visitor"
    last = words[-1]
    if last in START_WORDS_SET or last in WORD_FULL:
        return f"{who}, nothing clicked yet. Browse or add something to cart to see it move."
    parts = last.split("_")
    item = f"{BAND_WORDS[parts[1]]} item"
    did = f"added {item} to cart" if parts[0] == "cart" else f"viewed {item}" + (" again" if "again" in parts else "")
    return f"Just {did}, {PACE_WORDS[parts[-1]]}." + (f" {who}." if words[0] == "returning" else "")



@st.cache_data(max_entries=20_000, show_spinner=False)
def chance(words: tuple[str, ...]) -> float:
    """predict() for one list of words, cached: the chart re-scores every earlier click on each rerun."""
    return predict(list(words))


def pct(p: float) -> int:
    return round(p * 100)


def chance_chart(hist: list[float], low: float, high: float, height: int = 168, big: bool = False,
                 end_label: str = "") -> str:
    """Line chart of the chance after each click (x = start, click 1..n) on the three decision zones, with the two
    cut-offs as dashed lines. Zones span the full width; the line stays left of the zone names (--lab).
    Shapes are HTML/SVG in % of the plot, colours come from the CSS tokens. Only the newest segment animates."""
    n = len(hist) - 1
    top = max(hist)
    ymax = 0.5 if top <= 0.5 else min(1.0, math.ceil(top * 10) / 10)
    dom = max(n, 6)                     # a short visit doesn't stretch across the whole width

    def x(i: int) -> float:            # % of the line area
        return 3 + 94 * i / dom

    def y(p: float) -> float:          # % from the bottom
        return min(p, ymax) / ymax * 100

    on = ZONE_OF[decide(hist[-1], generosity)]
    lo, hi = y(low), y(high)
    zones = (f'<div class="nm-zone z2{" on" if on == 2 else ""}" style="bottom:{hi:.2f}%;top:0"></div>'
             f'<div class="nm-zone z1{" on" if on == 1 else ""}" style="bottom:{lo:.2f}%;height:{hi - lo:.2f}%"></div>'
             f'<div class="nm-zone z0{" on" if on == 0 else ""}" style="bottom:0;height:{lo:.2f}%"></div>')
    cuts = f'<i class="nm-cut" style="bottom:{lo:.2f}%"></i><i class="nm-cut" style="bottom:{hi:.2f}%"></i>'

    # y axis: 0 / half / top, plus the two cut-off values; a tick too close to a cut-off gives way
    near = 21 / height * 100
    ticks = [(y(v), f"{pct(v)}%" if v else "0", "") for v in (0, ymax / 2, ymax)
             if abs(y(v) - lo) > near and abs(y(v) - hi) > near]
    ticks += [(lo, f"{pct(low)}%", " cut"), (hi, f"{pct(high)}%", " cut")]
    yaxis = "".join(f'<span class="{c.strip()}" style="bottom:{b:.2f}%">{t}</span>' for b, t, c in ticks)

    pts = [(x(i), 100 - y(p)) for i, p in enumerate(hist)]
    line = ""
    if n >= 1:
        old = " L".join(f"{a:.2f},{b:.2f}" for a, b in pts[:-1])
        (a0, b0), (a1, b1) = pts[-2], pts[-1]
        line = (f'<svg class="nm-svg" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">'
                + (f'<path d="M{old}" />' if n >= 2 else "") + '</svg>'
                f'<svg class="nm-svg nm-new" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true" '
                f'style="--clip-from: inset(0 {100 - a0:.2f}% 0 0)"><path d="M{a0:.2f},{b0:.2f} L{a1:.2f},{b1:.2f}" /></svg>')
    dots = "".join(f'<i class="nm-dot" style="left:{x(i):.2f}%;bottom:{y(p):.2f}%"></i>' for i, p in enumerate(hist[:-1]))
    dots += f'<i class="nm-dot now z{on}" style="left:{x(n):.2f}%;bottom:{y(hist[-1]):.2f}%"></i>'
    vy = y(hist[-1])   # value tag above the dot (below it near the top), on its own background so no line crosses it
    dots += (f'<span class="nm-val z{on}{" below" if vy > 75 else ""}" style="left:{x(n):.2f}%;bottom:{vy:.2f}%">'
             f'{pct(hist[-1])}%</span>')

    # zone names at the right edge, centred on their zone but never on top of each other
    gap, edge = (20, 10) if big else (17, 9)
    lo_px, hi_px = low / ymax * height, min(high, ymax) / ymax * height
    c0 = min(height - lo_px / 2, height - edge)
    c1 = min(height - (lo_px + hi_px) / 2, c0 - gap)
    c2 = max(min((height - hi_px) / 2, c1 - gap), edge)
    names = "".join(f'<span class="nm-zl z{i}{" on" if i == on else ""}" style="top:{c:.1f}px">{ZONE_NAME[i]}</span>'
                    for i, c in ((2, c2), (1, c1), (0, c0)))
    label = (f"Chance to come back and buy after each click: " + ", ".join(f"{pct(p)}%" for p in hist)
             + f". Zones: under {pct(low)}% no message, {pct(low)}–{pct(high)}% reminder + {DISCOUNT_RATE:.0%}, "
             f"over {pct(high)}% reminder only.")
    right = end_label or (f"click {n}" if n else "")
    return (f'<div class="nm-chart{" big" if big else ""}" role="img" aria-label="{esc(label)}" style="--h:{height}px">'
            f'<div class="nm-y">{yaxis}</div>'
            f'<div class="nm-plot"><div class="nm-zones">{zones}{cuts}</div>{names}'
            f'<div class="nm-line">{line}{dots}</div></div>'
            f'<div class="nm-x"><span>start</span><span>{esc(right)}</span></div></div>')


# --- State -----------------------------------------------------------------------------------------
if "clicks" not in st.session_state:
    st.session_state.update(clicks=[], cart=[], page="shop", product=None, limit=PAGE_SIZE, left=False,
                            before=0, removed=False)
    _start = True
else:
    _start = False


def log(event: str, pid: str) -> None:
    """Record a click the model understands (view or cart). A cart without a view first also records the
    view, because real shoppers almost always open a product before adding it (99.6% in the data)."""
    now = time.time()
    st.session_state.update(before=len(st.session_state.clicks), removed=False)  # for "▲ +3 pts" after this action
    price = float(CATALOG.loc[pid, "price"])
    if event == "cart" and all(c["product_id"] != pid for c in st.session_state.clicks):
        st.session_state.clicks.append({"event": "view", "product_id": pid, "price": price, "time": now})
        now += 1
    st.session_state.clicks.append({"event": event, "product_id": pid, "price": price, "time": now})


def open_product(pid: str) -> None:
    log("view", pid)
    st.session_state.update(page="product", product=pid)


def add_to_cart(pid: str) -> None:
    log("cart", pid)
    st.session_state.cart.append(pid)


def remove_from_cart(pid: str) -> None:
    # Not sent to the model: the training data has no remove clicks.
    st.session_state.cart.remove(pid)
    st.session_state.update(before=len(st.session_state.clicks), removed=True)


def go_shop() -> None:
    st.session_state.update(page="shop", product=None)


def leave() -> None:
    st.session_state.update(page="left", left=True)


def checkout() -> None:
    st.session_state.update(page="bought")


def new_visitor() -> None:
    st.session_state.update(clicks=[], cart=[], page="shop", product=None, limit=PAGE_SIZE, left=False,
                            before=0, removed=False)


VISITS = {"1": 1, "2–4": 3, "5+": 5}                                        # label -> visits_14d
LAST_VISIT = {"Today": 0.5, "1–3 days ago": 2.0, "4–14 days ago": 7.0}      # label -> days_since_last

if _start:  # screenshot shortcuts (see the module docstring); a normal visit has no query parameters
    qp = st.query_params
    views = [p for p in qp.get("views", "").split(",") if p in CATALOG.index]
    for k, pid in enumerate(views):
        log("view", pid)
        st.session_state.clicks[-1]["time"] -= 25 * (len(views) - k)
    for pid in [p for p in qp.get("cart", "").split(",") if p in CATALOG.index]:
        add_to_cart(pid)
    if qp.get("view") in CATALOG.index:
        open_product(qp["view"])
    if qp.get("page") == "left":
        leave()
    v = {"1": "1", "3": "2–4", "5": "5+"}.get(qp.get("visits", ""))
    last = {"0.5": "Today", "2": "1–3 days ago", "7": "4–14 days ago"}.get(qp.get("last", ""), "1–3 days ago")
    st.session_state.update(h_returning=bool(v), h_visits=v or "1", h_last=last, h_bought=qp.get("bought") == "1")

# --- Presenter controls (sidebar, hidden until opened) ------------------------------------------------
with st.sidebar:
    st.markdown("### Presenter controls")
    generosity = st.slider("Shop owner: how generous with discounts?", 0, 100, 50, step=5) / 100
    low, high = cutoffs(generosity)
    st.caption(f"Under {low:.0%}: no message · {low:.0%}–{high:.0%}: reminder + {DISCOUNT_RATE:.0%} · over {high:.0%}: reminder only")
    returning = st.toggle("This customer has visited in the last 14 days", key="h_returning")
    visits_14d, days_since_last, bought_14d = 0, None, False
    if returning:   # the model's strongest signals: the % reacts as soon as these change
        for k, v in (("h_visits", "1"), ("h_last", "1–3 days ago"), ("h_bought", False)):
            st.session_state.setdefault(k, v)   # Streamlit forgets hidden widgets' values; start them again
        visits_14d = VISITS[st.segmented_control("Visits in the last 14 days", list(VISITS), key="h_visits") or "1"]
        days_since_last = LAST_VISIT[st.segmented_control("Last visit", list(LAST_VISIT), key="h_last") or "1–3 days ago"]
        bought_14d = st.toggle("Bought something in the last 14 days", key="h_bought")
    show_words = st.toggle("Show the words Nudge sees", value=True)
    st.button("New visitor", on_click=new_visitor, width="stretch")


def words_for(clicks: list[dict]) -> list[str]:
    """The model input for these clicks plus the presenter's shopper history (same for panel and leave page)."""
    return session_words(clicks, returning, visits_14d, bought_14d, days_since_last)


def chance_history(clicks: list[dict]) -> list[float]:
    """Chance after 0, 1, ..., n clicks: what Nudge would decide if the shopper left at that moment."""
    return [chance(tuple(words_for(clicks[:k]))) for k in range(len(clicks) + 1)]


def product_image(row) -> None:
    if row["image"]:
        st.image(row["image"], width="stretch")
    else:
        st.markdown('<div class="bz-noimg">Photo coming soon</div>', unsafe_allow_html=True)


def product_card(row, key_prefix: str) -> None:
    """Photo, category, name (2 lines max), price, Add to cart. The invisible "View" button covers the
    photo and text, so clicking anywhere on the card opens the product page (like any big shop)."""
    pid = row["id"]
    with st.container(key=f"card_{key_prefix}_{pid}", gap=None):
        st.button(f"View {row['name']}", key=f"{key_prefix}v_{pid}", on_click=open_product, args=(pid,))
        product_image(row)
        st.markdown(f'<div class="bz-meta"><div class="bz-cat">{esc(row["category"])}</div>'
                    f'<div class="bz-name" title="{esc(row["name"])}">{esc(row["name"])}</div>'
                    f'<div class="bz-price">{euro(row["price"])}</div></div>', unsafe_allow_html=True)
        in_cart = pid in st.session_state.cart
        st.button("In your cart" if in_cart else "Add to cart", key=f"{key_prefix}c_{pid}", on_click=add_to_cart,
                  args=(pid,), width="stretch", type="secondary", disabled=in_cart,
                  icon=":material/check:" if in_cart else None)


def product_grid(rows: pd.DataFrame, key_prefix: str, per_row: int = 4) -> None:
    for start in range(0, len(rows), per_row):
        cols = st.columns(per_row, gap="medium")
        for col, (_, row) in zip(cols, rows.iloc[start:start + per_row].iterrows()):
            with col:
                product_card(row, key_prefix)


cart_value = float(sum(CATALOG.loc[p, "price"] for p in st.session_state.cart))
page = st.session_state.page

# --- Header ------------------------------------------------------------------------------------------
st.markdown(f'<div class="bz-util"><div><b>Free delivery</b> over €50<span class="sep">|</span>30-day returns'
            f'<span class="sep">|</span>Ships in 1–2 days</div><div><b>Demo shop for the Nudge project</b></div></div>',
            unsafe_allow_html=True)
with st.container(key="hdr"):
    h1, h2, h3 = st.columns([1.15, 3, 1.15], vertical_alignment="center", gap="large")
    h1.markdown(f'<div class="bz-logo" aria-label="{SHOP}"><div class="mark"><span>m</span></div>'
                f'<div class="word">{SHOP.lower()}</div></div>', unsafe_allow_html=True)
    query = h2.text_input("Search", placeholder=f"Search {len(CATALOG)} products", label_visibility="collapsed",
                          key="q", icon=":material/search:", on_change=go_shop)
    n = len(st.session_state.cart) if page in ("shop", "product") else 0  # cleared once the visit is over
    badge = f'<span class="badge">{n}</span>' if n else ""
    h3.markdown(f'<div class="bz-hcart" title="Cart: {n} item{"" if n == 1 else "s"}"><div class="ico">{icon("bag", 28)}{badge}</div></div>',
                unsafe_allow_html=True)
    cats = ["All"] + sorted(CATALOG["category"].unique().tolist())
    if page != "shop":  # underline the product's category on a product page, nothing elsewhere
        k = cats.index(CATALOG.loc[st.session_state.product, "category"]) + 1 if page == "product" else 0
        st.markdown('<style>.st-key-nav button[data-variant="pills"][data-selected="true"] { box-shadow: none !important; color: var(--muted) !important; }'
                    + (f'.st-key-nav [role="radiogroup"] > button:nth-of-type({k}) {{ box-shadow: inset 0 -2px 0 var(--ink) !important; color: var(--ink) !important; }}' if k else "")
                    + '</style>', unsafe_allow_html=True)
    with st.container(key="nav"):
        cat = st.pills("Category", cats, default="All", label_visibility="collapsed", key="cat", on_change=go_shop) or "All"

# --- Result page: full width, the number is the hero ---------------------------------------------------
if page == "left":
    words = words_for(st.session_state.clicks)
    hist = chance_history(st.session_state.clicks)
    p = hist[-1]
    action = decide(p, generosity)
    with st.container(key="lefthead"):
        t1, t2 = st.columns([7, 1.6], vertical_alignment="bottom", gap="medium")
        t1.markdown('<div class="bz-eyebrow">The shopper left without buying</div>'
                    '<div class="bz-left-title">What does Nudge do now?</div>', unsafe_allow_html=True)
        t2.button("New visitor", on_click=new_visitor, key="left_new", width="stretch", icon=":material/refresh:")
    n_clicks = len(st.session_state.clicks)
    story = chance_chart(hist, low, high, height=180, big=True,
                         end_label=f"left after {n_clicks} click{'' if n_clicks == 1 else 's'}" if n_clicks else "left")
    verdict, why = {
        NOTHING: ("No message", "Looks like just browsing. A message would only annoy them."),
        REMINDER: ("Reminder only, no discount", "A good chance they come back and buy on their own, so the 10% stays in the shop."),
        DISCOUNT: (f"Reminder + {DISCOUNT_RATE:.0%} discount", "They're undecided, so a discount has the most room to matter. An A/B test would confirm it."),
    }[action]
    money = ""
    if action == REMINDER and cart_value:
        money = f'<div class="bz-money keep">Margin kept: {euro(DISCOUNT_RATE * cart_value)} a "discount everyone" flow would give away</div>'
    if action == DISCOUNT and cart_value:
        money = f'<div class="bz-money give">Discount offered: {euro(DISCOUNT_RATE * cart_value)} on a {euro(cart_value)} cart</div>'
    first = CATALOG.loc[st.session_state.cart[0]] if st.session_state.cart else None
    if first is not None:
        text = f"Hi! You left your <b>{esc(first['name'])}</b> at {SHOP}. It's still waiting for you 🙂"
    else:
        text = f"Hi! Thanks for visiting {SHOP}. The things you looked at are still waiting for you 🙂"
    if action == NOTHING:
        chat = (f'<div class="bz-quiet"><div class="bz-bubble ghost"><p>{text}</p></div>'
                f'<div class="bz-silent"><b>No message sent</b>Nudge stays quiet for this shopper.</div></div>')
    else:
        if action == DISCOUNT:
            text += f' Here\'s {DISCOUNT_RATE:.0%} off if you order today: <span class="code">NUDGE10</span>'
        img = thumb(first["image"], 480) if first is not None else ""
        pic = f'<img src="{img}" alt="">' if img else ""
        chat = f'<div class="bz-day">Today</div><div class="bz-bubble msg">{pic}<p>{text}</p><small>{time.strftime("%H:%M")}</small></div>'
    st.markdown(
        f'<div class="bz-result"><div class="bz-card">'
        f'<div class="bz-top"><div><div class="bz-hero-label">Chance they come back and buy on their own within 7 days '
        f'<b>without any message</b></div><div class="bz-hero">{pct(p)}<sup>%</sup></div></div>'
        f'<div class="bz-decision"><small>So Nudge sends</small><div class="d">{verdict}</div><div class="why">{why}</div>{money}</div></div>'
        f'<div class="bz-story"><small>The chance after each click</small>{story}</div>'
        f'<div class="bz-seen"><small>What Nudge saw</small><div class="bz-chips">{chips(words)}</div></div>'
        f'</div><div class="bz-phonecol"><div class="bz-phone"><div class="bz-phone-top"><div class="bz-ava">m</div><div><b>{SHOP}</b>'
        f'<small>Message preview · WhatsApp-style</small></div></div><div class="bz-chat">{chat}</div></div>'
        f'</div></div>', unsafe_allow_html=True)
    st.stop()

main, side = st.columns([2.6, 1], gap="large")

# --- Cart + what Nudge sees (right column) -----------------------------------------------------------
with side:
    # line the first card up with the first row of photos (measured: below the page title / the back link)
    st.markdown(f'<style>.st-key-nudgebox {{ margin-top: {74 if page == "shop" else 44}px; }}</style>', unsafe_allow_html=True)
    with st.container(key="nudgebox"):
        clicks = st.session_state.clicks
        words = words_for(clicks)
        hist = chance_history(clicks)
        now = hist[-1]
        on = ZONE_OF[decide(now, generosity)]
        if not clicks:
            delta = f'<span class="nm-delta">start, {"returning" if returning else "new"} visitor</span>'
        elif st.session_state.removed:
            delta = '<span class="nm-delta">same: removing isn\'t a click the model knows</span>'
        else:
            d = pct(now) - pct(hist[min(st.session_state.before, len(hist) - 1)])
            delta = ('<span class="nm-delta">about the same</span>' if abs(d) <= 1 else
                     f'<span class="nm-delta">{"▲ +" if d > 0 else "▼ −"}{abs(d)} pts</span>')
        count = f"{len(words)} word{'' if len(words) == 1 else 's'}"
        seen = (f'<div class="nm-words"><div class="nm-words-head" title="One word per click: what they did · price · '
                f'seen before · pace"><b>What Nudge sees</b><span>{count}</span></div>'
                f'<div class="bz-chips">{chips(words, last=4)}</div></div>') if show_words else ""
        st.markdown(
            f'<div class="bz-nudge-title"><i></i>Nudge · live</div>'
            f'<div class="nm-label">If they left now</div>'
            f'<div class="nm-hero-row"><span class="nm-hero">{pct(now)}<span class="pc">%</span></span>{delta}</div>'
            f'<div class="nm-sub">chance they come back and buy on their own within 7 days</div>'
            + chance_chart(hist, low, high)
            + f'<div class="nm-act z{on}"><small>Nudge would send</small><b><i></i>{ACTION_LINE[on]}</b></div>'
            f'<div class="nm-why">{esc(why_line(words))}</div>' + seen,
            unsafe_allow_html=True)
    with st.container(key="cartbox"):
        n = len(st.session_state.cart)
        st.markdown(f'<div class="bz-panel-title">Your cart <span>{n} item{"" if n == 1 else "s"}</span></div>',
                    unsafe_allow_html=True)
        for pid in st.session_state.cart:
            r = CATALOG.loc[pid]
            c1, c2 = st.columns([5, 1], vertical_alignment="center", gap="small")
            img = thumb(r["image"])
            c1.markdown(f'<div class="bz-citem">{f"<img src={chr(34)}{img}{chr(34)} alt={chr(34)}{chr(34)}>" if img else ""}'
                        f'<div><div class="t">{esc(r["name"])}</div><div class="p">{euro(r["price"])}</div></div></div>',
                        unsafe_allow_html=True)
            c2.button("Remove", key=f"rm_{pid}", on_click=remove_from_cart, args=(pid,), type="tertiary",
                      icon=":material/close:", help=f"Remove {r['name']} from the cart")
        if not st.session_state.cart:
            st.markdown('<div class="bz-cempty">Your cart is empty. Find something you didn\'t know you needed.</div>',
                        unsafe_allow_html=True)
        st.markdown(f'<div class="bz-total"><span>Total</span><b>{euro(cart_value)}</b></div>', unsafe_allow_html=True)
        st.button("Checkout", on_click=checkout, width="stretch", type="primary", disabled=not st.session_state.cart)
        st.button("Leave the shop", on_click=leave, width="stretch")

# --- Pages -------------------------------------------------------------------------------------------
with main:
    if page == "shop":
        shown = CATALOG
        if cat != "All":
            shown = shown[shown["category"] == cat]
        elif not query:  # classmates' products first, then the rest with categories mixed (first rows not all Kitchen)
            rest = shown[~shown["featured"]]
            rest = rest.assign(_r=rest.groupby("category").cumcount()).sort_values(["_r", "category"], kind="stable").drop(columns="_r")
            shown = pd.concat([shown[shown["featured"]], rest])
        if query:
            shown = shown[shown["name"].str.contains(query, case=False, regex=False)
                          | shown["description"].str.contains(query, case=False, regex=False)]
        title = f'Results for "{esc(query)}"' if query else ("All products" if cat == "All" else esc(cat))
        st.markdown(f'<div class="bz-listhead"><div class="h">{title}</div><span>{len(shown)} product{"" if len(shown) == 1 else "s"}</span></div>',
                    unsafe_allow_html=True)
        if shown.empty:
            st.markdown('<div class="bz-empty"><b>Nothing found</b>Try another word, or pick a category above.</div>',
                        unsafe_allow_html=True)
        product_grid(shown.head(st.session_state.limit), "g")
        if len(shown) > st.session_state.limit:
            _, mid, _ = st.columns([1, 1, 1])
            mid.button(f"Show more ({len(shown) - st.session_state.limit} left)", width="stretch",
                       on_click=lambda: st.session_state.update(limit=st.session_state.limit + PAGE_SIZE))

    elif page == "product":
        row = CATALOG.loc[st.session_state.product]
        st.button("Back to all products", on_click=go_shop, type="tertiary", icon=":material/arrow_back:")
        left_col, right_col = st.columns([1, 1], gap="large")
        with left_col, st.container(key="pdp_img"):
            product_image(row)
            if row["friend"]:   # easter egg: a real photo of the product "in the wild"
                st.markdown(f'<div class="bz-wild"><div class="bz-wild-title">Customer photos</div>'
                            f'<img src="{thumb(row["friend"], 900)}" alt="A customer with this product">'
                            f'<div class="bz-wild-cap">Spotted in the wild</div></div>', unsafe_allow_html=True)
        with right_col:
            st.markdown(f'<div class="bz-pdp-cat">{esc(row["category"])}</div>'
                        f'<div class="bz-pdp-name">{esc(row["name"])}</div>'
                        f'<div class="bz-big-price">{euro(row["price"])}</div><div class="bz-vat">VAT included</div>'
                        f'<div class="bz-desc">{esc(row["description"])}</div>', unsafe_allow_html=True)
            in_cart = row["id"] in st.session_state.cart
            st.button("In your cart" if in_cart else "Add to cart", on_click=add_to_cart, args=(row["id"],),
                      type="primary", disabled=in_cart, key="detail_cart",
                      icon=":material/check:" if in_cart else ":material/shopping_bag:")
            perk = "Free delivery" if row["price"] >= 50 else "Free delivery over €50"
            st.markdown(f'<ul class="bz-perks"><li>{icon("truck")}{perk}</li><li>{icon("clock")}Ships in 1–2 days</li>'
                        f'<li>{icon("return")}30-day returns, no questions asked</li></ul>', unsafe_allow_html=True)
        st.markdown('<div class="bz-section">You might also like</div>', unsafe_allow_html=True)
        similar = CATALOG[(CATALOG["category"] == row["category"]) & (CATALOG["id"] != row["id"])].head(4)
        product_grid(similar, "s")

    elif page == "bought":
        st.markdown(f'<div class="bz-thanks"><div class="tick">{icon("check", 28)}</div><h1>Thank you for your order</h1>'
                    f'<p>This shopper bought, so Nudge sends nothing: no reminder, no discount.</p></div>', unsafe_allow_html=True)
        _, mid, _ = st.columns([1, 0.6, 1])
        mid.button("New visitor", on_click=new_visitor, type="primary", width="stretch")
