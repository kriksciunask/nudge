"""Nudge demo: a fake electronics shop.

Run from the repo root:  python -m streamlit run app/app.py

A visitor views products, adds/removes them from the cart, then clicks "Leave the shop".
The click sequence goes to predict() (nudge/contract.py), and the decision layer
(nudge/decision.py) picks: no message / reminder / reminder + 10%.
"""

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nudge.contract import IS_PLACEHOLDER, predict, session_words  # noqa: E402
from nudge.decision import (  # noqa: E402
    DISCOUNT,
    DISCOUNT_RATE,
    LABELS,
    NOTHING,
    REMINDER,
    cutoffs,
    decide,
    expected_wasted_discount,
)

# ~6 products from categories that exist in the training data (multi-category store, electronics).
PRODUCTS = [
    {"id": "phone", "name": "Smartphone", "icon": "📱", "price": 299},
    {"id": "laptop", "name": "Laptop", "icon": "💻", "price": 749},
    {"id": "headphones", "name": "Headphones", "icon": "🎧", "price": 89},
    {"id": "watch", "name": "Smartwatch", "icon": "⌚", "price": 179},
    {"id": "tablet", "name": "Tablet", "icon": "📟", "price": 349},
    {"id": "speaker", "name": "Speaker", "icon": "🔊", "price": 59},
]
BY_ID = {p["id"]: p for p in PRODUCTS}

st.set_page_config(page_title="Nudge demo shop", page_icon="🛒", layout="wide")

# --- Session state: the click log (event, product) and the cart -------------------------
if "clicks" not in st.session_state:
    st.session_state.clicks = []  # list of {event, product_id, price, time}
    st.session_state.cart = []  # product ids in cart
    st.session_state.left = False


def click(event: str, pid: str) -> None:
    # Remove is a demo convenience only: the training data has no remove clicks, so it is not sent to the model.
    if event == "remove":
        st.session_state.cart.remove(pid)
    else:
        now = time.time()
        if event == "cart" and all(c["product_id"] != pid for c in st.session_state.clicks):
            # Real shoppers almost always open a product before adding it (99.6% in the data), so the
            # model has barely seen a cart without a view. Record the product view first.
            st.session_state.clicks.append({"event": "view", "product_id": pid, "price": BY_ID[pid]["price"], "time": now})
            now += 1
        st.session_state.clicks.append({"event": event, "product_id": pid, "price": BY_ID[pid]["price"], "time": now})
        if event == "cart":
            st.session_state.cart.append(pid)
    st.session_state.left = False


def reset() -> None:
    st.session_state.clicks = []
    st.session_state.cart = []
    st.session_state.left = False


# --- Sidebar: the shop owner's one setting -----------------------------------------------
with st.sidebar:
    st.header("Shop owner settings")
    generosity = st.slider("How generous are you with discounts?", 0, 100, 50, step=5) / 100
    low, high = cutoffs(generosity)
    st.caption(
        f"Below **{low:.0%}**: no message  \n"
        f"{low:.0%}–{high:.0%}: reminder + {DISCOUNT_RATE:.0%}  \n"
        f"Above **{high:.0%}**: reminder only"
    )
    returning = st.toggle("Customer has visited before", value=False)
    st.button("New visitor (reset)", on_click=reset, use_container_width=True)

st.title("🛒 Nudge demo shop")
if IS_PLACEHOLDER:
    st.warning("Placeholder model: predictions come from a hand-written rule, not the trained Transformer yet.")

# --- Product grid ----------------------------------------------------------------------
cols = st.columns(3)
for i, p in enumerate(PRODUCTS):
    with cols[i % 3]:
        with st.container(border=True):
            st.subheader(f"{p['icon']} {p['name']}")
            st.write(f"€{p['price']}")
            a, b = st.columns(2)
            a.button("View", key=f"view_{p['id']}", on_click=click, args=("view", p["id"]),
                     use_container_width=True)
            if p["id"] in st.session_state.cart:
                b.button("Remove", key=f"rm_{p['id']}", on_click=click, args=("remove", p["id"]),
                         use_container_width=True)
            else:
                b.button("Add to cart", key=f"cart_{p['id']}", on_click=click, args=("cart", p["id"]),
                         use_container_width=True, type="primary")

# --- Live click sequence ---------------------------------------------------------------
st.subheader("Your clicks so far")
session = session_words(st.session_state.clicks, returning)
if st.session_state.clicks:
    st.write(f"`{session[0]}` → " + " → ".join(
        f"{BY_ID[c['product_id']]['icon']} `{w}`" for c, w in zip(st.session_state.clicks, session[1:])))
    st.caption("Each click becomes one word: type, price band, 'again' if seen before, time since previous click.")
else:
    st.caption("No clicks yet.")
cart_value = sum(BY_ID[pid]["price"] for pid in st.session_state.cart)
st.write(f"Cart: {len(st.session_state.cart)} item(s), €{cart_value}")

if st.button("🚪 Leave the shop", type="primary"):
    st.session_state.left = True

# --- What Nudge decides ------------------------------------------------------------------
if st.session_state.left:
    p = predict(session)
    action = decide(p, generosity)
    st.divider()
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("What Nudge decides")
        st.metric("Chance they come back and buy within 7 days, without a message", f"{p:.0%}")
        st.write(f"**Decision:** {LABELS[action]}")
        if action == DISCOUNT and cart_value:
            waste = expected_wasted_discount(p, cart_value)
            st.caption(f"Discount €{DISCOUNT_RATE * cart_value:.2f}; expected wasted on would-buy-anyway: "
                       f"{p:.0%} × €{DISCOUNT_RATE * cart_value:.2f} = €{waste:.2f}")
        elif action == REMINDER and cart_value:
            st.caption(f"Saved €{DISCOUNT_RATE * cart_value:.2f} of margin: this shopper likely buys anyway.")

    with right_col:
        st.subheader("Message preview")
        first_item = BY_ID[st.session_state.cart[0]]["name"] if st.session_state.cart else None
        if action == NOTHING:
            st.info("No message sent: looks like just browsing.")
        else:
            thing = f"your {first_item}" if first_item else "the items you looked at"
            text = f"Hi! You left {thing} in our shop. It's still waiting for you 🙂"
            if action == DISCOUNT:
                text += f" Here's {DISCOUNT_RATE:.0%} off if you order today: <b>NUDGE10</b>"
            st.markdown(
                f"""<div style="background:#e5ddd5;padding:16px;border-radius:12px;">
                <div style="background:#dcf8c6;color:#111;padding:10px 14px;border-radius:8px;
                max-width:90%;margin-left:auto;font-size:15px;">{text}
                <div style="text-align:right;font-size:11px;color:#667;">now ✓✓</div></div></div>""",
                unsafe_allow_html=True,
            )

# Not built yet (see CLAUDE.md "Demo"): profit vs "discount everyone" chart, attention heatmap.
