"""The shared contract between the data, the model and the demo app.

Each click becomes ONE word (token) that says what kind of click it was, how expensive the product
was, whether it was seen before in this visit, and how long since the previous click:

    view_expensive_again_long  =  viewed an expensive product, for the 2nd+ time, after a long pause
    cart_cheap_quick           =  added a cheap product to the cart, quickly after the previous click

A session is the list of these words in time order. Purchase is never part of the input: we predict it.

    predict(session) -> float   chance (0 to 1) that the shopper buys without help

The data has no remove_from_cart clicks (checked 20 Nov on the whole of October 2019),
so the click types are only view and cart.
"""

import itertools
import math

import pandas as pd

# --- Cut-offs (measured on the October 2019 sample, rounded) -------------------------------------
# Price: roughly a third of clicks in each band (terciles were €93 and €257).
PRICE_CHEAP_MAX = 100.0       # price <  100        -> cheap
PRICE_MID_MAX = 250.0         # 100 <= price < 250  -> mid, else expensive
# Seconds since the previous click (quartiles were 19 s, 35 s, 69 s).
GAP_QUICK_MAX = 20.0          # < 20 s              -> quick
GAP_NORMAL_MAX = 70.0         # 20-70 s             -> normal, else long. First click -> start

CLICK_TYPES = ["view", "cart"]
PRICE_BANDS = ["cheap", "mid", "expensive"]
REPEAT = ["", "again"]
GAP_BANDS = ["start", "quick", "normal", "long"]


def click_token(event: str, price: float, again: bool, seconds_since_prev: float | None) -> str:
    """One click -> one word, e.g. ("view", 799.0, True, 95.0) -> "view_expensive_again_long"."""
    if event not in CLICK_TYPES:
        raise ValueError(f"Unknown click type {event!r}; allowed: {CLICK_TYPES}")
    band = "cheap" if price < PRICE_CHEAP_MAX else "mid" if price < PRICE_MID_MAX else "expensive"
    if seconds_since_prev is None:
        gap = "start"
    elif seconds_since_prev < GAP_QUICK_MAX:
        gap = "quick"
    elif seconds_since_prev < GAP_NORMAL_MAX:
        gap = "normal"
    else:
        gap = "long"
    parts = [event, band] + (["again"] if again else []) + [gap]
    return "_".join(parts)


# --- Vocabulary: every possible word gets a number (index 0 = padding) ------------------------------
VOCAB = ["<pad>"] + [
    "_".join(p for p in (e, b, r, g) if p)
    for e, b, r, g in itertools.product(CLICK_TYPES, PRICE_BANDS, REPEAT, GAP_BANDS)
]
TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}

# True until the trained model (app/model.pt) is plugged in. The app shows a banner while True.
IS_PLACEHOLDER = True


def tokens_from_clicks(clicks: list[dict]) -> list[str]:
    """Clicks in time order -> words. Each click: {"event", "product_id", "price", "time" (seconds or Timestamp)}.

    Used by BOTH the data pipeline and the demo app, so they always build words the same way.
    """
    seen, words, prev_time = set(), [], None
    for c in clicks:
        if prev_time is None:
            gap = None
        else:
            delta = c["time"] - prev_time
            gap = delta.total_seconds() if hasattr(delta, "total_seconds") else float(delta)
        words.append(click_token(c["event"], c["price"], c["product_id"] in seen, gap))
        seen.add(c["product_id"])
        prev_time = c["time"]
    return words


def build_sessions(events: pd.DataFrame) -> pd.DataFrame:
    """Raw clicks (one row per click, REES46 columns) -> one row per session with words and a label.

    Sort each session by time. If it contains a purchase, keep only the clicks BEFORE the first purchase
    and label 1 (the model must never see the purchase). Otherwise keep all clicks and label 0.
    Sessions that start with a purchase are dropped (nothing to predict from).
    """
    ev = events.copy()
    ev["time"] = pd.to_datetime(ev["event_time"].str.replace(" UTC", "", regex=False))
    ev = ev.sort_values(["user_session", "time"], kind="stable")
    rows = []
    for session_id, g in ev.groupby("user_session", sort=False):
        types = g["event_type"].tolist()
        label = int("purchase" in types)
        if label:
            g = g.iloc[: types.index("purchase")]
        if len(g) == 0:
            continue
        clicks = [
            {"event": e, "product_id": p, "price": pr, "time": t}
            for e, p, pr, t in zip(g["event_type"], g["product_id"], g["price"], g["time"])
        ]
        rows.append({
            "user_session": session_id,
            "user_id": int(g["user_id"].iloc[0]),
            "start_time": g["time"].iloc[0],
            "tokens": tokens_from_clicks(clicks),
            "n_clicks": len(g),
            "cart_value": float(g.loc[g["event_type"] == "cart", "price"].sum()),
            "label": label,
        })
    return pd.DataFrame(rows)


def encode(session: list[str]) -> list[int]:
    """Words -> numbers for the model, e.g. ["view_cheap_start"] -> [1]."""
    unknown = [w for w in session if w not in TOKEN_TO_ID or w == "<pad>"]
    if unknown:
        raise ValueError(f"Unknown words {unknown}")
    return [TOKEN_TO_ID[w] for w in session]


def predict(session: list[str]) -> float:
    """Chance (0 to 1) that this session ends in a purchase without a nudge.

    >>> PLACEHOLDER <<<  This is a hand-written rule, NOT the trained model.
    It exists only so the app can be built before the Transformer is trained.
    Replace the body with: load app/model.pt, encode(session), run the model, sigmoid.
    """
    encode(session)  # validates the words
    if not session:
        return 0.02
    score = -2.8
    score += 0.05 * min(len(session), 20)
    score += 1.2 * sum(w.startswith("cart") for w in session)
    score += 0.4 * sum("_again" in w for w in session)
    score += 0.2 * sum(w.endswith("_long") for w in session)
    if session[-1].startswith("cart"):
        score += 0.5
    return 1 / (1 + math.exp(-score))
