"""The shared contract between the data, the model and the demo app.

Each click becomes ONE word (token) that says what kind of click it was, how expensive the product
was, whether it was seen before in this visit, and how long since the previous click
(start / quick <20 s / normal 20-70 s / long 70 s-30 min / away >30 min):

    view_expensive_again_long  =  viewed an expensive product, for the 2nd+ time, after a long pause
    cart_cheap_quick           =  added a cheap product to the cart, quickly after the previous click

A session starts with one word about the shopper, "new" or "returning" (had a visit in the previous 14 days),
followed by the click words in time order:

    ["returning", "view_expensive_start", "view_expensive_again_long", "cart_expensive_again_quick"]

Purchase is never part of the input: we predict it.

    predict(session) -> float   chance (0 to 1) that the shopper buys without help

The data has no remove_from_cart clicks (checked 20 Nov on the whole of October 2019),
so the click types are only view and cart.
"""

import hashlib
from pathlib import Path
import itertools
import math

import numpy as np

import pandas as pd

# --- Cut-offs (measured on the October 2019 sample, rounded) -------------------------------------
# Price: roughly a third of clicks in each band (terciles were €93 and €257).
PRICE_CHEAP_MAX = 100.0       # price <  100        -> cheap
PRICE_MID_MAX = 250.0         # 100 <= price < 250  -> mid, else expensive
# Seconds since the previous click (quartiles were 19 s, 35 s, 69 s).
GAP_QUICK_MAX = 20.0          # < 20 s              -> quick
GAP_NORMAL_MAX = 70.0         # 20-70 s             -> normal
GAP_LONG_MAX = 1800.0         # 70 s-30 min         -> long, else away (left and came back;
                              # 30 min = the usual web-analytics session timeout). First click -> start

CLICK_TYPES = ["view", "cart"]
PRICE_BANDS = ["cheap", "mid", "expensive"]
REPEAT = ["", "again"]
GAP_BANDS = ["start", "quick", "normal", "long", "away"]


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
    elif seconds_since_prev < GAP_LONG_MAX:
        gap = "long"
    else:
        gap = "away"
    parts = [event, band] + (["again"] if again else []) + [gap]
    return "_".join(parts)


START_WORDS = ["new", "returning"]
# History words (added 20 Nov, round 2), all from the previous LOOKBACK_DAYS only, placed right after the start word:
HISTORY_WORDS = ["bought_recently",                         # bought something in the previous 14 days
                 "visits_1", "visits_2_4", "visits_5plus",   # how many visits in the previous 14 days
                 "last_today", "last_1_3d", "last_4_14d"]    # time since the end of the previous visit

# --- Vocabulary: every possible word gets a number (index 0 = padding) ------------------------------
VOCAB = ["<pad>"] + START_WORDS + HISTORY_WORDS + [
    "_".join(p for p in (e, b, r, g) if p)
    for e, b, r, g in itertools.product(CLICK_TYPES, PRICE_BANDS, REPEAT, GAP_BANDS)
]
TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}
BLOCK_SIZE = 64   # the model reads at most 64 words (same as nudge/train.py)

# The app showed a warning banner while this was True (hand-written rule instead of the model).
IS_PLACEHOLDER = False   # app/model.pt holds the trained Transformer since 21 Nov


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


def history_words(visits_14d: int, bought_14d: bool, days_since_last: float | None) -> list[str]:
    """What we know about the shopper's previous 14 days, as words. Empty for a new shopper."""
    words = []
    if bought_14d:
        words.append("bought_recently")
    if visits_14d >= 5:
        words.append("visits_5plus")
    elif visits_14d >= 2:
        words.append("visits_2_4")
    elif visits_14d == 1:
        words.append("visits_1")
    if days_since_last is not None and visits_14d > 0:
        words.append("last_today" if days_since_last < 1 else "last_1_3d" if days_since_last < 4 else "last_4_14d")
    return words


def session_words(clicks: list[dict], returning: bool, visits_14d: int = 0, bought_14d: bool = False,
                  days_since_last: float | None = None) -> list[str]:
    """The full model input for one visit: start word + history words + click words. Used by the app."""
    if not returning:
        visits_14d, bought_14d, days_since_last = 0, False, None
    elif visits_14d == 0:
        visits_14d = 1
    return (["returning" if returning else "new"] + history_words(visits_14d, bought_14d, days_since_last)
            + tokens_from_clicks(clicks))


# --- Building the training data (rules decided 20 Nov after the data audit) -------------------------
IDLE_MINUTES = 60      # a visit ends after 60 min without a click (= when Hostinger Reach would send its email)
WINDOW_DAYS = 7        # label: did the shopper come back and buy within 7 days of that moment?
LOOKBACK_DAYS = 14     # "returning" = had a visit in the 14 days before this one
EVENT_ORDER = {"view": 0, "cart": 1, "purchase": 2}   # fixed order for clicks with the same timestamp


def clean_events(events: pd.DataFrame) -> pd.DataFrame:
    """Raw REES46 rows -> clean clicks sorted per shopper: exact duplicates removed, price <= 0 removed,
    ties in time ordered view < cart < purchase. Adds `time` (datetime) and `t` (seconds since 1970)."""
    ev = events.drop_duplicates(subset=["event_time", "event_type", "product_id", "user_id", "user_session"]).copy()
    ev = ev[ev["price"] > 0]
    ev["event_type"] = ev["event_type"].astype(str)
    if pd.api.types.is_datetime64_any_dtype(ev["event_time"]):          # the 1-in-4 build stores real dates
        ev["time"] = ev["event_time"].dt.tz_localize(None) if ev["event_time"].dt.tz is not None else ev["event_time"]
    else:
        ev["time"] = pd.to_datetime(ev["event_time"].astype(str).str.replace(" UTC", "", regex=False))
    ev["t"] = (ev["time"] - pd.Timestamp("1970-01-01")).dt.total_seconds()
    ev["rank"] = ev["event_type"].map(EVENT_ORDER)
    ev = ev.sort_values(["user_id", "time", "rank"], kind="stable").reset_index(drop=True)
    return ev


def build_visits(events: pd.DataFrame) -> pd.DataFrame:
    """Clean clicks -> one row per ABANDONED visit, with words, label and money columns.

    1. A visit = a shopper's clicks with no gap longer than IDLE_MINUTES (we ignore the shop's own
       session id, which does not reset when people leave).
    2. Visits that contain a purchase are not scored (the shopper bought; no message would be sent).
       A purchase within IDLE_MINUTES of the last click is, by rule 1, part of the same visit.
    3. Decision moment = last click + IDLE_MINUTES. Label 1 if the same shopper buys within WINDOW_DAYS
       after that moment (the data has no messages, so this is "without help").
    4. Start word "returning" if the shopper had any visit starting in the LOOKBACK_DAYS before this one.
    5. Only visits we can judge fairly are kept: start at least LOOKBACK_DAYS after the data starts
       (full history), and decision moment + WINDOW_DAYS before the data ends (full label window).
    Money columns: `cart_value` counts each carted product once; `order_value` = total of the purchases
    in the visit where the shopper came back and bought (what a discount would really apply to).
    """
    ev = clean_events(events)
    user = ev["user_id"].to_numpy()
    t = ev["t"].to_numpy()
    new_visit = np.ones(len(ev), dtype=bool)
    new_visit[1:] = (user[1:] != user[:-1]) | (t[1:] - t[:-1] > IDLE_MINUTES * 60)
    ev["visit"] = np.cumsum(new_visit) - 1

    is_buy = ev["event_type"].to_numpy() == "purchase"
    v = ev.groupby("visit", sort=True).agg(user_id=("user_id", "first"), start=("t", "min"), last=("t", "max"),
                                            n_clicks=("t", "size"))
    v["has_purchase"] = pd.Series(is_buy).groupby(ev["visit"]).any()
    v["order_value"] = ev["price"].where(is_buy, 0.0).groupby(ev["visit"]).sum()

    # History from the previous LOOKBACK_DAYS only (visits are sorted by shopper, then time):
    # number of earlier visits, whether any of them had a purchase, time since the previous visit ended.
    starts, lasts = v["start"].to_numpy(), v["last"].to_numpy()
    buys_cum = np.cumsum(v["has_purchase"].to_numpy().astype(int))
    users = v["user_id"].to_numpy()
    group_start = np.r_[0, np.flatnonzero(users[1:] != users[:-1]) + 1]
    group_end = np.r_[group_start[1:], len(v)]
    n_recent = np.zeros(len(v), dtype=int)
    bought = np.zeros(len(v), dtype=bool)
    for g0, g1 in zip(group_start, group_end):
        lo = g0 + np.searchsorted(starts[g0:g1], starts[g0:g1] - LOOKBACK_DAYS * 86400, side="left")
        idx = np.arange(g0, g1)
        n_recent[g0:g1] = idx - lo
        before = np.where(idx > g0, buys_cum[idx - 1], 0)
        at_lo = np.where(lo > g0, buys_cum[lo - 1], 0)
        bought[g0:g1] = (before - at_lo) > 0
    prev_last = np.r_[np.nan, lasts[:-1]]
    prev_last[group_start] = np.nan
    v["visits_14d"] = n_recent
    v["bought_14d"] = bought
    v["days_since_last"] = (starts - prev_last) / 86400
    v["returning"] = v["visits_14d"] > 0

    # Label: first later visit WITH a purchase, starting after the decision moment
    v["decide"] = v["last"] + IDLE_MINUTES * 60
    buys = v.loc[v["has_purchase"], ["user_id", "start", "order_value"]].rename(
        columns={"start": "buy_start", "order_value": "next_order_value"}).sort_values("buy_start")
    a = v.loc[~v["has_purchase"]].reset_index().sort_values("decide")
    a = pd.merge_asof(a, buys, left_on="decide", right_on="buy_start", by="user_id", direction="forward")
    a["days_to_buy"] = (a["buy_start"] - a["decide"]) / 86400
    a["label"] = (a["days_to_buy"] <= WINDOW_DAYS).astype(int)
    a.loc[a["label"] == 0, ["days_to_buy", "next_order_value"]] = np.nan

    data_start, data_end = t.min(), t.max()
    keep = (a["start"] >= data_start + LOOKBACK_DAYS * 86400) & (a["decide"] + WINDOW_DAYS * 86400 <= data_end)
    a = a.loc[keep].sort_values("visit").reset_index(drop=True)

    # Words for the kept visits (same tokens_from_clicks as the demo app)
    bounds = np.searchsorted(ev["visit"].to_numpy(), np.r_[a["visit"].to_numpy(), a["visit"].to_numpy() + 1])
    lo, hi = bounds[: len(a)], bounds[len(a):]
    et, pid, pr = ev["event_type"].to_numpy(), ev["product_id"].to_numpy(), ev["price"].to_numpy()
    tokens, cart_values = [], []
    for i, j, ret, nv, bt, dl in zip(lo, hi, a["returning"].to_numpy(), a["visits_14d"].to_numpy(),
                                     a["bought_14d"].to_numpy(), a["days_since_last"].to_numpy()):
        clicks = [{"event": et[k], "product_id": pid[k], "price": pr[k], "time": t[k]} for k in range(i, j)]
        tokens.append(["returning" if ret else "new"] + history_words(int(nv), bool(bt), None if np.isnan(dl) else float(dl))
                      + tokens_from_clicks(clicks))
        carted = {pid[k]: pr[k] for k in range(i, j) if et[k] == "cart"}
        cart_values.append(float(sum(carted.values())))
    a["tokens"] = tokens
    a["cart_value"] = cart_values
    a["start_time"] = pd.to_datetime(a["start"], unit="s")
    a["decide_time"] = pd.to_datetime(a["decide"], unit="s")
    a["split"] = split_of(a["user_id"])
    return a[["visit", "user_id", "start_time", "decide_time", "tokens", "n_clicks", "cart_value",
              "visits_14d", "bought_14d", "days_since_last", "label", "days_to_buy", "next_order_value", "split"]]


def split_of(user_ids: pd.Series) -> pd.Series:
    """Fixed train/valid/test group per shopper from an md5 hash of the id (stable on every computer):
    0-7 train (80%), 8 valid (10%), 9 test (10%)."""
    digit = user_ids.astype(str).map(lambda u: int(hashlib.md5(u.encode()).hexdigest(), 16) % 10)
    return digit.map(lambda d: "train" if d < 8 else "valid" if d == 8 else "test")


def encode(session: list[str]) -> list[int]:
    """Words -> numbers for the model, e.g. ["new", "view_cheap_start"] -> [1, 3]."""
    unknown = [w for w in session if w not in TOKEN_TO_ID or w == "<pad>"]
    if unknown:
        raise ValueError(f"Unknown words {unknown}")
    return [TOKEN_TO_ID[w] for w in session]


MODEL_PATH = Path(__file__).resolve().parent.parent / "app" / "model.pt"
_MODELS = None


def _load_models():
    """The 3 final Transformers (seeds 42, 43, 44) from app/model.pt, loaded once and kept in memory."""
    global _MODELS
    if _MODELS is None:
        import torch
        from nudge.transformer import VisitTransformer
        bundle = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        cfg = bundle["config"]
        _MODELS = []
        for state in bundle["state_dicts"]:
            model = VisitTransformer(len(VOCAB), BLOCK_SIZE, d=cfg["d"], heads=cfg["heads"], blocks=cfg["blocks"],
                                     dropout=cfg["dropout"])
            model.load_state_dict(state)
            model.eval()
            _MODELS.append(model)
    return _MODELS


def predict(session: list[str]) -> float:
    """Chance (0 to 1) that this shopper, leaving now, comes back and buys within 7 days WITHOUT a message.

    The trained Transformer (round 2, 21 Nov; test AUC 0.729): the average of the 3 seeds, exactly as scored in
    notebooks/05_evaluation.ipynb. Long visits keep the first word + the last BLOCK_SIZE - 1 words, as in training.
    """
    import torch
    ids = encode(session)
    if len(ids) > BLOCK_SIZE:
        ids = [ids[0]] + ids[-(BLOCK_SIZE - 1):]
    x = torch.tensor([ids])
    with torch.no_grad():
        probs = [torch.sigmoid(m(x)).item() for m in _load_models()]
    return float(sum(probs) / len(probs))
