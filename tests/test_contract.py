"""Checks for nudge/contract.py on hand-made shoppers. Run from the repo root:  .venv/bin/python tests/test_contract.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nudge.contract import build_visits, encode, split_of  # noqa: E402

D0 = pd.Timestamp("2019-10-01")


def ev(day, minute, kind, product, price, user):
    t = D0 + pd.Timedelta(days=day, minutes=minute)
    return (t.strftime("%Y-%m-%d %H:%M:%S UTC"), kind, product, price, user, f"s{user}")


rows = [
    # user 1: abandons on day 20 (cart), comes back day 22 and buys -> label 1, next_order_value 80
    ev(20, 0, "view", 1, 100.0, 1), ev(20, 1, "cart", 1, 100.0, 1), ev(20, 1, "cart", 1, 100.0, 1),  # duplicate cart
    ev(22, 0, "view", 9, 80.0, 1), ev(22, 2, "purchase", 9, 80.0, 1),
    # user 2: abandons day 20, buys 30 min later -> same visit (bought) -> not scored at all
    ev(20, 0, "view", 2, 50.0, 2), ev(20, 30, "purchase", 2, 50.0, 2),
    # user 3: visits day 10 and day 20 (10 days apart -> returning), never buys -> label 0
    ev(10, 0, "view", 3, 500.0, 3), ev(20, 0, "view", 3, 500.0, 3),
    # user 4: visits day 1 and day 20 (19 days apart -> NOT returning), buys day 29 (8.9 days later) -> label 0
    ev(1, 0, "view", 4, 20.0, 4), ev(20, 0, "view", 4, 20.0, 4), ev(29, 0, "purchase", 4, 20.0, 4),
    # user 5: visit on day 5 -> too early (no full 14-day history) -> dropped
    ev(5, 0, "view", 5, 20.0, 5),
    # user 6: visit on day 45 -> too late (no full 7-day window before data end) -> dropped
    ev(45, 0, "view", 6, 20.0, 6),
    # user 7: price 0 click only -> removed by cleaning
    ev(20, 0, "view", 7, 0.0, 7),
    # data end marker (user 8 on day 50)
    ev(50, 0, "view", 8, 20.0, 8),
]
events = pd.DataFrame(rows, columns=["event_time", "event_type", "product_id", "price", "user_id", "user_session"])
v = build_visits(events).set_index("user_id")

assert set(v.index) == {1, 3, 4}, f"wrong visits kept: {sorted(v.index)}"
assert v.loc[1, "label"] == 1 and v.loc[1, "next_order_value"] == 80.0
assert v.loc[1, "cart_value"] == 100.0, "duplicate cart counted twice"
assert v.loc[1, "tokens"][0] == "new"
assert v.loc[3, "label"] == 0 and v.loc[3, "tokens"][0] == "returning"
assert v.loc[4, "label"] == 0 and v.loc[4, "tokens"][0] == "new", "19 days back is not returning; 8.9 days is outside 7"
for t in v["tokens"]:
    encode(t)
s = split_of(pd.Series(range(100_000)))
share = s.value_counts(normalize=True)
assert abs(share["train"] - 0.8) < 0.01 and abs(share["test"] - 0.1) < 0.01
assert split_of(pd.Series([12345]))[0] == split_of(pd.Series([12345]))[0]
print("all contract tests passed")

# --- evaluate.py ---------------------------------------------------------------------------------
from nudge.evaluate import money, one_per_week  # noqa: E402

t = pd.Timestamp("2019-11-01")
m = pd.DataFrame({
    "user_id": [1, 1, 1, 2, 3],
    "decide_time": [t, t + pd.Timedelta(days=2), t + pd.Timedelta(days=9), t, t],
    "cart_value": [100.0, 100.0, 100.0, 50.0, 0.0],
    "label": [1, 1, 0, 0, 1],
    "next_order_value": [80.0, 80.0, None, None, 30.0],
})
give = one_per_week(m, m["cart_value"].to_numpy() > 0)
assert give.tolist() == [True, False, True, True, False], give   # 2nd discount within 7 days blocked
r = money(m, m["cart_value"].to_numpy() > 0)                      # industry rule: every cart abandoner
assert r["discounts_sent"] == 3 and abs(r["wasted_eur"] - 8.0) < 1e-9, r
assert r["left_without_nudge"] == 0
r = money(m, m["cart_value"].to_numpy() < 0)                      # policy that sends nothing
assert r["wasted_eur"] == 0 and r["left_without_nudge"] == 2 and r["spared_discount_came_back"] == 2, r
print("all evaluate tests passed")

# --- history words (round 2) -----------------------------------------------------------------------
from nudge.contract import history_words, session_words  # noqa: E402

h = pd.DataFrame([
    ev(10, 0, "view", 1, 50.0, 9), ev(10, 1, "purchase", 1, 50.0, 9),      # day 10: a visit with a purchase
    ev(12, 0, "view", 2, 50.0, 9),                                          # day 12: browsing
    ev(20, 0, "view", 3, 50.0, 9),                                          # day 20: scored visit
    ev(30, 0, "view", 4, 50.0, 9),                                          # day 30: 10 days later, only day 20 in window
    ev(50, 0, "view", 8, 20.0, 8),                                          # data end marker
    ev(0, 0, "view", 8, 20.0, 7),                                           # data start marker (day 0)
], columns=["event_time", "event_type", "product_id", "price", "user_id", "user_session"])
hv = build_visits(h).set_index("start_time")
d20, d30 = hv.loc[D0 + pd.Timedelta(days=20)], hv.loc[D0 + pd.Timedelta(days=30)]
assert d20["tokens"][:4] == ["returning", "bought_recently", "visits_2_4", "last_4_14d"], d20["tokens"]
assert d30["tokens"][:3] == ["returning", "visits_1", "last_4_14d"], d30["tokens"]          # day-10 purchase is >14 days back
assert session_words([], returning=False, visits_14d=3, bought_14d=True) == ["new"]
assert session_words([], returning=True, bought_14d=True, days_since_last=0.5) == ["returning", "bought_recently", "visits_1", "last_today"]
for t in hv["tokens"]:
    encode(t)
print("all history tests passed")
