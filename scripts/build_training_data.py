"""Build the training data on the Mac from the Kaggle download.

Input:  data/raw/events_<size>.parquet  (from Kaggle kernel nudge-01-data-build; 10pct = 1 in 10, 25pct = 1 in 4 shoppers)
Output: data/raw/visits_<size>.parquet  (all abandoned visits with words, label, money, split; not in git)
        data/sample/events_sample.parquet + visits_sample.parquet  (1 in 150 shoppers; in git for the team)
Run from the repo root:  .venv/bin/python scripts/build_training_data.py 25pct   (default 10pct)
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nudge.contract import build_visits  # noqa: E402

t0 = time.time()
size = sys.argv[1] if len(sys.argv) > 1 else "10pct"
cols = ["event_time", "event_type", "product_id", "category_code", "price", "user_id", "user_session"]
events = pd.read_parquet(f"data/raw/events_{size}.parquet", columns=cols)
print(f"read {len(events):,} clicks in {time.time() - t0:.0f} s", flush=True)

visits = build_visits(events)
visits.to_parquet(f"data/raw/visits_{size}.parquet", index=False)
print(f"built {len(visits):,} visits in {time.time() - t0:.0f} s", flush=True)

small = events["user_id"] % 150 == 0
if size == "10pct":   # the team sample stays the same 1 in 150 shoppers
    events[small].to_parquet("data/sample/events_sample.parquet", index=False)
    visits[visits["user_id"] % 150 == 0].to_parquet("data/sample/visits_sample.parquet", index=False)

v = visits
cart = v["cart_value"] > 0
print("\nvisits by split and month (label = comes back and buys within 7 days):")
print(v.groupby(["split", v["start_time"].dt.strftime("%Y-%m")])["label"].agg(visits="size", label_pct="mean")
      .assign(label_pct=lambda d: (100 * d["label_pct"]).round(1)))
print(f"\nall: {len(v):,} visits, {100 * v.label.mean():.1f}% come back and buy")
print(f"cart abandoners: {cart.sum():,}, {100 * v[cart].label.mean():.1f}% come back and buy")
print(f"returning share: {100 * (v.tokens.str[0] == 'returning').mean():.1f}% | come-back rate returning "
      f"{100 * v[v.tokens.str[0] == 'returning'].label.mean():.1f}% vs new {100 * v[v.tokens.str[0] == 'new'].label.mean():.1f}%")
print(f"returning share by week: {(v.tokens.str[0] == 'returning').groupby(v.start_time.dt.isocalendar().week).mean().mul(100).round(0).to_dict()}")
print(f"length (words incl. start): p50 {v.tokens.str.len().median():.0f}, p99 {v.tokens.str.len().quantile(.99):.0f}, max {v.tokens.str.len().max()}")
print(f"shoppers in two splits: {v.groupby('user_id')['split'].nunique().gt(1).sum()}")
print(f"sample files: {len(events[small]):,} clicks, {(visits['user_id'] % 150 == 0).sum():,} visits")
print(f"done in {time.time() - t0:.0f} s")
