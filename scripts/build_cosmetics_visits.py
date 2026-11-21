"""Second-shop test, step 1: cosmetics shop clicks -> visits, with EXACTLY the same rules as the main data.

Input:  data/raw/cosmetics/2019-Oct.csv, 2019-Nov.csv  (REES46 cosmetics shop; not in git)
Output: data/raw/cosmetics/visits_25pct.parquet, data/raw/cosmetics/build_stats.json
Same sampling as the main data (user_id % 4 == 0), same months (Oct 1 - Nov 30 2019), same build_visits.
remove_from_cart (and any other type outside view/cart/purchase) is dropped, as the contract says.
Run from the repo root:  .venv/bin/python scripts/build_cosmetics_visits.py
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nudge.contract import build_visits  # noqa: E402

t0 = time.time()
cols = ["event_time", "event_type", "product_id", "category_code", "price", "user_id", "user_session"]
parts, types_all, types_kept_users = [], pd.Series(dtype="int64"), pd.Series(dtype="int64")
for month in ["2019-Oct", "2019-Nov"]:
    for chunk in pd.read_csv(f"data/raw/cosmetics/{month}.csv", usecols=cols, chunksize=3_000_000):
        types_all = types_all.add(chunk["event_type"].value_counts(), fill_value=0)
        keep = chunk[chunk["user_id"] % 4 == 0].copy()
        types_kept_users = types_kept_users.add(keep["event_type"].value_counts(), fill_value=0)
        keep = keep[keep["event_type"].isin(["view", "cart", "purchase"])]
        keep["event_time"] = pd.to_datetime(keep["event_time"], format="%Y-%m-%d %H:%M:%S UTC")
        parts.append(keep)
    print(month, "done", round(time.time() - t0), "s", flush=True)
events = pd.concat(parts, ignore_index=True).dropna(subset=["user_session"])
del parts
print(f"{len(events):,} clicks, {events.user_id.nunique():,} shoppers", flush=True)

visits = build_visits(events)
visits.to_parquet("data/raw/cosmetics/visits_25pct.parquet", index=False)

# price band mix of the clicks the model sees (cheap <100, mid 100-250, expensive >=250), main shop for contrast
band = lambda p: pd.cut(p, [0, 100, 250, float("inf")], right=False, labels=["cheap", "mid", "expensive"])  # noqa: E731
mix = band(events.loc[events.price > 0, "price"]).value_counts(normalize=True).round(4).to_dict()
main = pd.read_parquet("data/raw/events_25pct.parquet", columns=["price"])
main_mix = band(main.loc[main.price > 0, "price"]).value_counts(normalize=True).round(4).to_dict()
stats = {"events_all_users_by_type": types_all.astype(int).to_dict(),
         "events_sampled_users_by_type": types_kept_users.astype(int).to_dict(),
         "clicks_kept": int(len(events)), "shoppers": int(events.user_id.nunique()), "visits": int(len(visits)),
         "base_rate": float(visits.label.mean()),
         "price_band_share_cosmetics": {str(k): float(v) for k, v in mix.items()},
         "price_band_share_main": {str(k): float(v) for k, v in main_mix.items()},
         "median_price_cosmetics": float(events.price.median()), "median_price_main": float(main.price.median())}
Path("data/raw/cosmetics/build_stats.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
print(f"done in {time.time() - t0:.0f} s")
