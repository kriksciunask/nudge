"""Money curve for the owner view: wasted discount money and shoppers left without a nudge,
for every setting of the "how generous are you with discounts" slider (generosity g = 0, 0.05, ..., 1).

Policy = nudge/decision.py: a cart abandoner gets "reminder + 10%" when low <= score < high,
where (low, high) = cutoffs(g). Same rules as nudge/evaluate.py money(): TEST split, cart abandoners only,
at most one discount per shopper per 7 days, wasted = 10% of what shoppers who came back anyway actually bought.
Scores = average of the 3 final Transformer seeds (results/tuned/transformer_4eb04ce3_s{42,43,44}_test.npy),
in test-split row order (the order notebooks/05_evaluation.ipynb uses).

How this relates to evaluation_money.csv: "Transformer (60%)" there gives a discount to the 60% of cart abandoners
with the LOWEST scores (cut-off = 60th percentile of validation cart scores, no lower limit). The decision layer's
HIGH cut-off at g=0.5 is that same validation number (0.2146). It adds a LOWER limit (just browsing gets nothing,
8% at g=0 down to 3% at g=1), so g=0.5 differs slightly from the pure @60% numbers; the script prints both.
The industry rule row must match exactly (printed below).

Run: .venv/bin/python scripts/make_money_curve.py   (light: ~1 min, one process)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nudge.decision import cutoffs  # noqa: E402
from nudge.evaluate import bootstrap, money  # noqa: E402

tests = [np.load(ROOT / f"results/tuned/transformer_4eb04ce3_s{s}_test.npy") for s in (42, 43, 44)]
v = pd.read_parquet(ROOT / "data/raw/visits_25pct.parquet",
                    columns=["split", "user_id", "label", "decide_time", "cart_value", "next_order_value"])
test = v[v["split"] == "test"].reset_index(drop=True)
score = np.mean(tests, axis=0)
assert len(score) == len(test)

KEYS = ("discounts_sent", "wasted_eur", "came_back_anyway_with_discount", "left_without_nudge")
GRID = [round(i * 0.05, 2) for i in range(21)]
BOOT_G = (0.0, 0.5, 1.0)
rule = money(test, np.ones(len(test), dtype=bool))

def give_at(g):
    low, high = cutoffs(g)
    return (score >= low) & (score < high)

rows = []
for g in GRID:
    low, high = cutoffs(g)
    m = money(test, give_at(g))
    rows.append({"g": g, "low": round(low, 4), "high": round(high, 4), **{k: m[k] for k in KEYS},
                 "rule_discounts_sent": rule["discounts_sent"], "rule_wasted_eur": rule["wasted_eur"],
                 "rule_came_back_anyway_with_discount": rule["came_back_anyway_with_discount"],
                 "rule_left_without_nudge": rule["left_without_nudge"],
                 "wasted_lo": np.nan, "wasted_hi": np.nan})
curve = pd.DataFrame(rows)

frame = test.copy()
for g in BOOT_G:
    frame[f"give{g}"] = give_at(g)
cols = ["user_id", "label", "decide_time", "cart_value", "next_order_value"] + [f"give{g}" for g in BOOT_G]
rng = bootstrap(frame[cols], lambda d: {g: money(d, d[f"give{g}"].to_numpy())["wasted_eur"] for g in BOOT_G}, n=200, seed=0)
for g in BOOT_G:
    curve.loc[curve["g"] == g, ["wasted_lo", "wasted_hi"]] = rng[g]

curve.to_csv(ROOT / "results/money_curve.csv", index=False)
print(curve[curve["g"].isin(BOOT_G)].T.to_string())

off = pd.read_csv(ROOT / "results/evaluation_money.csv").set_index("policy")
r = off.loc["Industry rule (everyone)"]
print("\nCHECK industry rule vs evaluation_money.csv:",
      rule["wasted_eur"].__round__() == r["€ wasted"] and rule["discounts_sent"] == r["discounts"]
      and rule["left_without_nudge"] == r["left without a nudge"], f"(wasted {rule['wasted_eur']:.0f} vs {r['€ wasted']})")
# Reproduce the official Transformer (60%) from the same scores (needs validation scores)
val = v[v["split"] == "valid"].reset_index(drop=True)
vs = np.mean([np.load(ROOT / f"results/tuned/transformer_4eb04ce3_s{s}_val.npy") for s in (42, 43, 44)], axis=0)
cut = np.quantile(vs[(val["cart_value"] > 0).to_numpy()], 0.6)
m60 = money(test, score < cut)
o = off.loc["Transformer (60%)"]
print(f"CHECK Transformer@60% reproduced: wasted {m60['wasted_eur']:.0f} (official {o['€ wasted']}), left {m60['left_without_nudge']} (official {o['left without a nudge']})")
lo5, hi5 = cutoffs(0.5)
c5 = curve[curve["g"] == 0.5].iloc[0]
print(f"g=0.5 cut-offs low {lo5:.3f} high {hi5:.4f} (validation 60th percentile = {cut:.4f}); "
      f"with the low limit: wasted {c5['wasted_eur']:.0f}, left {c5['left_without_nudge']:.0f}; pure @60%: wasted {m60['wasted_eur']:.0f}, left {m60['left_without_nudge']}")
