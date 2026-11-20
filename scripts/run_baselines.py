"""Baselines: random, industry rule, logistic regression on word counts.

Input:  data/raw/visits_10pct.parquet (scripts/build_training_data.py)
Output: results/baselines.md (tables for the report) and results/baselines.json
Run from the repo root:  .venv/bin/python scripts/run_baselines.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nudge.contract import VOCAB  # noqa: E402
from nudge.evaluate import bootstrap, money, pick_cutoff, quality  # noqa: E402

SEED = 0
t0 = time.time()
v = pd.read_parquet("data/raw/visits_10pct.parquet")
tr, va, te = (v[v["split"] == s].reset_index(drop=True) for s in ("train", "valid", "test"))
print(f"train {len(tr):,} | valid {len(va):,} | test {len(te):,} visits", flush=True)

# --- 1. Logistic regression on counts of every word (no order) ------------------------------------
vec = CountVectorizer(analyzer=lambda words: words, vocabulary=VOCAB[1:])
Xtr, Xva, Xte = (vec.transform(d["tokens"]) for d in (tr, va, te))
lr = LogisticRegression(max_iter=2000, C=1.0)
lr.fit(Xtr, tr["label"])
print(f"logistic regression trained in {time.time() - t0:.0f} s", flush=True)

rng = np.random.default_rng(SEED)
scores = {
    "Random": (rng.uniform(size=len(va)), rng.uniform(size=len(te))),
    # Industry rule as a score: "has a cart" (1) or not (0). Ties broken randomly so AUC is fair.
    "Industry rule (cart = 1)": tuple((d["cart_value"].to_numpy() > 0) + 1e-6 * rng.uniform(size=len(d)) for d in (va, te)),
    "Logistic regression": (lr.predict_proba(Xva)[:, 1], lr.predict_proba(Xte)[:, 1]),
}

# --- 2. Prediction quality on test, cut-off picked on valid ---------------------------------------
rows, ranges = [], {}
for name, (s_va, s_te) in scores.items():
    cut = pick_cutoff(va["label"].to_numpy(), s_va)
    q = quality(te["label"].to_numpy(), s_te, cut)
    te_ = te.assign(_s=s_te)
    ci = bootstrap(te_, lambda d, c=cut: {k: quality(d["label"].to_numpy(), d["_s"].to_numpy(), c)[k]
                                          for k in ("auc", "pr_auc", "f1")}, n=200, seed=SEED)
    one = te_["n_clicks"] == 1
    q_one = quality(te_.loc[one, "label"].to_numpy(), te_.loc[one, "_s"].to_numpy(), cut)
    rows.append({"model": name, "cutoff": cut, **q, "auc_one_click": q_one["auc"]})
    ranges[name] = ci
    print(f"{name}: AUC {q['auc']:.3f} PR-AUC {q['pr_auc']:.3f} F1 {q['f1']:.3f}", flush=True)

# --- 3. Money on cart abandoners: same number of discounts, chosen by the model vs at random -------
cart_va = va["cart_value"].to_numpy() > 0
p_va_cart = scores["Logistic regression"][0][cart_va]
p_te = scores["Logistic regression"][1]
money_rows = []
for keep in (1.0, 0.8, 0.6, 0.4):
    # LR policy: give the discount to the `keep` share LEAST likely to come back on their own
    cut = np.quantile(p_va_cart, keep) if keep < 1 else np.inf
    r_draw = np.random.default_rng(SEED + 1).uniform(size=len(te))
    pol = {
        "Logistic regression": lambda d, c=cut: d["_p"].to_numpy() < c,
        "Random": lambda d, k=keep: d["_r"].to_numpy() < k,
    }
    te_ = te.assign(_p=p_te, _r=r_draw)
    for name, f in pol.items():
        if keep == 1.0 and name == "Random":
            continue
        label = "Industry rule (everyone)" if keep == 1.0 else name
        m = money(te_, f(te_))
        ci = bootstrap(te_, lambda d, f=f: money(d, f(d)), n=200, seed=SEED)
        money_rows.append({"policy": label, "keep_share": keep, **m,
                           "wasted_eur_95": ci["wasted_eur"], "left_95": ci["left_without_nudge"]})
        print(f"money keep {keep:.0%} {label}: wasted €{m['wasted_eur']:,.0f}, left without nudge {m['left_without_nudge']}", flush=True)

# --- 4. What logistic regression learned: words that most raise / lower the come-back chance -------
coef = pd.Series(lr.coef_[0], index=vec.get_feature_names_out()).sort_values()
used = pd.Series(np.asarray(Xtr.sum(axis=0)).ravel(), index=vec.get_feature_names_out())
coef = coef[used[coef.index] >= 200]

# --- Save ---------------------------------------------------------------------------------------
Path("results").mkdir(exist_ok=True)
json.dump({"quality": rows, "quality_95": ranges, "money": money_rows,
           "coef_top": coef.tail(8).to_dict(), "coef_bottom": coef.head(8).to_dict(),
           "sizes": {"train": len(tr), "valid": len(va), "test": len(te)}, "seed": SEED},
          open("results/baselines.json", "w"), indent=1, default=float)

fmt = lambda x: f"{x:.3f}"  # noqa: E731
md = ["# Baselines (test shoppers, never seen in training)", "",
      f"Visits: train {len(tr):,}, valid {len(va):,}, test {len(te):,}. Label = comes back and buys within 7 days "
      f"({100 * te['label'].mean():.1f}% of test visits). 95% ranges from resampling shoppers 200 times.", "",
      "## Prediction quality", "",
      "| Model | AUC | PR-AUC | Precision | Recall | F1 | AUC on one-click visits |", "|---|---|---|---|---|---|---|"]
for r in rows:
    c = ranges[r["model"]]
    md.append(f"| {r['model']} | {fmt(r['auc'])} ({fmt(c['auc'][0])}–{fmt(c['auc'][1])}) | {fmt(r['pr_auc'])} "
              f"({fmt(c['pr_auc'][0])}–{fmt(c['pr_auc'][1])}) | {fmt(r['precision'])} | {fmt(r['recall'])} | "
              f"{fmt(r['f1'])} ({fmt(c['f1'][0])}–{fmt(c['f1'][1])}) | {fmt(r['auc_one_click'])} |")
md += ["", "## Money on cart abandoners (10% discount, max one per shopper per week)", "",
       "| Policy | Discounts sent | € given to shoppers who came back anyway | Spared a discount and came back | Left without a nudge |",
       "|---|---|---|---|---|"]
for m in money_rows:
    lo, hi = m["wasted_eur_95"]
    l_lo, l_hi = m["left_95"]
    md.append(f"| {m['policy']} ({m['keep_share']:.0%} of cart abandoners) | {m['discounts_sent']:,} | €{m['wasted_eur']:,.0f} "
              f"(€{lo:,.0f}–€{hi:,.0f}) | {m['spared_discount_came_back']:,} | {m['left_without_nudge']:,} ({l_lo:,.0f}–{l_hi:,.0f}) |")
md += ["", "## What logistic regression learned (words used 200+ times)", "",
       "Raise the come-back chance most: " + ", ".join(f"`{w}` ({c:+.2f})" for w, c in coef.tail(6)[::-1].items()),
       "", "Lower it most: " + ", ".join(f"`{w}` ({c:+.2f})" for w, c in coef.head(6).items())]
Path("results/baselines.md").write_text("\n".join(md) + "\n")
print(f"done in {time.time() - t0:.0f} s")
