"""One way to score every model (random, industry rule, logistic regression, RNN, Transformer).

Prediction quality (does the score separate shoppers who come back and buy from those who don't?):
    AUC        chance a random come-back shopper gets a higher score than a random non-buyer (0.5 = coin flip)
    PR-AUC     average precision; better than AUC when positives are rare (here ~9%). Random = the positive rate.
    precision / recall / F1 at one cut-off, chosen on the VALID split (max F1), never on test.

Money (cart abandoners only, the people today's flows discount):
    Industry rule: every cart abandoner gets 10% off.
    A model: a cart abandoner gets 10% off only if the model thinks they will NOT come back on their own.
    WASTED = 10% of what a shopper actually bought, for shoppers who got the discount and came back anyway.
    LEFT WITHOUT A NUDGE = cart abandoners the model skipped who did not come back.
    Every policy: at most one discount per shopper per 7 days (a shopper uses one code on one order).

Uncertainty: every number gets a 95% range from resampling SHOPPERS (not visits) 200 times.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

DISCOUNT = 0.10
WINDOW_DAYS = 7


def pick_cutoff(y_valid: np.ndarray, score_valid: np.ndarray) -> float:
    """Cut-off with the best F1 on the validation split."""
    qs = np.unique(np.quantile(score_valid, np.linspace(0.5, 0.995, 200)))
    best, best_f1 = qs[0], -1.0
    for q in qs:
        pred = score_valid >= q
        tp = np.sum(pred & (y_valid == 1))
        if pred.sum() == 0 or tp == 0:
            continue
        p, r = tp / pred.sum(), tp / y_valid.sum()
        f1 = 2 * p * r / (p + r)
        if f1 > best_f1:
            best, best_f1 = q, f1
    return float(best)


def quality(y: np.ndarray, score: np.ndarray, cutoff: float) -> dict:
    pred = score >= cutoff
    tp, fp = np.sum(pred & (y == 1)), np.sum(pred & (y == 0))
    fn = np.sum(~pred & (y == 1))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "auc": roc_auc_score(y, score) if len(np.unique(y)) > 1 else np.nan,
        "pr_auc": average_precision_score(y, score),
        "precision": precision, "recall": recall, "f1": f1,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(np.sum(~pred & (y == 0))),
    }


def one_per_week(visits: pd.DataFrame, give: np.ndarray) -> np.ndarray:
    """Keep at most one discount per shopper per WINDOW_DAYS (the earliest one wins)."""
    give = give.copy()
    order = np.lexsort((visits["decide_time"].to_numpy(), visits["user_id"].to_numpy()))
    users, times = visits["user_id"].to_numpy()[order], visits["decide_time"].to_numpy()[order]
    last_user, last_time = None, None
    week = np.timedelta64(WINDOW_DAYS, "D")
    for k in range(len(order)):
        i = order[k]
        if not give[i]:
            continue
        if users[k] == last_user and times[k] - last_time < week:
            give[i] = False
        else:
            last_user, last_time = users[k], times[k]
    return give


def money(visits: pd.DataFrame, give: np.ndarray) -> dict:
    """Money for one discount policy, on cart abandoners. `give` = True where the policy sends 10% off."""
    cart = (visits["cart_value"] > 0).to_numpy()
    give = one_per_week(visits, give & cart)
    y = visits["label"].to_numpy()
    order_value = visits["next_order_value"].fillna(0).to_numpy()
    wasted = DISCOUNT * order_value[give & (y == 1)].sum()
    return {
        "cart_abandoners": int(cart.sum()),
        "discounts_sent": int(give.sum()),
        "wasted_eur": float(wasted),
        "came_back_anyway_with_discount": int(np.sum(give & (y == 1))),
        "spared_discount_came_back": int(np.sum(cart & ~give & (y == 1))),
        "left_without_nudge": int(np.sum(cart & ~give & (y == 0))),
    }


def bootstrap(visits: pd.DataFrame, fn, n: int = 200, seed: int = 0) -> dict:
    """95% range of every number fn(visits_subset) returns, resampling shoppers with replacement."""
    rng = np.random.default_rng(seed)
    codes, uniques = pd.factorize(visits["user_id"])
    rows_of = pd.Series(np.arange(len(visits))).groupby(codes).apply(np.array).to_numpy()
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(uniques), len(uniques))
        parts = rows_of[pick]
        sub = visits.iloc[np.concatenate(parts)].reset_index(drop=True)
        # A shopper drawn twice counts as two different shoppers (else one-per-week would merge the copies)
        sub["user_id"] = np.repeat(np.arange(len(parts)), [len(x) for x in parts])
        draws.append(fn(sub))
    out = {}
    for key in draws[0]:
        vals = np.array([d[key] for d in draws], dtype=float)
        out[key] = (float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)))
    return out


def industry_rule_score(visits: pd.DataFrame) -> np.ndarray:
    """Today's rule as a score: 1.0 if the visit left a cart, else 0.0. No random tie-break noise:
    the many tied visits are handled by sklearn's AUC / PR-AUC exactly (a tie counts as half a win),
    so the rule's scores are the same on every run and every computer."""
    return (visits["cart_value"].to_numpy() > 0).astype(np.float64)


def top_share_rate(y: np.ndarray, score: np.ndarray, share: float = 0.10) -> float:
    """Come-back rate among the `share` of visits with the highest score.
    If the line falls inside a group of tied scores, that group counts in proportion: the places left
    are filled at the group's own come-back rate. So the answer never depends on row order."""
    y, score = np.asarray(y, dtype=float), np.asarray(score)
    n_top = max(1, int(round(share * len(score))))
    line = np.sort(score)[::-1][n_top - 1]          # score of the last visit inside the top
    above, tied = score > line, score == line
    places_left = n_top - above.sum()               # how many of the tied visits fit in the top
    return float((y[above].sum() + places_left * y[tied].mean()) / n_top)
