"""Checks for the helpers 05_evaluation adds to nudge/evaluate.py. Run from the repo root:
    .venv/bin/python tests/test_evaluate.py      (or: .venv/bin/python -m pytest tests/test_evaluate.py -q)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nudge.evaluate import bootstrap, industry_rule_score, quality, top_share_rate  # noqa: E402


def test_industry_rule_score_is_deterministic():
    v = pd.DataFrame({"cart_value": [0.0, 120.0, 0.0, 30.0], "label": [0, 1, 1, 0]})
    s = industry_rule_score(v)
    assert s.tolist() == [0.0, 1.0, 0.0, 1.0]
    # ties count as half a win: pairs (pos, neg) = (1,0)->win, (1,3)->tie, (2,0)->tie, (2,3)->loss -> (1+.5+.5)/4
    assert abs(quality(v["label"].to_numpy(), s, 0.5)["auc"] - 0.5) < 1e-12
    assert np.array_equal(s, industry_rule_score(v))


def test_top_share_rate():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    score = np.array([0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.95])
    assert top_share_rate(y, score, 0.2) == 1.0     # the two highest scores are both positives
    assert top_share_rate(y, -score, 0.2) == 0.0    # negative control: reversed ranking finds none
    assert abs(top_share_rate(y, np.zeros(10), 0.2) - 0.2) < 1e-12   # all tied -> the overall rate, not "first rows"


def test_top_share_rate_ties_do_not_depend_on_row_order():
    # 4 visits with score 1 (1 positive), 6 with score 0 (3 positives); top 20% = 2 places inside the tied group of 4
    y = np.array([1, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    score = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    assert abs(top_share_rate(y, score, 0.2) - 0.25) < 1e-12          # 2 places x 1/4 come-back rate / 2
    assert abs(top_share_rate(y, score, 0.6) - 2 / 6) < 1e-12         # 4 above (1 positive) + 2 places x 3/6 = 2 of 6
    rng = np.random.default_rng(1)
    for _ in range(5):                                                # same answer for every row order
        order = rng.permutation(10)
        assert abs(top_share_rate(y[order], score[order], 0.2) - 0.25) < 1e-12


def test_paired_bootstrap_uses_same_draw():
    # Two identical score columns -> their difference must be exactly 0 in every draw.
    rng = np.random.default_rng(0)
    v = pd.DataFrame({"user_id": np.repeat(np.arange(50), 2), "label": rng.integers(0, 2, 100)})
    v["_a"] = rng.uniform(size=100)
    v["_b"] = v["_a"]
    diff = lambda d: {"a-b": quality(d["label"].to_numpy(), d["_a"].to_numpy(), .5)["auc"]  # noqa: E731
                      - quality(d["label"].to_numpy(), d["_b"].to_numpy(), .5)["auc"]}
    lo, hi = bootstrap(v, diff, n=20, seed=0)["a-b"]
    assert lo == 0.0 and hi == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all evaluate helper tests passed")
