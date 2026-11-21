"""Checks for nudge/explain.py. Run from the repo root:  .venv/bin/python tests/test_explain.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nudge.contract import BLOCK_SIZE, predict  # noqa: E402
from nudge.explain import attention_matrix, attention_to_summary, kept, effect_of_each  # noqa: E402

VISIT = ["returning", "visits_5plus", "last_today", "view_cheap_start", "view_expensive_again_quick",
         "cart_expensive_again_quick"]


def test_weights_sum_to_one_and_match_words():
    w = attention_to_summary(VISIT)
    assert len(w) == len(VISIT) and abs(sum(w) - 1) < 1e-5 and min(w) >= 0
    m = attention_matrix(VISIT)
    assert m.shape == (len(VISIT), len(VISIT)) and abs(m.sum(axis=1) - 1).max() < 1e-5
    assert abs(m[0] - w).max() < 1e-9                  # the summary is row 0 of the grid


def test_one_word_visit():
    assert attention_to_summary(["new"]) == [1.0] or abs(attention_to_summary(["new"])[0] - 1) < 1e-6
    assert effect_of_each(["new"]) == [None]


def test_start_and_history_words_cannot_be_removed():
    d = effect_of_each(VISIT)
    assert len(d) == len(VISIT) and d[:3] == [None, None, None]
    assert all(isinstance(x, float) for x in d[3:])


def test_leave_one_out_matches_predict():
    d = effect_of_each(VISIT)
    expected = 100 * (predict(VISIT) - predict(VISIT[:5]))     # chance with the last click minus without it
    assert abs(d[5] - expected) < 1e-9


def test_long_visit_is_cut_like_predict():
    long = ["new"] + ["view_cheap_normal"] * 100
    k = kept(long)
    assert len(k) == BLOCK_SIZE and k[0] == "new"
    assert len(attention_to_summary(long)) == BLOCK_SIZE
    assert attention_matrix(long).shape == (BLOCK_SIZE, BLOCK_SIZE)
    assert len(effect_of_each(long)) == BLOCK_SIZE


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"all {len(tests)} explain checks passed")
