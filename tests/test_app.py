"""Checks for the demo shop (app/app.py) with Streamlit's AppTest, no browser needed.
Run from the repo root:  .venv/bin/python tests/test_app.py"""
import re
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
APP = str(ROOT / "app" / "app.py")


def panel(at) -> str:
    """The live meter's HTML (the Nudge panel markdown, not the CSS block)."""
    return next(m.value for m in at.markdown if 'class="nm-hero"' in m.value)


def chance_shown(at) -> int:
    return int(re.search(r'class="nm-hero">(\d+)<sup>', panel(at)).group(1))


def points(at) -> int:
    """Number of points on the chart = start + one per click."""
    return panel(at).count('class="nm-dot')


def words_shown(at) -> list[str]:
    return re.findall(r'class="bz-chip[^"]*" title="([^"]+)"', panel(at))


def fresh() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    return at


def test_meter_starts_calm():
    at = fresh()
    html = panel(at)
    assert points(at) == 1
    assert "Browse or add something to cart to see it move" in html
    assert "start, new visitor" in html
    assert words_shown(at) == ["new"]


def test_meter_moves_after_clicks():
    at = fresh()
    before = panel(at)
    view = next(b for b in at.button if b.key and b.key.startswith("gv_"))
    view.click().run()                                  # open a product = one view click
    assert not at.exception, at.exception
    assert points(at) == 2 and panel(at) != before
    assert "Browse or add" not in panel(at)
    at.button(key="detail_cart").click().run()          # add it to the cart = one cart click
    assert points(at) == 3
    assert words_shown(at)[-1].startswith("cart_")
    assert re.search(r'class="nm-delta">(▲ \+\d+ pts|▼ −\d+ pts|about the same)<', panel(at))
    rm = next(b for b in at.button if b.key and b.key.startswith("rm_"))
    rm.click().run()                                    # remove: not a model click, the chance stays
    assert points(at) == 3 and "removing isn" in panel(at)


def test_history_controls_change_words_and_chance():
    at = fresh()
    new_chance = chance_shown(at)
    at.sidebar.toggle(key="h_returning").set_value(True).run()
    assert words_shown(at)[:3] == ["returning", "visits_1", "last_1_3d"]
    at.sidebar.segmented_control(key="h_visits").set_value("5+").run()
    at.sidebar.segmented_control(key="h_last").set_value("Today").run()
    at.sidebar.toggle(key="h_bought").set_value(True).run()
    assert not at.exception, at.exception
    assert words_shown(at)[:4] == ["returning", "bought_recently", "visits_5plus", "last_today"]
    assert chance_shown(at) > new_chance + 5           # the strongest signals: the % visibly reacts


def test_panel_copy_and_chips():
    at = fresh()
    html = panel(at)
    assert "If they leave now → no message" in html and "New visitor, nothing clicked yet." in html
    next(b for b in at.button if b.key and b.key.startswith("gv_")).click().run()     # view 1 (product page)
    for _ in range(2):                                                                # views 2-3 from "You might also like"
        next(b for b in at.button if b.key and b.key.startswith("sv_")).click().run()
    at.button(key="detail_cart").click().run()                                       # cart click
    assert not at.exception, at.exception
    html = panel(at)
    assert re.search(r'class="nm-why">Just added (a|an) [a-z-]+ item to cart, ', html)
    assert len(words_shown(at)) == 4 and "+ 1 earlier word<" in html                # 5 words: last 4 + count
    assert html.count('class="nm-cut"') == 2 and html.count('class="cut"') == 2     # cut-off lines with their values


def test_leave_page_uses_same_words_and_chart():
    at = fresh()
    at.sidebar.toggle(key="h_returning").set_value(True).run()
    meter = chance_shown(at)
    next(b for b in at.button if b.label == "Leave the shop").click().run()
    assert not at.exception, at.exception
    page = next(m.value for m in at.markdown if 'class="bz-hero"' in m.value)
    assert f'class="bz-hero">{meter}<sup>' in page      # same probability as the live meter
    assert 'class="nm-chart big"' in page and 'title="returning"' in page
    assert "placeholder" not in " ".join(m.value.lower() for m in at.markdown if "<style>" not in m.value)


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"all {len(tests)} app checks passed")
