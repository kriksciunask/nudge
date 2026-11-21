"""Checks for the demo shop (app/app.py) with Streamlit's AppTest, no browser needed.
Run from the repo root:  .venv/bin/python tests/test_app.py"""
import re
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
APP = str(ROOT / "app" / "app.py")


def panel(at) -> str:
    """The live meter's HTML (the Nudge panel markdown, not the CSS block)."""
    return next(m.value for m in at.markdown if 'class="nm-hero"' in m.value)


def chance_shown(at) -> int:
    return int(re.search(r'class="nm-hero">(\d+)<span class="pc">', panel(at)).group(1))


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
    assert "Browse or add something to cart to see it move" in html and 'class="nm-val' in html
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
    assert "Nudge would send</small><b><i></i>No message</b>" in html and "New visitor, nothing clicked yet." in html
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


def leave_markdown(at) -> str:
    return " ".join(m.value for m in at.markdown if "<style>" not in m.value)


def test_leave_page_look_inside():
    at = fresh()
    next(b for b in at.button if b.key and b.key.startswith("gv_")).click().run()
    at.button(key="detail_cart").click().run()
    next(b for b in at.button if b.label == "Leave the shop").click().run()
    assert not at.exception, at.exception
    text = leave_markdown(at)
    assert "Look inside the model" in text and text.count("Looked at most") == 2   # tag beside the word + title on the short phone tag
    rows = re.findall(r'class="bz-in-row[^"]*"', text)
    assert len(rows) == len(words_on_leave(at))                       # one row per word the model read
    widths = [float(x) for x in re.findall(r'class="fill" style="width:([\d.]+)%', text)]
    assert max(widths) == 100.0 and "softmax(QK" in text and "Effect of this click" in text and "Chance now" in text
    assert any("Full attention grid" in e.label for e in at.expander)


def words_on_leave(at) -> list[str]:
    return re.findall(r'class="bz-chip[^"]*" title="([^"]+)"', next(m.value for m in at.markdown if 'class="bz-seen"' in m.value))


def test_look_inside_long_visit_is_capped():
    at = fresh()
    for _ in range(16):                                     # 16 views = 17 words with the start word
        next(b for b in at.button if b.key and b.key.startswith("gv_")).click().run()
        next(b for b in at.button if b.label.startswith("Back to all")).click().run()
    next(b for b in at.button if b.label == "Leave the shop").click().run()
    assert not at.exception, at.exception
    text = leave_markdown(at)
    assert len(re.findall(r'class="bz-in-row', text)) == 16 and "1 earlier word not shown" in text


def owner_html_of(at) -> str:
    return next(m.value for m in at.markdown if 'class="ow-hero"' in m.value)


def test_owner_view_shows_money_and_follows_slider():
    import pandas as pd
    curve = pd.read_csv(ROOT / "results" / "money_curve.csv").set_index("g")
    # The curve's industry-rule row ties to the official evaluation numbers
    assert round(curve["rule_wasted_eur"].iloc[0]) == 119729 and curve["rule_left_without_nudge"].iloc[0] == 4080
    at = fresh()
    next(b for b in at.button if b.label == "Leave the shop").click().run()
    assert not at.exception, at.exception
    for g in (50, 100):
        at.sidebar.slider[0].set_value(g).run()
        assert not at.exception, at.exception
        html = owner_html_of(at)
        assert f'{curve.loc[g / 100, "wasted_eur"]:,.0f}' in html and "119,729" in html and "A/B test" in html
        assert at.slider[0].value == g          # the owner slider moves with the presenter slider


def test_owner_slider_moves_the_decision_cutoffs():
    at = fresh()
    next(b for b in at.button if b.label == "Leave the shop").click().run()
    at.slider[0].set_value(0).run()
    assert not at.exception, at.exception
    assert at.sidebar.slider[0].value == 0 and "Under 8%" in owner_html_of(at)


def meter_cutoffs(at) -> tuple[str, str]:
    """The two cut-off ticks of the live meter chart (bold ticks), e.g. ("6%", "21%")."""
    html = next(m.value for m in at.markdown if 'class="nm-chart' in m.value) if any('class="nm-chart' in m.value for m in at.markdown) else ""
    return tuple(re.findall(r'class="cut"[^>]*>(\d+%)</span>', html))


def test_two_sliders_are_one_state_both_ways():
    from nudge.decision import cutoffs
    at = fresh()
    next(b for b in at.button if b.label == "Leave the shop").click().run()
    assert not at.exception, at.exception
    # presenter slider -> owner card and cut-offs
    at.sidebar.slider[0].set_value(20).run()
    low, high = cutoffs(0.2)
    assert at.slider[0].value == 20 and f"Under {low:.0%}" in owner_html_of(at) and f"{high:.0%}" in owner_html_of(at)
    assert meter_cutoffs(at) == (f"{low:.0%}", f"{high:.0%}")
    # owner slider -> presenter slider, card and cut-offs
    at.slider[0].set_value(90).run()
    low, high = cutoffs(0.9)
    assert at.sidebar.slider[0].value == 90 and f"Under {low:.0%}" in owner_html_of(at)
    assert meter_cutoffs(at) == (f"{low:.0%}", f"{high:.0%}")
    # and back once more, to be sure nothing sticks
    at.sidebar.slider[0].set_value(50).run()
    assert at.slider[0].value == 50 and at.sidebar.slider[0].value == 50


def test_owner_view_names_the_cost_too():
    at = fresh()
    next(b for b in at.button if b.label == "Leave the shop").click().run()
    html = owner_html_of(at)
    assert "less wasted" in html and "% left out" in html and "more shoppers got no nudge and did not come back" in html and "A/B test" in html


def board_html(at) -> str:
    return next(m.value for m in at.markdown if 'class="bd"' in m.value)


def board_app(key="k", **params) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.secrets["board_key"] = "k"
    at.query_params["board"] = "1"
    if key:
        at.query_params["key"] = key
    for name, value in params.items():
        at.query_params[name] = value
    return at.run()


def test_board_shows_fake_shoppers_and_hero():
    import live_board
    live_board.get_store().reset()
    at = board_app(demo="12")
    assert not at.exception, at.exception
    html = board_html(at)
    assert '<span class="num">12</span>' in html and html.count('class="bd-card') == 12
    assert "Just left" in html and "Left ·" in html and "Reminder + 10%" in html
    assert any(b.key == "bd_reset" for b in at.button)       # presenter menu only with the key


def test_board_without_key_has_no_menu_and_no_backdoor():
    import live_board
    live_board.get_store().reset()
    at = board_app(key="", demo="12")                                      # demo is ignored without the key
    assert not at.exception, at.exception
    assert "Scan to join" in board_html(at) and 'class="bd-card' not in board_html(at)
    assert not any(b.key == "bd_reset" for b in at.button)
    at = board_app(key="wrong", demo="12")
    assert 'class="bd-card' not in board_html(at)


def test_shopper_click_appears_on_board_with_nickname():
    import live_board
    live_board.get_store().reset()
    at = fresh()
    at.text_input(key="nick_in").set_value("Ona").run()
    next(b for b in at.button if b.key and b.key.startswith("gv_")).click().run()
    rows = live_board.get_store().snapshot()
    assert len(rows) == 1 and rows[0]["nick"] == "Ona" and len(rows[0]["hist"]) == 2
    next(b for b in at.button if b.label == "Leave the shop").click().run()
    assert live_board.get_store().snapshot()[0]["status"] == "left"
    live_board.get_store().reset()


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"all {len(tests)} app checks passed")
