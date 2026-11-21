"""Checks for the live audience board store (nudge/live.py). Plain script:  .venv/bin/python tests/test_live.py"""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nudge.live import IDLE_SECONDS, LiveBoard, clean_nickname, fill_demo, summary  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def write(b, sid, p=0.1, zone=1, **kw):
    b.update(sid, hist=[0.05, p], zone=zone, **kw)


def test_50_threads_write_counts():
    b = LiveBoard()

    def shopper(i):
        for k in range(20):                 # 20 clicks each
            write(b, f"s{i}", 0.01 * k, k % 3, n_cart=k % 2, cart_value=10.0 * (k % 2))
    ts = [threading.Thread(target=shopper, args=(i,)) for i in range(50)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    rows = b.snapshot()
    assert len(rows) == 50 and len({r["sid"] for r in rows}) == 50
    assert len({r["nick"] for r in rows}) == 50                       # auto nicknames are unique
    assert summary(rows)["shoppers"] == 50
    sm = summary(rows)
    assert sm["shopping"] == 50 and sm["left"] == 0 and sum(sm[k] for k in ("none", "discount", "reminder")) == 50


def test_idle_expiry():
    c = Clock()
    b = LiveBoard(clock=c)
    write(b, "a")
    c.t += IDLE_SECONDS - 1
    write(b, "b")
    assert len(b.snapshot()) == 2
    c.t += 2                                    # "a" is now idle > 15 min, "b" is not
    assert [r["sid"] for r in b.snapshot()] == ["b"]


def test_cap_drops_stalest():
    c = Clock()
    b = LiveBoard(max_sessions=200, clock=c)
    for i in range(250):
        c.t += 1
        write(b, f"s{i}")
    rows = b.snapshot()
    assert len(rows) == 200 and len(b) == 200
    assert rows[0]["sid"] == "s249" and "s0" not in {r["sid"] for r in rows}


def test_leave_hide_reset_and_nick():
    c = Clock()
    b = LiveBoard(clock=c)
    write(b, "a", nick="Ona")
    write(b, "a", status="left")
    row = b.snapshot()[0]
    assert row["status"] == "left" and row["left_at"] == c.t and row["nick"] == "Ona"
    assert b.hide("ona") == 1 and b.snapshot() == []
    assert len(b.snapshot(include_hidden=True)) == 1
    write(b, "a")                               # a hidden row stays hidden when the shopper clicks again
    assert b.snapshot() == []
    b.reset()
    assert len(b) == 0


def test_nicknames_are_filtered():
    assert clean_nickname("  Ona  ") == "Ona"
    assert clean_nickname("a" * 40) == "a" * 16
    assert clean_nickname("<b>x</b>") == "bxb"          # markup characters are stripped
    assert clean_nickname("F u c k") is None and clean_nickname("sh1t") is None
    assert clean_nickname("") is None and clean_nickname("x") is None


def test_summary_separates_shopping_from_left():
    b = LiveBoard()
    write(b, "a", zone=1)
    write(b, "b", zone=2)
    write(b, "c", zone=1, n_cart=1, cart_value=50.0, status="left")
    write(b, "d", zone=2, n_cart=2, cart_value=80.0, status="left")
    write(b, "e", zone=0, status="left")
    s = summary(b.snapshot())
    assert (s["shoppers"], s["shopping"], s["left"]) == (5, 2, 3)
    assert (s["discount"], s["reminder"], s["none"]) == (1, 1, 0)            # the bar counts only those still shopping
    assert (s["left_discount"], s["left_reminder"], s["left_none"]) == (1, 1, 1)
    assert (s["left_carts"], s["left_carts_discounted"]) == (2, 1)


def test_demo_fill():
    b = LiveBoard()
    fill_demo(b, 12)
    rows = b.snapshot()
    assert len(rows) == 12 and all(r["demo"] for r in rows)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
