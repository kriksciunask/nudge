"""Live audience board: the shared, in-memory state behind the presenter page.

Every shopper's phone has its own Streamlit session. After each click the session writes one small record
here (an auto nickname, the chance, the last ~20 chances, the action Nudge would take, the cart value, and
whether they pressed Leave). The presenter page reads a snapshot every ~2 s.

Pure Python (no Streamlit), so it can be tested with plain threads. One instance per server process
(app/live_board.py keeps it in st.cache_resource). No personal data: no IP, no browser id, only what the
shopper typed as a nickname (filtered) and their click chances.
"""

import re
import threading
import time

MAX_SESSIONS = 200          # the store never holds more than this; the stalest session is dropped
IDLE_SECONDS = 15 * 60      # a session idle this long leaves the board
HIST_KEEP = 20              # chance values kept per shopper (sparkline)
NICK_MAX = 16

# Zone numbers match the live meter (app.py ZONE_OF): 0 no message, 1 reminder + 10%, 2 reminder only.
NOTHING_Z, DISCOUNT_Z, REMINDER_Z = 0, 1, 2

# Deliberately short: obvious slurs and sexual or abusive words, matched after removing punctuation and
# common letter swaps. The presenter can also hide a row, so this is a first filter, not the only one.
_BLOCKED = (
    "fuck", "shit", "bitch", "cunt", "dick", "cock", "pussy", "penis", "vagina", "nigg", "fagg", "retard",
    "whore", "slut", "rape", "nazi", "hitler", "porn", "sex", "asshole", "bastard", "idiot", "stupid",
    "suicide", "pydar", "bybis", "kurva", "xxx",
)
_SWAPS = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "$": "s", "@": "a", "!": "i"})


def clean_nickname(text: str | None) -> str | None:
    """A safe nickname (<= 16 chars), or None when empty or not acceptable (then the auto name is used)."""
    if not text:
        return None
    nick = re.sub(r"[^\w .'\-]", "", str(text), flags=re.UNICODE)   # letters, digits, space, . ' -
    nick = re.sub(r"\s+", " ", nick).strip()[:NICK_MAX].strip()
    if len(nick) < 2:
        return None
    flat = re.sub(r"[^a-z]", "", nick.lower().translate(_SWAPS))
    if any(bad in flat for bad in _BLOCKED):
        return None
    return nick


class LiveBoard:
    """Thread-safe dict of live shopper sessions. The lock is held only while copying or updating a record."""

    def __init__(self, max_sessions: int = MAX_SESSIONS, idle_seconds: float = IDLE_SECONDS, clock=time.time):
        self.max_sessions = max_sessions
        self.idle_seconds = idle_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._rows: dict[str, dict] = {}
        self._counter = 0

    # -- writing (shopper sessions) ---------------------------------------------------------------
    def update(self, sid: str, *, hist: list[float], zone: int, cart_value: float = 0.0, n_cart: int = 0,
               status: str = "shopping", words: list[str] | None = None, nick: str | None = None) -> None:
        """Record the shopper's latest state. status: "shopping", "left" (pressed Leave) or "bought"."""
        now = self._clock()
        nick = clean_nickname(nick)
        with self._lock:
            self._expire(now)
            row = self._rows.get(sid)
            if row is None:
                if len(self._rows) >= self.max_sessions:      # make room: drop the stalest session
                    stale = min(self._rows, key=lambda k: self._rows[k]["last_seen"])
                    del self._rows[stale]
                self._counter += 1
                row = self._rows[sid] = {"sid": sid, "n": self._counter, "nick": f"Shopper {self._counter}",
                                         "custom_nick": False, "hist": [], "chance": 0.0, "hidden": False, "left_at": None,
                                         "status": "shopping", "demo": False}
            if nick:
                row["nick"], row["custom_nick"] = nick, True
            if status == "left" and row["status"] != "left":
                row["left_at"] = now                         # the board highlights this card for a few seconds
            elif status != "left":
                row["left_at"] = None
            row.update(hist=[float(h) for h in hist[-HIST_KEEP:]], chance=float(hist[-1]), zone=int(zone), cart_value=float(cart_value), n_cart=int(n_cart),
                       status=status, words=list(words or [])[-6:], last_seen=now)

    # -- reading (the board) ------------------------------------------------------------------------
    def snapshot(self, include_hidden: bool = False) -> list[dict]:
        """Copies of the live rows, most recent activity first."""
        now = self._clock()
        with self._lock:
            self._expire(now)
            rows = [dict(r, hist=list(r["hist"])) for r in self._rows.values() if include_hidden or not r["hidden"]]
        rows.sort(key=lambda r: r["last_seen"], reverse=True)
        return rows

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)

    # -- presenter actions ---------------------------------------------------------------------------
    def hide(self, sid_or_nick: str) -> int:
        """Hide rows whose session id or nickname matches (case-insensitive). Returns how many were hidden."""
        key = sid_or_nick.strip().lower()
        hit = 0
        with self._lock:
            for r in self._rows.values():
                if key and key in (r["sid"].lower(), r["nick"].lower()):
                    r["hidden"] = True
                    hit += 1
        return hit

    def reset(self) -> None:
        with self._lock:
            self._rows.clear()

    # -- internals -----------------------------------------------------------------------------------
    def _expire(self, now: float) -> None:     # caller holds the lock
        cutoff = now - self.idle_seconds
        for sid in [k for k, r in self._rows.items() if r["last_seen"] < cutoff]:
            del self._rows[sid]


def summary(rows: list[dict]) -> dict:
    """Counts for the board. "none/discount/reminder" = what Nudge would do for those still shopping;
    "left_*" = final decisions for those who pressed Leave; bought shoppers get no message and are not scored."""
    shopping = [r for r in rows if r["status"] == "shopping"]
    left = [r for r in rows if r["status"] == "left"]
    carts = [r for r in left if r["n_cart"]]
    return {"shoppers": len(rows), "shopping": len(shopping), "left": len(left),
            "bought": sum(r["status"] == "bought" for r in rows),
            "none": sum(r["zone"] == NOTHING_Z for r in shopping),
            "discount": sum(r["zone"] == DISCOUNT_Z for r in shopping),
            "reminder": sum(r["zone"] == REMINDER_Z for r in shopping),
            "left_none": sum(r["zone"] == NOTHING_Z for r in left),
            "left_discount": sum(r["zone"] == DISCOUNT_Z for r in left),
            "left_reminder": sum(r["zone"] == REMINDER_Z for r in left),
            "left_carts": len(carts),                                          # "discount everyone" would discount these
            "left_carts_discounted": sum(r["zone"] == DISCOUNT_Z for r in carts),   # what Nudge gave
            # cart value of the leavers Nudge did NOT discount: 10% of it is what it holds back vs "discount everyone"
            "left_held_value": sum(r["cart_value"] for r in carts if r["zone"] != DISCOUNT_Z)}


def fill_demo(board: LiveBoard, n: int = 12) -> None:
    """Fake shoppers in mixed states, for rehearsal and screenshots only (the page guards this with the key)."""
    import random
    rng = random.Random(7)
    names = ["Ona", "Tomas", "Mantas", "Rasa", "Eglė", "Jonas", "Gabija", "Lukas", "Austėja", "Dovydas", "Ieva",
             "Rokas", "Aistė", "Paulius", "Milda", "Karolis", "Neda", "Justas"]
    for i in range(n):
        sid = f"demo-{i}"
        steps = rng.randint(1, 9)
        p, hist = rng.uniform(0.03, 0.12), []
        for _ in range(steps):
            p = min(0.45, max(0.02, p + rng.uniform(-0.04, 0.07)))
            hist.append(p)
        zone = 0 if p < 0.065 else (1 if p < 0.215 else 2)
        status = "left" if i % 4 == 1 else "shopping"
        value = rng.choice([0, 0, 39, 85, 120, 240])
        board.update(sid, hist=hist, zone=zone, cart_value=value, n_cart=0 if not value else rng.choice([1, 2]), status="shopping", nick=names[i % len(names)] if i % 3 else None)
        with board._lock:
            row = board._rows[sid]
            row["demo"] = True
            if status == "left":
                row["status"], row["left_at"] = "left", board._clock() - (2 if i == 1 else 120)
