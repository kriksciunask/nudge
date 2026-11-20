"""Summary tables for the Nudge Data Explorer page.

Reads data/sample/events_sample.parquet (October 2019, 1 in 150 users) and writes one JSON file
per table, as {"rows": [...], "meta": [...]}, into the folder given as the first argument.
Run from the repo root:  .venv/bin/python scripts/make_explorer_data.py <out_dir>
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nudge.contract import build_sessions  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
CODE = Path(__file__).read_text()
INPUT = "data/sample/events_sample.parquet (Oct 2019, users with user_id % 150 == 0)"

# Whole-month click counts, printed by the Kaggle notebook kriksciunas/nudge-01-data (all of 2019-Oct.csv).
MONTH_COUNTS = {"view": 40_779_399, "cart": 926_516, "purchase": 742_849}


def save(name, df):
    rows = json.loads(df.to_json(orient="records"))
    meta = [{"label": "make_explorer_data.py", "value": CODE, "type": "code", "lang": "python"},
            {"label": "Input", "value": INPUT}]
    (OUT / f"{name}.json").write_text(json.dumps({"rows": rows, "meta": meta}))


ev = pd.read_parquet("data/sample/events_sample.parquet")
ev["time"] = pd.to_datetime(ev["event_time"].str.replace(" UTC", "", regex=False))
s = build_sessions(ev)
s["start_time"] = pd.to_datetime(s["start_time"])
has = lambda f: s["tokens"].apply(lambda t: any(f(w) for w in t))  # noqa: E731
cart = s["cart_value"] > 0

# --- overview: one row of headline numbers -----------------------------------------------------
save("overview", pd.DataFrame([{
    "sample_clicks": len(ev), "sessions": len(s), "users": int(ev["user_id"].nunique()),
    "buy_pct": round(100 * s["label"].mean(), 1),
    "cart_pct": round(100 * cart.mean(), 1),
    "cart_buy_pct": round(100 * s.loc[cart, "label"].mean(), 1),
    "one_click_pct": round(100 * (s["n_clicks"] == 1).mean(), 1),
    "median_clicks": float(s["n_clicks"].median()),
    "p90_clicks": float(s["n_clicks"].quantile(0.9)),
    "max_clicks": int(s["n_clicks"].max()),
    "first_day": ev["time"].min().strftime("%-d %b %Y"), "last_day": ev["time"].max().strftime("%-d %b %Y"),
}]))

# --- columns: what each raw column is, and how much is missing ------------------------------------
about = {
    "event_time": ("When the click happened (UTC)", "Order of clicks; split by date"),
    "event_type": ("view, cart or purchase", "The click word; purchase becomes the label"),
    "product_id": ("Which product", "'again' = same product seen before"),
    "category_code": ("Category path, e.g. electronics.smartphone", "Maybe in the click word"),
    "brand": ("Brand name", "Failure analysis only"),
    "price": ("Price in euros", "Price band; cart value"),
    "user_id": ("Which shopper", "Split train/test by user"),
    "user_session": ("Which visit", "Groups clicks into sessions"),
}
cols = []
for c, (meaning, use) in about.items():
    nn = ev[c].dropna()
    cols.append({"column": c, "meaning": meaning, "missing_pct": round(100 * ev[c].isna().mean(), 1),
                 "distinct": int(ev[c].nunique()), "example": str(nn.iloc[len(nn) // 2]), "use": use})
save("columns", pd.DataFrame(cols))

save("month_events", pd.DataFrame([{"event_type": k, "clicks": v} for k, v in MONTH_COUNTS.items()]))

# --- signals: buy rate of sessions with / without a pattern ---------------------------------------
first_session = s.sort_values("start_time").groupby("user_id")["user_session"].transform("first").reindex(s.index)
returning = s["user_session"] != first_session
sig = [
    ("Added something to the cart", cart),
    ("Viewed the same product again", has(lambda w: "_again" in w)),
    ("Returning shopper (earlier visit this month)", returning),
    ("5 or more clicks", s["n_clicks"] >= 5),
    ("Had a long pause (70 s to 30 min)", has(lambda w: w.endswith("_long"))),
    ("Left and came back (30+ min away)", has(lambda w: w.endswith("_away"))),
    ("Looked at an expensive item (€250+)", has(lambda w: "_expensive" in w)),
    ("Only cheap items (under €100)", s["tokens"].apply(lambda t: all("_cheap" in w for w in t))),
    ("Never repeated a product", ~has(lambda w: "_again" in w)),
    ("Only one click", s["n_clicks"] == 1),
]
save("signals", pd.DataFrame([{"signal": n, "sessions_pct": round(100 * m.mean(), 1),
                                "buy_pct": round(100 * s.loc[m, "label"].mean(), 1)} for n, m in sig]))

# --- session length -------------------------------------------------------------------------------
bins = [0, 1, 2, 3, 5, 10, 20, 10_000]
names = ["1", "2", "3", "4–5", "6–10", "11–20", "21+"]
s["len_bucket"] = pd.cut(s["n_clicks"], bins=bins, labels=names)
g = s.groupby("len_bucket", observed=False)["label"].agg(["size", "mean"]).reset_index()
save("length", pd.DataFrame({"clicks": g["len_bucket"].astype(str), "sessions": g["size"],
                             "buy_pct": (100 * g["mean"]).round(1)}))

# --- time: hour of day (UTC), weekday, day of month -------------------------------------------------
g = s.groupby(s["start_time"].dt.hour)["label"].agg(["size", "mean"]).reset_index()
save("hour", pd.DataFrame({"hour": g["start_time"], "sessions": g["size"], "buy_pct": (100 * g["mean"]).round(1)}))
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
g = s.groupby(s["start_time"].dt.day_of_week)["label"].agg(["size", "mean"]).reset_index()
save("weekday", pd.DataFrame({"day": [days[d] for d in g["start_time"]], "sessions": g["size"],
                              "buy_pct": (100 * g["mean"]).round(1)}))
g = s.groupby(s["start_time"].dt.strftime("%Y-%m-%d"))["label"].agg(["size", "mean"]).reset_index()
save("daily", pd.DataFrame({"date": g["start_time"], "sessions": g["size"], "buy_pct": (100 * g["mean"]).round(1)}))

# --- what they look at: categories and prices (non-purchase clicks) --------------------------------
clicks = ev[ev["event_type"] != "purchase"]
top = clicks["category_code"].str.split(".").str[0].fillna("(missing)")
g = top.value_counts().reset_index()
g.columns = ["category", "clicks"]
g["pct"] = (100 * g["clicks"] / g["clicks"].sum()).round(1)
save("categories", g.head(10))
pb = [-0.01, 25, 50, 100, 250, 500, 1000, 1e9]
pn = ["under €25", "€25–50", "€50–100", "€100–250", "€250–500", "€500–1000", "€1000+"]
band = {"under €25": "cheap", "€25–50": "cheap", "€50–100": "cheap", "€100–250": "mid"}
g = pd.cut(clicks["price"], bins=pb, labels=pn, right=False).value_counts(sort=False).reset_index()
g.columns = ["price", "clicks"]
g["price"] = g["price"].astype(str)
g["band"] = g["price"].map(band).fillna("expensive")
save("prices", g)

# --- click words: how often, and buy rate of sessions that contain each ----------------------------
ex = s[["tokens", "label"]].explode("tokens")
cnt = ex["tokens"].value_counts()
per = ex.reset_index().drop_duplicates(["index", "tokens"]).groupby("tokens")["label"].agg(["size", "mean"])
w = pd.DataFrame({"word": cnt.index, "clicks": cnt.values})
w["sessions"] = w["word"].map(per["size"]).astype(int)
w["buy_pct"] = w["word"].map(100 * per["mean"]).round(1)
save("words", w)

# --- example sessions: real ones, picked at random (fixed seed) ------------------------------------
mid = s[(s["n_clicks"] >= 3) & (s["n_clicks"] <= 9)]
pick = pd.concat([mid[mid["label"] == 1].sample(4, random_state=7), mid[mid["label"] == 0].sample(4, random_state=7),
                  s[s["n_clicks"] == 1].sample(1, random_state=7)])
save("examples", pd.DataFrame({
    "session": pick["user_session"].str[:8], "outcome": pick["label"].map({1: "bought", 0: "left"}),
    "clicks": pick["n_clicks"], "cart_value": pick["cart_value"].round(0),
    "words": pick["tokens"].apply(" ".join)}))

print("wrote", sorted(p.name for p in OUT.glob("*.json")))
