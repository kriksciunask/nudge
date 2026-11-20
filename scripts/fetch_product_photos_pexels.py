"""Download one product photo per product from Pexels (free licence, attribution not required).

Needs a free API key in .env (PEXELS_API_KEY=...), which git ignores.
Reads app/shop/catalog.csv (photo_query column), saves app/shop/images/<id>.jpg (square, 800 px)
and app/shop/photo_credits.csv (photographer, Pexels page) as a courtesy credit.
For each product the 5 most relevant unused results compete; the most catalogue-like one wins
(bright, even background, real colour), with a small bonus for relevance. No photo is used twice.
Run from the repo root:  .venv/bin/python scripts/fetch_product_photos_pexels.py [--all]
  --all  replace every photo (default: only products without a photo)
"""
import csv
import io
import json
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import certifi
import numpy as np
from PIL import Image

KEY = dict(line.strip().split("=", 1) for line in open(".env") if "=" in line)["PEXELS_API_KEY"]
SSL = ssl.create_default_context(cafile=certifi.where())
IMAGES = Path("app/shop/images")
IMAGES.mkdir(parents=True, exist_ok=True)
REPLACE_ALL = "--all" in sys.argv


def get(url, auth=False):
    headers = {"User-Agent": "student-demo-shop/1.0"}
    if auth:
        headers["Authorization"] = KEY
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30, context=SSL) as r:
        return r.read()


def square(data, size):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    side = min(img.size)
    left, top = (img.width - side) // 2, (img.height - side) // 2
    return img.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)


def catalogue_score(img):
    """How much a photo looks like a shop catalogue shot: bright, even edges and real colour."""
    g = np.asarray(img.convert("L"), dtype=float)
    k = g.shape[0] // 12
    border = np.concatenate([g[:k].ravel(), g[-k:].ravel(), g[:, :k].ravel(), g[:, -k:].ravel()])
    colour = np.asarray(img.convert("HSV"), dtype=float)[..., 1].mean() / 255
    return border.mean() - 0.8 * border.std() - (80 if colour < 0.10 else 0)


credits = {}
used = set()   # never use the same photo twice
rows = list(csv.DictReader(open("app/shop/catalog.csv")))
for row in rows:
    out = IMAGES / f"{row['id']}.jpg"
    if out.exists() and not REPLACE_ALL:
        continue
    q = urllib.parse.quote(row["photo_query"])
    try:
        found = json.loads(get(f"https://api.pexels.com/v1/search?query={q}&per_page=12", auth=True))["photos"]
    except Exception as e:  # noqa: BLE001
        print("search failed", row["id"], e, flush=True)
        continue
    # Pexels sorts by relevance: only the 5 most relevant unused photos compete, so the product stays recognisable;
    # among those, the cleanest catalogue-like shot wins, with a small bonus for being more relevant.
    candidates = [ph for ph in found if ph["id"] not in used][:5]
    best = None
    for rank, photo in enumerate(candidates):
        try:
            score = catalogue_score(square(get(photo["src"]["medium"]), 200)) - 12 * rank
        except Exception:  # noqa: BLE001
            continue
        if best is None or score > best[0]:
            best = (score, photo)
    if best is None:
        print("NO PHOTO", row["id"], row["photo_query"], flush=True)
        continue
    score, photo = best
    used.add(photo["id"])
    square(get(photo["src"]["large"]), 800).save(out, quality=86)
    credits[row["id"]] = {"id": row["id"], "title": photo.get("alt", ""), "creator": photo.get("photographer", ""),
                          "license": "Pexels License", "source": photo.get("url", ""), "score": round(score, 1)}
    print("ok", row["id"], row["photo_query"], round(score), flush=True)
    time.sleep(0.3)

path = Path("app/shop/photo_credits.csv")
old = {r["id"]: r for r in csv.DictReader(open(path))} if path.exists() else {}
old.update(credits)
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "title", "creator", "license", "source", "score"])
    w.writeheader(); w.writerows(sorted(old.values(), key=lambda r: r["id"]))
print("done:", len(credits), "photos from Pexels")
