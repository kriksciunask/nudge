"""Download one openly licensed photo per product from Openverse (no account needed).

Reads app/shop/catalog.csv (photo_query column), saves app/shop/images/<id>.jpg (square, 600 px)
and app/shop/photo_credits.csv (creator, licence, source) for attribution.
Run from the repo root:  .venv/bin/python scripts/fetch_product_photos.py
"""
import csv
import io
import json
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

import certifi
import numpy as np
from PIL import Image

HEADERS = {"User-Agent": "student-demo-shop/1.0"}
SSL = ssl.create_default_context(cafile=certifi.where())  # python.org builds ship without CA certificates
IMAGES = Path("app/shop/images")
IMAGES.mkdir(parents=True, exist_ok=True)


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30, context=SSL) as r:
        return r.read()


def square(data):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    side = min(img.size)
    left, top = (img.width - side) // 2, (img.height - side) // 2
    return img.crop((left, top, left + side, top + side)).resize((600, 600), Image.LANCZOS)


def catalogue_score(img):
    """How much a photo looks like a shop catalogue shot: a bright, even background around the edges."""
    g = np.asarray(img.convert("L"), dtype=float)
    k = g.shape[0] // 12
    border = np.concatenate([g[:k].ravel(), g[-k:].ravel(), g[:, :k].ravel(), g[:, -k:].ravel()])
    rgb = np.asarray(img.convert("HSV"), dtype=float)
    colour = rgb[..., 1].mean() / 255                     # 0 = black-and-white / sepia, higher = real colour photo
    return border.mean() - 0.8 * border.std() - (80 if colour < 0.10 else 0)


credits = []
rows = list(csv.DictReader(open("app/shop/catalog.csv")))
for row in rows:
    out = IMAGES / f"{row['id']}.jpg"
    if out.exists():
        continue
    q = urllib.parse.quote(row["photo_query"])
    found = None
    for attempt in range(3):   # Openverse sometimes answers 504 when busy: wait and try again
        try:
            found = json.loads(get(f"https://api.openverse.org/v1/images/?q={q}&category=photograph&page_size=16"))["results"]
            break
        except Exception as e:  # noqa: BLE001
            print("search failed", row["id"], e, f"(try {attempt + 1}/3)", flush=True)
            time.sleep(10 * (attempt + 1))
    if found is None:
        continue
    best = None
    for hit in found:
        if (hit.get("width") or 0) < 400 or (hit.get("height") or 0) < 300:
            continue
        try:
            img = square(get(hit["thumbnail"]))
        except Exception:  # noqa: BLE001
            continue
        score = catalogue_score(img)
        if best is None or score > best[0]:
            best = (score, img, hit)
    if best is None:
        print("NO PHOTO", row["id"], row["photo_query"], flush=True)
    else:
        score, img, hit = best
        img.save(out, quality=85)
        credits.append({"id": row["id"], "title": hit.get("title", ""), "creator": hit.get("creator", ""),
                        "license": f"{hit.get('license', '')} {hit.get('license_version', '')}".strip(),
                        "source": hit.get("foreign_landing_url", ""), "score": round(score, 1)})
        print("ok", row["id"], row["photo_query"], round(score), flush=True)
    time.sleep(1.2)

path = Path("app/shop/photo_credits.csv")
old = list(csv.DictReader(open(path))) if path.exists() else []
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "title", "creator", "license", "source", "score"])
    w.writeheader(); w.writerows(old + credits)
