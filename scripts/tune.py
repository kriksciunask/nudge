"""Overnight tuning, judged ONLY on validation shoppers (the test set is never used to choose anything).

Stage 1: every setting below, trained on a 25% sample of the training visits, up to 15 epochs with early
         stopping (patience 2). Score = validation PR-AUC at the best epoch.
Stage 2: the 2 best settings per model, retrained on ALL training visits with 3 seeds (42, 43, 44),
         so we can report the average and the spread. Test scores are saved for the final notebook only.
Every finished run is appended to results/tuning/stage1.csv or stage2.csv at once, so a crash resumes
where it stopped (runs already in the file are skipped).

Run from the repo root (one heavy job; leave it overnight):
    .venv/bin/python -u scripts/tune.py data/raw/visits_25pct.parquet > results/tuning/log.txt 2>&1
"""
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nudge.models import LogisticRegression, RNNClassifier  # noqa: E402
from nudge.train import BLOCK_SIZE, VOCAB_SIZE, make_loader, predict_proba, set_seed, train_model, visits_to_ids, word_counts  # noqa: E402
from nudge.transformer import VisitTransformer  # noqa: E402

torch.set_num_threads(4)
SMOKE = os.environ.get("TUNE_SMOKE") == "1"   # quick end-to-end check: 1 setting per model, 1 seed, 1 epoch
OUT = Path("results/tuning"); OUT.mkdir(parents=True, exist_ok=True)
MODELS_OUT = Path("results/tuned"); MODELS_OUT.mkdir(parents=True, exist_ok=True)

GRID = {
    "logistic_regression": [dict(lr=lr, weight_decay=wd) for lr, wd in itertools.product([0.01, 0.003], [0.0, 1e-5, 1e-4])],
    "rnn": [dict(d=d, hidden=h, dropout=0.2, lr=lr) for d, h, lr in itertools.product([16, 32], [64, 128], [0.001, 0.003])],
    "transformer": ([dict(d=d, heads=2, blocks=b, dropout=dr, lr=lr) for d, b, dr, lr in
                     itertools.product([32, 64], [1, 2, 3], [0.1, 0.2], [0.001])]
                    + [dict(d=32, heads=2, blocks=2, dropout=0.2, lr=0.003), dict(d=64, heads=4, blocks=2, dropout=0.1, lr=0.001)]),
}


def build(model_name, cfg):
    if model_name == "logistic_regression":
        return LogisticRegression(VOCAB_SIZE - 1)
    if model_name == "rnn":
        return RNNClassifier(d=cfg["d"], hidden=cfg["hidden"], dropout=cfg["dropout"])
    return VisitTransformer(VOCAB_SIZE, BLOCK_SIZE, d=cfg["d"], heads=cfg["heads"], blocks=cfg["blocks"], dropout=cfg["dropout"])


if SMOKE:
    GRID = {k: v[:1] for k, v in GRID.items()}


def run(model_name, cfg, seed, train_x, train_y, val_x, val_y):
    set_seed(seed)
    model = build(model_name, cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    t0 = time.time()
    history = train_model(model, make_loader(train_x, train_y, batch_size=512, shuffle=True),
                          make_loader(val_x, val_y, batch_size=4096), nn.BCEWithLogitsLoss(), optimizer,
                          epochs=1 if SMOKE else 15, print_every=1, clip=1.0 if model_name == "rnn" else None, patience=2)
    val_scores = predict_proba(model, make_loader(val_x, val_y, batch_size=4096))
    best = history["best_epoch"] - 1
    return model, {"model": model_name, "config": json.dumps(cfg, sort_keys=True), "seed": seed,
                   "best_epoch": history["best_epoch"], "val_loss": history["val_loss"][best],
                   "val_pr_auc": history["val_pr_auc"][best], "val_auc": roc_auc_score(val_y, val_scores),
                   "seconds": round(time.time() - t0)}


def done(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["model", "config", "seed"])


def main(data_path):
    v = pd.read_parquet(data_path, columns=["tokens", "label", "split"])
    parts = {s: v[v["split"] == s].reset_index(drop=True) for s in ("train", "valid", "test")}
    ids = {s: visits_to_ids(df["tokens"]) for s, df in parts.items()}
    y = {s: df["label"].to_numpy() for s, df in parts.items()}
    counts = {s: word_counts(x) for s, x in ids.items()}
    print({s: len(df) for s, df in parts.items()}, flush=True)
    feats = lambda m, s: counts[s] if m == "logistic_regression" else ids[s]  # noqa: E731

    # ---- Stage 1: 25% sample of training visits ----
    rng = np.random.default_rng(42)
    sample = np.sort(rng.choice(len(y["train"]), size=len(y["train"]) // 4, replace=False))
    s1 = OUT / "stage1.csv"
    for model_name, configs in GRID.items():
        for cfg in configs:
            key = json.dumps(cfg, sort_keys=True)
            prev = done(s1)
            if ((prev["model"] == model_name) & (prev["config"] == key)).any():
                continue
            print(f"\n=== stage 1 | {model_name} | {key}", flush=True)
            _, row = run(model_name, cfg, 42, feats(model_name, "train")[sample], y["train"][sample], feats(model_name, "valid"), y["valid"])
            pd.concat([prev, pd.DataFrame([row])]).to_csv(s1, index=False)
            print(row, flush=True)

    # ---- Stage 2: best 2 settings per model, all training visits, 3 seeds ----
    st1 = pd.read_csv(s1)
    s2 = OUT / "stage2.csv"
    for model_name in GRID:
        top = st1[st1["model"] == model_name].sort_values("val_pr_auc", ascending=False).head(2)["config"]
        for key in top:
            cfg = json.loads(key)
            for seed in ((42,) if SMOKE else (42, 43, 44)):
                prev = done(s2)
                if ((prev["model"] == model_name) & (prev["config"] == key) & (prev["seed"] == seed)).any():
                    continue
                print(f"\n=== stage 2 | {model_name} | {key} | seed {seed}", flush=True)
                model, row = run(model_name, cfg, seed, feats(model_name, "train"), y["train"], feats(model_name, "valid"), y["valid"])
                tag = f"{model_name}_{hashlib.md5(key.encode()).hexdigest()[:8]}_s{seed}"
                np.save(MODELS_OUT / f"{tag}_test.npy", predict_proba(model, make_loader(feats(model_name, "test"), y["test"], batch_size=4096)).astype(np.float32))
                np.save(MODELS_OUT / f"{tag}_val.npy", predict_proba(model, make_loader(feats(model_name, "valid"), y["valid"], batch_size=4096)).astype(np.float32))
                torch.save({"model": model_name, "config": cfg, "seed": seed, "state_dict": model.state_dict()}, MODELS_OUT / f"{tag}.pt")
                row["tag"] = tag
                pd.concat([prev, pd.DataFrame([row])]).to_csv(s2, index=False)
                print(row, flush=True)
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/visits_25pct.parquet")
