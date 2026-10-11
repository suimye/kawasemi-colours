"""Train the SEM-image -> color model.

Two models:
  1. A small CNN (PyTorch, MPS on Apple Silicon) mapping a grayscale SEM image
     to color as (peak_nm, saturation, value). Trained primarily on unlimited
     synthetic data from synth.py, then optionally fine-tuned on
     data/dataset.jsonl if it exists. Saves model/checkpoint.pt.
  2. A tiny interpretable FEATS_V1 -> color linear/ridge regressor implemented
     in pure numpy (no sklearn dependency). Saves model/baseline.npz. This runs
     everywhere (torch-free) and is used as a fallback + sanity check.

Runtime note: with the default settings the CNN trains in a couple of minutes
on CPU/MPS. Device is auto-selected: torch.device("mps") if available else
"cpu".
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from synth import generate_dataset, forward_color
from features import extract_features
from colorutil import (
    VISIBLE_MIN_NM,
    VISIBLE_MAX_NM,
    srgb8_to_peak_sat_val,
    hsv_like_to_rgb,
    rgb01_to_srgb8,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "checkpoint.pt")
BASELINE = os.path.join(HERE, "baseline.npz")
DATASET_JSONL = os.path.normpath(os.path.join(HERE, "..", "data", "dataset.jsonl"))


# ---------------------------------------------------------------------------
# Target encoding: (peak_nm, sat, val) normalized to ~[0,1]
# ---------------------------------------------------------------------------
def encode_color_target(color):
    peak = (color["peak_nm"] - VISIBLE_MIN_NM) / (VISIBLE_MAX_NM - VISIBLE_MIN_NM)
    return np.array([peak, color["saturation"], color["value"]], dtype=np.float32)


def decode_color_target(y):
    peak = float(y[0]) * (VISIBLE_MAX_NM - VISIBLE_MIN_NM) + VISIBLE_MIN_NM
    return peak, float(y[1]), float(y[2])


# ---------------------------------------------------------------------------
# Interpretable numpy baseline: FEATS_V1 -> (peak,sat,val) ridge regression
# ---------------------------------------------------------------------------
def train_baseline(records, feat_size=128):
    X, Y = [], []
    for r in records:
        img = r["_image"]
        vec, _ = extract_features(img)
        X.append(vec)
        Y.append(encode_color_target(r["color"]))
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    # standardize features
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    Xs = (X - mu) / sd
    Xa = np.hstack([Xs, np.ones((len(Xs), 1))])  # bias

    lam = 1.0
    A = Xa.T @ Xa + lam * np.eye(Xa.shape[1])
    W = np.linalg.solve(A, Xa.T @ Y)  # (F+1, 3)

    pred = Xa @ W
    rmse = np.sqrt(((pred - Y) ** 2).mean(0))
    np.savez(BASELINE, W=W, mu=mu, sd=sd)
    return W, mu, sd, rmse


def baseline_predict(vec, W, mu, sd):
    xs = (np.asarray(vec, dtype=np.float64) - mu) / sd
    xa = np.hstack([xs, [1.0]])
    return xa @ W


# ---------------------------------------------------------------------------
# CNN (torch). Imported lazily so this module loads without torch.
# ---------------------------------------------------------------------------
def _build_torch(size):
    import torch
    import torch.nn as nn

    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.MaxPool2d(2),                                        # /2
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),                                        # /4
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),                                        # /8
                nn.AdaptiveAvgPool2d(4),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 3), nn.Sigmoid(),  # (peak,sat,val) in [0,1]
            )

        def forward(self, x):
            return self.head(self.features(x))

    return SmallCNN


def device_select():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _records_to_tensors(records, size, torch):
    import cv2
    imgs, ys = [], []
    for r in records:
        img = r["_image"]
        if img.ndim == 3:                     # defensive: force grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.shape[:2] != (size, size):     # always normalize to size x size
            img = cv2.resize(img, (size, size))
        imgs.append(img.astype(np.float32) / 255.0)
        ys.append(encode_color_target(r["color"]))
    X = torch.tensor(np.stack(imgs)[:, None, :, :], dtype=torch.float32)
    Y = torch.tensor(np.stack(ys), dtype=torch.float32)
    return X, Y


def train_cnn(train_recs, val_recs, size=96, epochs=12, batch=32, lr=2e-3,
              finetune_recs=None):
    import torch
    import torch.nn as nn

    dev = device_select()
    print(f"[cnn] device = {dev}")
    Model = _build_torch(size)
    net = Model().to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[cnn] parameters = {n_params:,}")

    Xtr, Ytr = _records_to_tensors(train_recs, size, torch)
    Xva, Yva = _records_to_tensors(val_recs, size, torch)
    Xtr, Ytr, Xva, Yva = [t.to(dev) for t in (Xtr, Ytr, Xva, Yva)]

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.SmoothL1Loss()

    def run_epochs(X, Y, n_ep, tag):
        for ep in range(n_ep):
            net.train()
            perm = torch.randperm(len(X))
            tot = 0.0
            for i in range(0, len(X), batch):
                idx = perm[i:i + batch]
                opt.zero_grad()
                out = net(X[idx])
                loss = lossf(out, Y[idx])
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
            net.eval()
            with torch.no_grad():
                vpred = net(Xva)
                vloss = lossf(vpred, Yva).item()
            print(f"[cnn:{tag}] epoch {ep+1}/{n_ep} train={tot/len(X):.4f} val={vloss:.4f}")

    run_epochs(Xtr, Ytr, epochs, "synth")

    if finetune_recs:
        print(f"[cnn] fine-tuning on {len(finetune_recs)} real samples")
        Xft, Yft = _records_to_tensors(finetune_recs, size, torch)
        Xft, Yft = Xft.to(dev), Yft.to(dev)
        for g in opt.param_groups:
            g["lr"] = lr * 0.2
        run_epochs(Xft, Yft, max(4, epochs // 2), "finetune")

    # final validation metrics in physical units
    net.eval()
    with torch.no_grad():
        vpred = net(Xva).cpu().numpy()
    vtrue = Yva.cpu().numpy()
    peak_pred = vpred[:, 0] * (VISIBLE_MAX_NM - VISIBLE_MIN_NM) + VISIBLE_MIN_NM
    peak_true = vtrue[:, 0] * (VISIBLE_MAX_NM - VISIBLE_MIN_NM) + VISIBLE_MIN_NM
    peak_mae = float(np.abs(peak_pred - peak_true).mean())
    sat_mae = float(np.abs(vpred[:, 1] - vtrue[:, 1]).mean())
    val_mae = float(np.abs(vpred[:, 2] - vtrue[:, 2]).mean())

    torch.save({
        "state_dict": net.state_dict(),
        "size": size,
        "target": ["peak_norm", "saturation", "value"],
        "visible_range": [VISIBLE_MIN_NM, VISIBLE_MAX_NM],
    }, CKPT)
    print(f"[cnn] saved -> {CKPT}")
    print(f"[cnn] VAL peak MAE={peak_mae:.1f}nm  sat MAE={sat_mae:.3f}  val MAE={val_mae:.3f}")
    return dict(peak_mae_nm=peak_mae, sat_mae=sat_mae, val_mae=val_mae, n_params=n_params)


# ---------------------------------------------------------------------------
# Real dataset loader (data/dataset.jsonl) — optional
# ---------------------------------------------------------------------------
def load_real_records(size=96):
    if not os.path.exists(DATASET_JSONL):
        return []
    import cv2
    base = os.path.dirname(DATASET_JSONL)
    recs = []
    with open(DATASET_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            img_path = obj.get("image_file")
            if img_path and not os.path.isabs(img_path):
                img_path = os.path.join(os.path.dirname(base), img_path) \
                    if not img_path.startswith("images") \
                    else os.path.join(base, img_path)
            if not img_path or not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            color = obj.get("color", {})
            srgb = color.get("srgb")
            peak = color.get("peak_nm")
            if srgb is None:
                continue
            p2, s2, v2 = srgb8_to_peak_sat_val(srgb)
            colrec = dict(
                peak_nm=float(peak) if peak else p2,
                saturation=s2, value=v2, srgb=srgb,
            )
            recs.append({"_image": img, "color": colrec})
    return recs


def main():
    ap = argparse.ArgumentParser(description="Train SEM->color model")
    ap.add_argument("--n", type=int, default=800, help="synthetic samples")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cnn", action="store_true",
                    help="skip torch CNN (baseline only; runs torch-free)")
    args = ap.parse_args()

    print(f"Generating {args.n} synthetic samples...")
    recs = generate_dataset(args.n, size=args.size, seed=args.seed)
    nval = max(20, int(0.15 * len(recs)))
    train_recs, val_recs = recs[nval:], recs[:nval]

    print("Training interpretable FEATS_V1 baseline (numpy ridge)...")
    W, mu, sd, rmse = train_baseline(train_recs)
    # baseline val error
    bx, by = [], []
    for r in val_recs:
        vec, _ = extract_features(r["_image"])
        bx.append(baseline_predict(vec, W, mu, sd))
        by.append(encode_color_target(r["color"]))
    bx, by = np.asarray(bx), np.asarray(by)
    bpeak_mae = float(np.abs((bx[:, 0] - by[:, 0]) * (VISIBLE_MAX_NM - VISIBLE_MIN_NM)).mean())
    print(f"[baseline] saved -> {BASELINE}")
    print(f"[baseline] VAL peak MAE={bpeak_mae:.1f}nm "
          f"sat MAE={np.abs(bx[:,1]-by[:,1]).mean():.3f} "
          f"val MAE={np.abs(bx[:,2]-by[:,2]).mean():.3f}")

    if args.no_cnn:
        print("Skipping CNN (--no-cnn).")
        return

    try:
        import torch  # noqa: F401
    except Exception as e:
        print(f"[cnn] torch unavailable ({e}); skipping CNN. Baseline is saved.")
        return

    finetune = load_real_records(size=args.size)
    if finetune:
        print(f"Found {len(finetune)} real samples in dataset.jsonl for fine-tuning.")
    else:
        print("No data/dataset.jsonl yet -> training on synthetic only.")

    train_cnn(train_recs, val_recs, size=args.size, epochs=args.epochs,
              finetune_recs=finetune)


if __name__ == "__main__":
    main()
