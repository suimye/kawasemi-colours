#!/usr/bin/env python3
"""Predict the structural COLOUR produced by a nanostructure, from an SEM/TEM image.

Uses the trained CNN (model/checkpoint.pt, image -> colour) when torch is
available; always also reports the interpretable FEATS_V1 -> colour baseline
(model/baseline.npz) for comparison and so it runs torch-free.

Usage:
  python predict.py path/to/sem.png [more.png ...]
  python predict.py --id PMC4685390_fig6c_jay        # pull an image from the dataset
  python predict.py --swatch out.png img1.png img2.png  # also save a colour-swatch montage

Prints, per image: predicted dominant wavelength (nm), saturation, value,
sRGB and hex. If the dataset has a ground-truth colour for that image it is
shown alongside with the error.
"""
import os, sys, json, argparse
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import extract_features                         # noqa
from colorutil import (VISIBLE_MIN_NM, VISIBLE_MAX_NM,        # noqa
                       hsv_like_to_rgb, rgb01_to_srgb8, rgb_to_hex,
                       srgb8_to_peak_sat_val)
CKPT = os.path.join(HERE, "checkpoint.pt")
BASELINE = os.path.join(HERE, "baseline.npz")
DATASET = os.path.normpath(os.path.join(HERE, "..", "data", "dataset.jsonl"))
SPAN = VISIBLE_MAX_NM - VISIBLE_MIN_NM


def decode(y):
    return float(y[0]) * SPAN + VISIBLE_MIN_NM, float(y[1]), float(y[2])


def color_fields(peak, sat, val):
    rgb01 = hsv_like_to_rgb(peak, sat, val)
    srgb = rgb01_to_srgb8(rgb01)
    return srgb, rgb_to_hex(srgb)


# ---- baseline (numpy, always available) ----
def baseline_predict(img):
    if not os.path.exists(BASELINE):
        return None
    d = np.load(BASELINE); W, mu, sd = d["W"], d["mu"], d["sd"]
    vec, _ = extract_features(img)
    xa = np.hstack([(vec - mu) / sd, [1.0]])
    return decode(xa @ W)


# ---- CNN (torch, optional) ----
_net = None
def cnn_predict(img):
    global _net
    try:
        import torch
    except Exception:
        return None
    if not os.path.exists(CKPT):
        return None
    from train import _build_torch
    ck = torch.load(CKPT, map_location="cpu")
    size = ck.get("size", 96)
    if _net is None:
        _net = _build_torch(size)()
        _net.load_state_dict(ck["state_dict"]); _net.eval()
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size, size)).astype(np.float32) / 255.0
    with torch.no_grad():
        y = _net(torch.tensor(g[None, None], dtype=torch.float32)).numpy()[0]
    return decode(y)


def load_dataset_index():
    idx = {}
    if os.path.exists(DATASET):
        base = os.path.dirname(DATASET)
        for line in open(DATASET):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            p = o.get("image_file", "")
            idx[o["id"]] = (os.path.join(base, p), o)
    return idx


def truth_for(path, idx):
    real = os.path.realpath(path)
    for _id, (p, o) in idx.items():
        if os.path.realpath(p) == real:
            return o
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--id", action="append", default=[], help="dataset record id(s)")
    ap.add_argument("--swatch", default=None, help="save a colour-swatch montage PNG")
    args = ap.parse_args()

    idx = load_dataset_index()
    paths = list(args.images)
    for rid in args.id:
        if rid in idx:
            paths.append(idx[rid][0])
        else:
            print(f"[warn] id not found: {rid}")
    if not paths:
        print("Give an image path or --id <record>. Example:\n"
              "  python predict.py --id PMC4685390_fig6c_jay")
        return

    tiles = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[skip] unreadable: {p}"); continue
        b = baseline_predict(img)
        c = cnn_predict(img)
        print(f"\n{os.path.basename(p)}")
        use = c or b
        for name, pred in [("CNN(image->colour)", c), ("baseline(feats->colour)", b)]:
            if pred is None:
                print(f"  {name:26} (unavailable)"); continue
            peak, sat, val = pred
            srgb, hexv = color_fields(peak, sat, val)
            print(f"  {name:26} peak={peak:6.1f}nm sat={sat:.2f} val={val:.2f}  {hexv}  sRGB{tuple(srgb)}")
        t = truth_for(p, idx)
        if t:
            tp = t["color"].get("peak_nm"); tn = t["color"].get("name")
            if use and tp:
                print(f"  reference label: {tn} peak={tp}nm  (CNN err {abs(use[0]-tp):.0f}nm)"
                      if c else f"  reference label: {tn} peak={tp}nm  (baseline err {abs(use[0]-tp):.0f}nm)")
        if args.swatch and use:
            srgb, _ = color_fields(*use)
            g = cv2.resize(img, (150, 150))
            tile = np.vstack([cv2.cvtColor(g, cv2.COLOR_GRAY2BGR),
                              np.full((36, 150, 3), (srgb[2], srgb[1], srgb[0]), np.uint8)])
            cv2.putText(tile, os.path.basename(p)[:18], (3, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            tiles.append(tile)
    if args.swatch and tiles:
        per = min(5, len(tiles))
        rows = [np.hstack(tiles[i:i+per]) for i in range(0, len(tiles), per)]
        w = max(r.shape[1] for r in rows)
        rows = [cv2.copyMakeBorder(r, 0, 0, 0, w-r.shape[1], cv2.BORDER_CONSTANT, value=(20,20,20)) for r in rows]
        cv2.imwrite(args.swatch, np.vstack(rows))
        print(f"\nsaved swatch montage -> {args.swatch}")


if __name__ == "__main__":
    main()
