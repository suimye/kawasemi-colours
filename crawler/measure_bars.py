"""Detect scale-bar pixel length inside a region of interest and render a
zoomed verification montage. Bars are wide, short, near-solid runs of bright
(SEM) or dark (TEM) pixels."""
import cv2, numpy as np, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "real_images")
OUT = "/sessions/great-quirky-hamilton/mnt/outputs/bars_preview.png"

# (label, file, roi(x,y,w,h), polarity, phys_value)
BARS = [
    ("bluetit_C 20um",  "zoab011f1.jpg",        (355, 715, 270, 55),  "white", "20um"),
    ("bluetit_D 20um",  "zoab011f1.jpg",        (355, 1180, 270, 55), "white", "20um"),
    ("pheas2c_blue 300nm",  "rsos220786f02.jpg",(395, 86, 70, 26),   "white", "300nm"),
    ("pheas2c_green 300nm", "rsos220786f02.jpg",(395, 200, 70, 26),  "white", "300nm"),
    ("pheas3a_ii 200nm",  "rsos220786f03.jpg",  (370, 128, 90, 30),  "black", "200nm"),
    ("pheas3a_iii 200nm", "rsos220786f03.jpg",  (600, 128, 90, 30),  "black", "200nm"),
    ("jay 2um",     "PMC8505553_42649_2021_63_Fig4_HTML.jpg", (345, 8, 40, 16),  "black", "2um"),
    ("rockdove 2um","PMC8505553_42649_2021_63_Fig4_HTML.jpg", (735, 8, 40, 16),  "black", "2um"),
    ("parotia 5um", "PMC8505553_42649_2021_63_Fig5_HTML.jpg", (350, 118, 55, 20),"black", "5um"),
    ("teal 500nm",  "PMC8505553_42649_2021_63_Fig5_HTML.jpg", (720, 138, 55, 22),"white", "500nm"),
    ("starling 500nm","PMC8505553_42649_2021_63_Fig5_HTML.jpg",(350, 320, 55, 22),"white","500nm"),
    ("racket 500nm","PMC8505553_42649_2021_63_Fig5_HTML.jpg", (720, 320, 55, 22),"white", "500nm"),
]

def measure(gray, polarity):
    if polarity == "white":
        _, m = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    else:
        _, m = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w >= 8 and h <= max(6, w) and w > h and area > w * 0.4:
            if best is None or w > best[2]:
                best = (x, y, w, h)
    return best

def main():
    tiles = []
    for label, f, (x, y, w, h), pol, phys in BARS:
        im = cv2.imread(os.path.join(D, f), 0)
        roi = im[y:y+h, x:x+w]
        bb = measure(roi, pol)
        vis = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        barpx = None
        if bb:
            bx, by, bw, bh = bb
            cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), (0, 0, 255), 1)
            barpx = bw
        z = cv2.resize(vis, (w*5, h*5), interpolation=cv2.INTER_NEAREST)
        z = cv2.copyMakeBorder(z, 20, 4, 4, 4, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        txt = f"{label}: bar={barpx}px" if barpx else f"{label}: NOT FOUND"
        cv2.putText(z, txt, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        tiles.append((z, label, barpx, phys))
        print(txt)
    # montage
    H = max(t[0].shape[0] for t in tiles); W = max(t[0].shape[1] for t in tiles)
    padded = [cv2.copyMakeBorder(t[0], 0, H-t[0].shape[0], 0, W-t[0].shape[1],
              cv2.BORDER_CONSTANT, value=(25, 25, 25)) for t in tiles]
    rows = [np.hstack(padded[i:i+3]) for i in range(0, len(padded), 3)]
    mw = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, mw-r.shape[1], cv2.BORDER_CONSTANT, value=(25,25,25)) for r in rows]
    cv2.imwrite(OUT, np.vstack(rows))
    print("wrote", OUT)

if __name__ == "__main__":
    main()
