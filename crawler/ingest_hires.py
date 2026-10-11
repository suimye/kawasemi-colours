#!/usr/bin/env python3
"""Ingest the full-resolution replacements: Eurasian jay (PMC4685390 fig6),
Saranathan (PMC3427513 fig2: broadbill / eastern bluebird / cotinga), and
hi-res Prum (PMC2706469 fig2, replacing the earlier low-res crops).

Scale strategy: pixel scale-bars in these small insets are unreliable to
measure, so absolute period is recorded from the PAPER-STATED value where the
paper gives one (jay fig6b plots Long Period 150nm[blue]->200nm[white]) in a
`reported_period_nm` field with `reported_source`. Extracted scale-free features
(order, anisotropy, pore fraction) remain valid; scale.calibrated=false.
Torch-free, safe to re-run.
"""
import os, sys, json, csv, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model")); sys.path.insert(0, os.path.join(ROOT, "crawler"))
from features import extract_features   # noqa
import colorutil as cu                   # noqa
DATA = os.path.join(ROOT, "data"); DROP = os.path.join(DATA, "real_images")
IMAGES = os.path.join(DATA, "images"); JSONL = os.path.join(DATA, "dataset.jsonl")

# id, src, box, species, common, structure, color_name, srgb, peak, reported_period_nm, reported_src, license, url, figure, note
ITEMS = [
 ("PMC4685390_fig6c_jay","PMC4685390_fig6.webp",(190,629,248,266),
  "Garrulus glandarius","Eurasian jay","channel","blue",[70,120,195],480,150,
  "fig6b Long Period plot (blue region ~150nm)","CC BY 4.0",
  "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4685390/","Fig 6c (channel nanostructure, 1um)",
  "spinodal channel keratin; blue barb region"),
 ("PMC4685390_fig6d_jay","PMC4685390_fig6.webp",(486,629,238,266),
  "Garrulus glandarius","Eurasian jay","channel","light-blue",[120,160,205],490,180,
  "fig6b Long Period plot (transition ~180nm)","CC BY 4.0",
  "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4685390/","Fig 6d (channel nanostructure, 1um)",
  "spinodal channel keratin; blue->white transition region"),
 ("PMC4685390_fig6e_jay","PMC4685390_fig6.webp",(867,619,276,276),
  "Garrulus glandarius","Eurasian jay","channel","blue",[70,120,195],480,175,
  "fig6b Long Period plot (150-200nm)","CC BY 4.0",
  "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4685390/","Fig 6e (TEM channel nanostructure, 1um)",
  "TEM of channel keratin nanostructure"),
 ("PMC3427513_fig2d_broadbill","PMC3427513_fig2.png",(16,626,511,461),
  "Serilophus lunatus","silver-breasted broadbill","channel","blue",[95,125,185],470,None,
  None,"OA (confirm reuse)","https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3427513/",
  "Fig 2d (SEM channel nanostructure)","quasi-ordered channel keratin"),
 ("PMC3427513_fig2e_bluebird","PMC3427513_fig2.png",(560,626,511,461),
  "Sialia sialis","eastern bluebird","channel","blue",[45,85,180],465,None,
  None,"OA (confirm reuse)","https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3427513/",
  "Fig 2e (SEM channel nanostructure)","REAL eastern-bluebird channel keratin (was synthetic proxy before)"),
 ("PMC3427513_fig2f_cotinga","PMC3427513_fig2.png",(1104,626,511,461),
  "Cotinga maynana","plum-throated cotinga","quasi_ordered_sphere","cyan-blue",[40,160,190],485,None,
  None,"OA (confirm reuse)","https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3427513/",
  "Fig 2f (SEM/TEM sphere nanostructure)","REAL cotinga sphere-type keratin (was synthetic proxy before)"),
 # hi-res Prum replaces the earlier low-res fig1 crops (same ids kept where possible)
 ("PMC2706469_fig1a_channel","PMC2706469_fig2.png",(17,17,515,531),
  "Aves (Prum 2009, channel-type)","structural-blue barb (channel-type)","channel","blue",[50,90,180],470,None,
  None,"CC BY 2.0","https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2706469/",
  "Fig 2a (channel beta-keratin, hi-res)","UPGRADED to full-res; per-panel species not confirmed"),
 ("PMC2706469_fig1c_sphere","PMC2706469_fig2.png",(565,17,515,531),
  "Aves (Prum 2009, sphere-type)","structural-blue barb (sphere-type)","quasi_ordered_sphere","blue",[45,95,175],460,None,
  None,"CC BY 2.0","https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2706469/",
  "Fig 2c (sphere beta-keratin, hi-res)","UPGRADED to full-res; per-panel species not confirmed"),
 ("PMC2706469_fig1e_channel","PMC2706469_fig2.png",(17,1147,515,515),
  "Aves (Prum 2009, channel-type)","structural-blue barb (channel-type)","channel","blue",[50,90,180],470,None,
  None,"CC BY 2.0","https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2706469/",
  "Fig 2e (high-mag channel, 120nm bar, hi-res)","UPGRADED to full-res"),
 ("PMC2706469_fig2f_sphere","PMC2706469_fig2.png",(565,1147,515,515),
  "Aves (Prum 2009, sphere-type)","structural-blue barb (ordered sphere)","quasi_ordered_sphere","blue",[45,95,175],460,None,
  None,"CC BY 2.0","https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2706469/",
  "Fig 2f (ordered sphere array, hi-res)","ordered close-packed sphere array"),
]


def load(): return [json.loads(l) for l in open(JSONL) if l.strip()]


def save(recs):
    with open(JSONL, "w") as f:
        for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cols = ["id","image_file","real_image","license","url","figure","species","common_name",
            "structure_type","color_name","color_hex","color_srgb","peak_nm","reported_period_nm",
            "char_length_nm","pore_fraction","calibrated","notes"]
    with open(os.path.join(DATA, "dataset.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in recs:
            s, c = r["source"], r["color"]
            w.writerow([r["id"], r["image_file"], r.get("real_image", False), s.get("license",""),
                        s.get("url",""), s.get("figure",""), r.get("species",""), r.get("common_name",""),
                        r.get("structure_type",""), c.get("name",""), c.get("hex",""),
                        "|".join(map(str, c.get("srgb", []))), c.get("peak_nm",""),
                        r.get("reported_period_nm",""), r.get("features",{}).get("char_length_nm",""),
                        r.get("features",{}).get("pore_fraction",""),
                        r.get("scale",{}).get("calibrated", False), r.get("notes","")])


def main():
    recs = load(); idx = {r["id"]: i for i, r in enumerate(recs)}
    upd = added = 0
    for (rid, src, box, sp, cn, st, coln, srgb, peak, rep, reps, lic, url, fig, note) in ITEMS:
        im = cv2.imread(os.path.join(DROP, src))
        if im is None: print("  ! missing", src); continue
        x, y, w, h = box
        gray = cv2.cvtColor(im[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        base = rid + ".png"; cv2.imwrite(os.path.join(IMAGES, base), gray)
        _, feat = extract_features(gray, px_per_um=None)
        rec = {"id": rid, "image_file": "images/"+base, "real_image": True,
               "source": {"pmcid": rid.split("_")[0] if rid.startswith("PMC") else "", "doi": "",
                          "title": fig.split("(")[0].strip(), "license": lic, "url": url, "figure": fig},
               "species": sp, "common_name": cn, "structure_type": st,
               "color": {"name": coln, "hex": cu.rgb_to_hex(srgb), "srgb": srgb, "peak_nm": peak},
               "reported_period_nm": rep, "reported_period_source": reps,
               "scale": {"bar_um": None, "px_per_um": None, "calibrated": False},
               "features": feat,
               "notes": f"real_image (full-res) cropped from {src} box={box}; "
                        f"{'reported period '+str(rep)+'nm ['+str(reps)+']' if rep else 'scale uncalibrated'}. {note}"}
        if rid in idx: recs[idx[rid]] = rec; upd += 1
        else: recs.append(rec); added += 1
        rp = f"reported≈{rep}nm" if rep else "uncal"
        print(f"  {rid:30} char_length≈{feat['char_length_nm']:6.0f}nm aniso={feat['anisotropy']:.2f} [{rp}]")
    save(recs)
    real = sum(1 for r in recs if r.get("real_image"))
    print(f"\nUpdated {upd}, added {added}. Dataset: {len(recs)} total | {real} real images.")


if __name__ == "__main__":
    main()
