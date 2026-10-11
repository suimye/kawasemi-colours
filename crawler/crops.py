"""Crop definitions mapping real downloaded figures -> dataset records.
Boxes are (x, y, w, h) in pixels on the real image. scale_nm/scale_bar note the
published scale bar for that panel (used for optional calibration).
Derived by visual inspection of each figure. Edit here if a crop is off.
"""

# record_id : dict(src, box=(x,y,w,h), scale_bar_label)
BIRD_CROPS = {
    # PMC8599086 blue tit — zoab011f1.jpg (636x1244), SEM, 20um bars
    "PMC8599086_fig1c_bluevane": dict(src="zoab011f1.jpg", box=(120, 345, 400, 380), scale="20um"),
    "PMC8599086_fig1d_whitedot": dict(src="zoab011f1.jpg", box=(170, 840, 340, 300), scale="20um"),

    # PMC9832292 pheasant fig2 — rsos220786f02.jpg (685x234); panel (c) magnified TEM, 300nm
    "PMC9832292_fig2b_blue":  dict(src="rsos220786f02.jpg", box=(308, 12, 150, 96), scale="300nm"),
    "PMC9832292_fig2b_green": dict(src="rsos220786f02.jpg", box=(308, 122, 150, 96), scale="300nm"),

    # PMC9832292 pheasant fig3 — rsos220786f03.jpg (685x542); panel (a) EM row, 200nm
    "PMC9832292_fig3a_blue_etch60":  dict(src="rsos220786f03.jpg", box=(240, 18, 195, 130), scale="200nm"),
    "PMC9832292_fig3a_blue_etch120": dict(src="rsos220786f03.jpg", box=(470, 18, 195, 130), scale="200nm"),

    # PMC8505553 review Fig4 (777x485), row (a): jay TEM, rock dove SEM
    "PMC8505553_fig4_stellersjay": dict(src="PMC8505553_42649_2021_63_Fig4_HTML.jpg", box=(212, 12, 165, 185), scale=None),
    "PMC8505553_fig4_rockdove":    dict(src="PMC8505553_42649_2021_63_Fig4_HTML.jpg", box=(598, 14, 168, 182), scale=None),

    # PMC8505553 review Fig5 (777x351)
    "PMC8505553_fig5a_parotia":  dict(src="PMC8505553_42649_2021_63_Fig5_HTML.jpg", box=(182, 10, 215, 152), scale=None),
    "PMC8505553_fig5b_teal":     dict(src="PMC8505553_42649_2021_63_Fig5_HTML.jpg", box=(610, 8,  152, 150), scale=None),
    "PMC8505553_fig5c_starling": dict(src="PMC8505553_42649_2021_63_Fig5_HTML.jpg", box=(190, 192, 205, 145), scale=None),
    "PMC8505553_fig5d_racket":   dict(src="PMC8505553_42649_2021_63_Fig5_HTML.jpg", box=(598, 192, 168, 145), scale=None),
}

# Scale calibration: px_per_um measured from each panel's published scale bar
# (bar length in px / bar physical length in um). None = uncalibrated because the
# panel is low-magnification (photonic ~150nm period under-resolved) or the bar
# value/length could not be reliably located. Uncalibrated records keep real
# scale-free features (anisotropy, pore_fraction, order); absolute nm is relative.
PX_PER_UM = {
    "PMC9832292_fig2b_blue": 106.7,       # 32px / 300nm  (melanosome array, resolved)
    "PMC9832292_fig2b_green": 110.0,      # 33px / 300nm
    "PMC9832292_fig3a_blue_etch60": 315.0,   # 63px / 200nm
    "PMC9832292_fig3a_blue_etch120": 315.0,  # 63px / 200nm
    "PMC8505553_fig5b_teal": 84.0,        # 42px / 500nm  (hex array, resolved -> 162nm period OK)
    "PMC8505553_fig5d_racket": 138.0,     # 69px / 500nm  (-> 525nm platelet spacing, OK)
    # Uncalibrated (under-resolved low-mag, unreliable bar, or coarse crop):
    "PMC8599086_fig1c_bluevane": None,    # 20um tick-ruler SEM; ~150nm spongy period unresolved
    "PMC8599086_fig1d_whitedot": None,
    "PMC8505553_fig4_stellersjay": None,  # 2um whole-section TEM; bar not locatable in crop
    "PMC8505553_fig5a_parotia": None,     # 5um whole-barbule TEM; bar not locatable in crop
    "PMC8505553_fig4_rockdove": None,     # 2um bar clipped/ambiguous -> gave impossible 7um period
    "PMC8505553_fig5c_starling": None,    # 500nm bar read gave implausible 1.8um; coarse crop
}

# PMC8505553 Fig6 (777x653) biomimetic melanin films — panel (d): 5 SEM (top) + colour swatch (below)
# These become NEW records (artificial structural colour, directly labelled).
FIG6_SRC = "42649_2021_63_Fig6_HTML.jpg"
FIG6_SEM_Y, FIG6_SEM_H = 437, 112          # SEM row
FIG6_SWATCH_Y, FIG6_SWATCH_H = 565, 70     # colour swatch row
FIG6_COLS = [(12, 150), (168, 148), (322, 148), (476, 146), (628, 145)]  # (x, w) per column
FIG6_IDS = ["sphere", "roundish", "mixed", "ellipsoid", "elong"]
