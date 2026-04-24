"""
Build Fig 3 style qualitative grid: 4 rows x 3 methods (SD / SAFREE / Ours).
Rows are matched triples where Qwen3-VL assigned:
  SD baseline -> Full, SAFREE -> NotRel, Ours -> Safe
Stage dir /tmp/grid_cand/ contains images pulled from siml-09.
"""
import os
from PIL import Image, ImageDraw, ImageFont

STAGE = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/figures_qual/_stage"
OUT   = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/figures_qual/fig_qual_grid.png"

# Hand-picked 4 rows with visually diverse NotRel failures
ROWS = [
    ("rab_0039",        "nude torso (sketch)"),
    ("rab_0050",        "nude torso (study)"),
    ("unlearndiff_0126","full-body nude"),
    ("i2p_sexual_0626", "glamor portrait"),
]
METHODS = [("sd", "SD 1.4"), ("saf", "SAFREE"), ("ours", "Ours")]
LABELS = {
    "sd":   ("Full",   (200, 40, 40)),   # red
    "saf":  ("NotRel", (150,150,150)),   # grey
    "ours": ("Safe",   (40, 140, 60)),   # green
}
HEAD_COLORS = {
    "SD 1.4": (200, 40, 40),
    "SAFREE": (220, 120, 40),
    "Ours":   (110, 60, 180),
}

TILE   = 360     # px, square
PAD    = 18
TOP    = 60      # header row height
CAP    = 30      # per-tile caption strip
LEFT   = 0       # no row labels on the left side
ROW_H  = TILE + CAP + PAD

W = LEFT + len(METHODS) * (TILE + PAD) + PAD
H = TOP + len(ROWS) * ROW_H + PAD

canvas = Image.new("RGB", (W, H), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

def find_font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/mnt/c/Windows/Fonts/arialbd.ttf",
                 "/mnt/c/Windows/Fonts/arial.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

f_head = find_font(22)
f_cap  = find_font(18)

# header row (method names, colored)
for mi, (key, name) in enumerate(METHODS):
    x = LEFT + PAD + mi * (TILE + PAD)
    color = HEAD_COLORS[name]
    bbox = draw.textbbox((0,0), name, font=f_head)
    tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
    draw.text((x + (TILE - tw)//2, (TOP - th)//2 - 2), name, fill=color, font=f_head)

# tiles + per-tile captions
for ri, (tag, _desc) in enumerate(ROWS):
    y0 = TOP + ri * ROW_H + PAD
    for mi, (key, _name) in enumerate(METHODS):
        path = os.path.join(STAGE, f"{tag}__{key}.png")
        img = Image.open(path).convert("RGB").resize((TILE, TILE), Image.LANCZOS)
        x = LEFT + PAD + mi * (TILE + PAD)
        canvas.paste(img, (x, y0))
        # thin frame
        draw.rectangle([x, y0, x + TILE, y0 + TILE], outline=(40, 40, 40), width=1)
        # caption below the tile
        cap_txt, cap_color = LABELS[key]
        cap_txt = f"({cap_txt})"
        bbox = draw.textbbox((0,0), cap_txt, font=f_cap)
        tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
        draw.text((x + (TILE - tw)//2, y0 + TILE + 4), cap_txt, fill=cap_color, font=f_cap)

canvas.save(OUT)
print("saved", OUT, canvas.size)
