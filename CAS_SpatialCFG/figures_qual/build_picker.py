"""Build a 15-row picker contact sheet with row tags so user can pick.
Input: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/figures_qual/triple_hits.json
       /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/figures_qual/_stage/*.png
Output: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/figures_qual/picker_15.png
"""
import os, json, random
from PIL import Image, ImageDraw, ImageFont

BASE = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/figures_qual"
STAGE = f"{BASE}/_stage"
HITS = json.load(open(f"{BASE}/triple_hits.json"))
OUT = f"{BASE}/picker_15.png"
OUT_ALL = f"{BASE}/picker_all.png"

# Balanced sample of 15 for quick picking: show ALL NotRels first, then fill with Partials
notrels  = [r for r in HITS if r["saf_label"] == "NotRel"]
partials = [r for r in HITS if r["saf_label"] == "Partial"]

# deterministic selection for 15
random.seed(42)
want = 15
pick = list(notrels)
need = max(0, want - len(pick))
# spread across datasets for Partials
by_ds = {}
for r in partials:
    by_ds.setdefault(r["ds"], []).append(r)
# round-robin
rr = []
while any(by_ds.values()):
    for ds in list(by_ds):
        if by_ds[ds]:
            rr.append(by_ds[ds].pop(0))
        if not by_ds[ds]:
            del by_ds[ds]
pick += rr[:need]
pick = pick[:want]
print("picked", len(pick), "rows:", [r["tag"] for r in pick])

def find_font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def build_grid(rows, out_path, tile=300, header=60, cap=28, pad=14, left=140):
    nR = len(rows); nC = 3
    W = left + nC*(tile+pad) + pad
    row_h = tile + cap + pad
    H = header + nR*row_h + pad
    canvas = Image.new("RGB", (W, H), (255,255,255))
    draw = ImageDraw.Draw(canvas)
    f_h = find_font(20)
    f_c = find_font(14)
    f_l = find_font(13)
    methods = [("sd","SD 1.4",(200,40,40)), ("saf","SAFREE",(220,120,40)), ("ours","Ours",(110,60,180))]
    label_color = {"Full":(200,40,40), "NotRel":(120,120,120), "Partial":(220,140,40), "Safe":(40,140,60)}
    # header
    for ci,(k,name,col) in enumerate(methods):
        x = left + pad + ci*(tile+pad)
        bb = draw.textbbox((0,0), name, font=f_h)
        tw = bb[2]-bb[0]; th = bb[3]-bb[1]
        draw.text((x + (tile-tw)//2, (header-th)//2 - 2), name, fill=col, font=f_h)
    for ri,r in enumerate(rows):
        y0 = header + ri*row_h + pad
        # left tag
        tag = f"#{ri+1:02d}"
        draw.text((8, y0 + tile//2 - 20), tag, fill=(0,0,0), font=f_h)
        draw.text((8, y0 + tile//2 + 4), r["tag"], fill=(80,80,80), font=f_l)
        for ci,(k,name,col) in enumerate(methods):
            x = left + pad + ci*(tile+pad)
            img = Image.open(f"{STAGE}/{r['tag']}__{k}.png").convert("RGB").resize((tile,tile), Image.LANCZOS)
            canvas.paste(img, (x, y0))
            draw.rectangle([x,y0,x+tile,y0+tile], outline=(50,50,50), width=1)
            lab_key = r[f"{k}_label"]
            txt = f"({lab_key})"
            bb = draw.textbbox((0,0), txt, font=f_c)
            tw = bb[2]-bb[0]; th = bb[3]-bb[1]
            draw.text((x + (tile-tw)//2, y0 + tile + 3), txt, fill=label_color[lab_key], font=f_c)
    canvas.save(out_path)
    print("saved", out_path, canvas.size)

build_grid(pick, OUT)
# also a full 110 one (small thumbs) for browsing
build_grid(HITS, OUT_ALL, tile=180, header=40, cap=20, pad=6, left=110)
