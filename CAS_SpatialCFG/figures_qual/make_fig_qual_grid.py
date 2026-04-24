"""Build the final qualitative grid (SD 1.4 / SAFREE / Ours) from final_picks.json."""
import os, json
from PIL import Image, ImageDraw, ImageFont

BASE  = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/figures_qual"
STAGE = f"{BASE}/_stage"
PICKS = json.load(open(f"{BASE}/final_picks.json"))
OUT   = f"{BASE}/fig_qual_grid.png"

METHODS = [("sd","SD 1.4",(200,40,40)), ("saf","SAFREE",(220,120,40)), ("ours","Ours",(110,60,180))]
LABEL_COLOR = {"Full":(200,40,40), "NotRel":(120,120,120), "Partial":(220,140,40), "Safe":(40,140,60)}

TILE = 360
PAD  = 18
TOP  = 60
CAP  = 30

def find_font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

f_head = find_font(22)
f_cap  = find_font(18)

nR = len(PICKS); nC = len(METHODS)
row_h = TILE + CAP + PAD
W = nC*(TILE+PAD) + PAD
H = TOP + nR*row_h + PAD

canvas = Image.new("RGB", (W, H), (255,255,255))
draw = ImageDraw.Draw(canvas)

# header
for ci,(k,name,col) in enumerate(METHODS):
    x = PAD + ci*(TILE+PAD)
    bb = draw.textbbox((0,0), name, font=f_head)
    tw = bb[2]-bb[0]; th = bb[3]-bb[1]
    draw.text((x + (TILE-tw)//2, (TOP-th)//2 - 2), name, fill=col, font=f_head)

# rows
for ri,r in enumerate(PICKS):
    y0 = TOP + ri*row_h + PAD
    for ci,(k,name,col) in enumerate(METHODS):
        x = PAD + ci*(TILE+PAD)
        path = f"{STAGE}/{r['tag']}__{k}.png"
        img = Image.open(path).convert("RGB").resize((TILE,TILE), Image.LANCZOS)
        canvas.paste(img, (x, y0))
        draw.rectangle([x,y0,x+TILE,y0+TILE], outline=(40,40,40), width=1)
        lab = r[f"{k}_label"]
        txt = f"({lab})"
        bb = draw.textbbox((0,0), txt, font=f_cap)
        tw = bb[2]-bb[0]
        draw.text((x + (TILE-tw)//2, y0 + TILE + 4), txt, fill=LABEL_COLOR[lab], font=f_cap)

canvas.save(OUT)
print("saved", OUT, canvas.size)
