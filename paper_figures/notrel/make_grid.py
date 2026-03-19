import json
import math
import os
from PIL import Image, ImageDraw, ImageFont

# === Config ===
JSON_PATH = "/mnt/home/yhgil99/unlearning/paper_figures/notrel/eval_per_image.json"
IMAGE_DIR = "/mnt/home/yhgil99/unlearning/paper_figures/notrel"
OUTPUT_PATH = "/mnt/home/yhgil99/unlearning/paper_figures/notrel/grid_notrel.png"

THUMB_SIZE = 256
LABEL_HEIGHT = 50
COLS = 3
PADDING = 4

# Colors
COLOR_NUDENET_UNSAFE = (220, 50, 50)    # red
COLOR_NUDENET_SAFE = (50, 130, 200)     # blue for nudenet safe
COLOR_VLM_SAFE = (50, 180, 50)          # green
COLOR_VLM_PARTIAL = (255, 165, 0)       # orange
COLOR_VLM_NOTREL = (120, 120, 120)      # gray for NotRel

# === Load data ===
with open(JSON_PATH) as f:
    data = json.load(f)

filtered = data  # use all images
print(f"Selected {len(filtered)} images")

# Try to load a font
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
except:
    font = ImageFont.load_default()
    font_small = font

cell_w = THUMB_SIZE + PADDING
cell_h = THUMB_SIZE + LABEL_HEIGHT + PADDING
rows = math.ceil(len(filtered) / COLS)

canvas = Image.new("RGB", (cell_w * COLS + PADDING, cell_h * rows + PADDING), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

for idx, entry in enumerate(filtered):
    col = idx % COLS
    row = idx // COLS
    x = PADDING + col * cell_w
    y = PADDING + row * cell_h

    img_path = os.path.join(IMAGE_DIR, entry["filename"])
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
    except Exception as e:
        img = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), (200, 200, 200))
        print(f"Failed to load {img_path}: {e}")

    canvas.paste(img, (x, y))

    label_y = y + THUMB_SIZE

    # NudeNet label
    nn_score = entry.get("nudenet_score", 0)
    nn_unsafe = entry.get("nudenet_unsafe", False)
    nn_color = COLOR_NUDENET_UNSAFE if nn_unsafe else COLOR_NUDENET_SAFE
    nn_label = "Unsafe" if nn_unsafe else "Safe"
    nn_text = f"NudeNet: {nn_label} ({nn_score:.2f})"
    draw.text((x + 4, label_y + 2), nn_text, fill=nn_color, font=font_small)

    # VLM label
    vlm_cat = entry["qwen_category"]
    vlm_color_map = {"Safe": COLOR_VLM_SAFE, "Partial": COLOR_VLM_PARTIAL, "NotRel": COLOR_VLM_NOTREL}
    vlm_color = vlm_color_map.get(vlm_cat, (220, 50, 50))
    vlm_text = f"VLM: {vlm_cat}"
    draw.text((x + 4, label_y + 20), vlm_text, fill=vlm_color, font=font)

canvas.save(OUTPUT_PATH, quality=95)
print(f"Saved grid to {OUTPUT_PATH}")
print(f"Grid size: {canvas.size}, {rows} rows x {COLS} cols")
