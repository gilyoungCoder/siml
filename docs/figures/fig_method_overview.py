"""
fig_method_overview.py
Publication-quality method overview for:
  "SafeGen: Training-Free Safe Image Generation via Dual-Probe Spatial Guidance"
NeurIPS submission figure — full-width single row, ~14×5 inches.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge
from matplotlib.patches import ConnectionPatch
import matplotlib.gridspec as gridspec
import numpy as np

# ── Aesthetics ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['DejaVu Serif', 'Times New Roman', 'Times', 'serif'],
    'mathtext.fontset':   'dejavuserif',
    'font.size':          9,
    'axes.linewidth':     0.6,
    'pdf.fonttype':       42,   # embed fonts
    'ps.fonttype':        42,
})

# Colour palette
C = {
    'safe':        '#2E6FAD',   # steel blue
    'safe_light':  '#C8DDED',
    'unsafe':      '#C0392B',   # warm red
    'unsafe_light':'#F5C6C3',
    'decision':    '#27AE60',   # green
    'decision_lt': '#C8EDD6',
    'neutral':     '#4A4A4A',
    'neutral_lt':  '#EFEFEF',
    'probe_text':  '#8B4513',   # brown
    'probe_img':   '#6A0DAD',   # violet
    'mask':        '#E67E22',   # orange
    'mask_light':  '#FDEBD0',
    'arrow':       '#333333',
    'bg':          '#FFFFFF',
    'border':      '#AAAAAA',
}

# ── Figure / axes ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 5.6), facecolor=C['bg'])

# We use a free-form layout with transform-based placement
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 15)
ax.set_ylim(0, 5.6)
ax.axis('off')
ax.set_facecolor(C['bg'])

# ── Helper utilities ─────────────────────────────────────────────────────────

def fbox(ax, x, y, w, h, fc, ec, lw=0.9, alpha=1.0, radius=0.12, zorder=2):
    """Rounded rectangle."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad=0,rounding_size={radius}',
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=zorder
    )
    ax.add_patch(box)
    return box


def txt(ax, x, y, s, **kw):
    defaults = dict(ha='center', va='center', fontsize=9, color=C['neutral'],
                    zorder=5)
    defaults.update(kw)
    return ax.text(x, y, s, **defaults)


def arrow(ax, x0, y0, x1, y1, color=C['arrow'], lw=1.2,
          arrowstyle='->', mutation_scale=10, zorder=4, **kw):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=arrowstyle,
                                color=color, lw=lw,
                                mutation_scale=mutation_scale,
                                connectionstyle='arc3,rad=0.0'),
                zorder=zorder, **kw)


def small_arrow(ax, x0, y0, x1, y1, color=C['arrow'], lw=0.9):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=8))


def section_header(ax, x, y, w, label, color):
    """Thin coloured header bar above a panel."""
    fbox(ax, x, y, w, 0.28, fc=color, ec=color, lw=0, alpha=0.85,
         radius=0.08, zorder=3)
    txt(ax, x + w/2, y + 0.14, label,
        fontsize=7.0, fontweight='bold', color='white', zorder=6)


def noise_icon(ax, cx, cy, r=0.25, color='#888888', label=''):
    """Small noisy-circle icon to represent a noise prediction ε."""
    theta = np.linspace(0, 2*np.pi, 80)
    np.random.seed(42)
    noise = np.random.randn(80) * 0.04
    xs = cx + (r + noise) * np.cos(theta)
    ys = cy + (r + noise) * np.sin(theta)
    ax.plot(xs, ys, color=color, lw=0.8, zorder=6)
    if label:
        txt(ax, cx, cy, label, fontsize=7, color=color)


def heatmap_block(ax, x, y, w, h, cmap='Reds', alpha=0.75):
    """Fake spatial heatmap as a coloured grid."""
    nx, ny = 8, 6
    np.random.seed(7)
    data = np.random.rand(ny, nx)
    # bias centre to be hot
    cx_, cy_ = nx//2, ny//2
    for i in range(ny):
        for j in nx // 3, nx // 3 + 1, nx // 3 + 2:
            data[i, j] += 0.6
    data = np.clip(data / data.max(), 0, 1)
    extent = [x, x+w, y, y+h]
    ax.imshow(data, extent=extent, origin='lower', cmap=cmap,
              alpha=alpha, aspect='auto', zorder=3,
              interpolation='bilinear')
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False,
                                edgecolor=C['border'], lw=0.7, zorder=4))


def mask_block(ax, x, y, w, h):
    """Binary spatial mask rendered as two-tone grid."""
    nx, ny = 8, 6
    np.random.seed(11)
    data = np.zeros((ny, nx))
    data[1:5, 2:5] = 1   # core unsafe region
    data[2, 1] = 1
    data[3, 5] = 1
    cmap = matplotlib.colors.ListedColormap([C['safe_light'], C['mask']])
    extent = [x, x+w, y, y+h]
    ax.imshow(data, extent=extent, origin='lower', cmap=cmap,
              alpha=0.85, aspect='auto', zorder=3,
              interpolation='none')
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False,
                                edgecolor=C['border'], lw=0.7, zorder=4))


# ════════════════════════════════════════════════════════════════════════════
# Layout constants — panel left edges, shared y-baseline
# ════════════════════════════════════════════════════════════════════════════
PAD   = 0.18   # outer margin
Y0    = 0.55   # bottom of all main panels
PH    = 3.80   # panel height
ARROW_Y = Y0 + PH/2  # vertical centre for inter-panel arrows

# x positions of each stage panel — wider figure (15 inches)
X_INPUT  = PAD
X_UNet   = 1.40
X_WHEN   = 3.60
X_WHERE  = 6.40
X_HOW    = 10.40
X_OUTPUT = 13.10

W_INPUT  = 0.95
W_UNet   = 1.80
W_WHEN   = 2.40
W_WHERE  = 3.50
W_HOW    = 2.30
W_OUTPUT = 1.50

# ════════════════════════════════════════════════════════════════════════════
# ① Input prompt box
# ════════════════════════════════════════════════════════════════════════════
fbox(ax, X_INPUT, Y0, W_INPUT, PH,
     fc='#F7F7F7', ec=C['neutral'], lw=0.8)
txt(ax, X_INPUT + W_INPUT/2, Y0 + PH*0.72,
    'Input\nPrompt', fontsize=8.5, fontweight='bold')
# small prompt lines
for i, (s, w_) in enumerate([(0.55, 0.04), (0.48, 0.03), (0.35, 0.025)]):
    yy = Y0 + PH*0.42 - i*0.26
    fbox(ax, X_INPUT + 0.08, yy, s, 0.12, fc='#DDDDDD', ec='#BBBBBB', lw=0.4)
txt(ax, X_INPUT + W_INPUT/2, Y0 + 0.22,
    '"a person\nat the beach"',
    fontsize=6.5, style='italic', color='#555555')

# ════════════════════════════════════════════════════════════════════════════
# ② Diffusion Model (UNet/DiT) — arrow + box
# ════════════════════════════════════════════════════════════════════════════
arrow(ax, X_INPUT + W_INPUT + 0.04, ARROW_Y,
          X_UNet - 0.04, ARROW_Y)

fbox(ax, X_UNet, Y0, W_UNet, PH,
     fc=C['safe_light'], ec=C['safe'], lw=1.0)
section_header(ax, X_UNet, Y0 + PH - 0.28, W_UNet, 'Diffusion Model', C['safe'])

txt(ax, X_UNet + W_UNet/2, Y0 + PH*0.60,
    'UNet / DiT\n/ MMDiT', fontsize=8.5, fontweight='bold', color=C['safe'])

# Three ε mini-boxes
eps_colors = [C['safe'], '#888888', C['unsafe']]
eps_labels = [r'$\varepsilon_{\rm prompt}$',
              r'$\varepsilon_{\rm null}$',
              r'$\varepsilon_{\rm target}$']
eps_subtitles = ['(user)', '(empty)', '(unsafe)']
for k, (ec_, el, es) in enumerate(zip(eps_colors, eps_labels, eps_subtitles)):
    bx = X_UNet + 0.10
    by = Y0 + 0.14 + k * 0.82
    fbox(ax, bx, by, W_UNet - 0.20, 0.60,
         fc='white', ec=ec_, lw=0.9, radius=0.07)
    txt(ax, bx + (W_UNet-0.20)/2, by + 0.38, el,
        fontsize=8, color=ec_)
    txt(ax, bx + (W_UNet-0.20)/2, by + 0.14, es,
        fontsize=6.5, color='#777777')

# ════════════════════════════════════════════════════════════════════════════
# ③ WHEN — CAS gate
# ════════════════════════════════════════════════════════════════════════════
arrow(ax, X_UNet + W_UNet + 0.04, ARROW_Y,
          X_WHEN - 0.04, ARROW_Y)

fbox(ax, X_WHEN, Y0, W_WHEN, PH,
     fc=C['decision_lt'], ec=C['decision'], lw=1.1)
section_header(ax, X_WHEN, Y0 + PH - 0.28, W_WHEN,
               'WHEN: Concept Alignment Score (CAS)', C['decision'])

# CAS formula box
fbox(ax, X_WHEN + 0.10, Y0 + PH*0.60, W_WHEN - 0.20, 0.85,
     fc='white', ec=C['decision'], lw=0.9, radius=0.08)
txt(ax, X_WHEN + W_WHEN/2, Y0 + PH*0.60 + 0.60,
    r'$\mathrm{CAS}_t = \cos\!\left(\varepsilon_{\rm p}{-}\varepsilon_{\rm n},\;'
    r'\varepsilon_{\rm t}{-}\varepsilon_{\rm n}\right)$',
    fontsize=8.0, color=C['neutral'])
txt(ax, X_WHEN + W_WHEN/2, Y0 + PH*0.60 + 0.22,
    r'cosine similarity of noise residuals',
    fontsize=6.5, color='#666666')

# Decision diamond
DX = X_WHEN + W_WHEN/2
DY = Y0 + 1.15
diamond_pts = np.array([[DX, DY+0.42], [DX+0.50, DY],
                         [DX, DY-0.42], [DX-0.50, DY]])
diam = plt.Polygon(diamond_pts, closed=True,
                   facecolor='white', edgecolor=C['decision'],
                   linewidth=1.1, zorder=4)
ax.add_patch(diam)
txt(ax, DX, DY + 0.14, r'$\mathrm{CAS}_t$', fontsize=8, color=C['decision'])
txt(ax, DX, DY - 0.15, r'$> \tau$?', fontsize=8, color=C['decision'])

# YES / NO labels
txt(ax, DX + 0.65, DY + 0.05, 'YES', fontsize=7, color=C['decision'],
    fontweight='bold')
ax.annotate('', xy=(X_WHEN + W_WHEN + 0.04, DY),
            xytext=(DX + 0.50, DY),
            arrowprops=dict(arrowstyle='->', color=C['decision'],
                            lw=1.2, mutation_scale=9))

txt(ax, DX - 0.66, DY - 0.55, 'NO\n(skip)', fontsize=7, color='#888888')
ax.annotate('', xy=(DX - 0.35, DY - 0.84),
            xytext=(DX - 0.50, DY - 0.42),
            arrowprops=dict(arrowstyle='->', color='#AAAAAA',
                            lw=0.9, mutation_scale=8))

# Sticky note
fbox(ax, X_WHEN + 0.10, Y0 + 0.10, W_WHEN - 0.20, 0.38,
     fc='#FFFDE7', ec='#F0C050', lw=0.7, radius=0.06)
txt(ax, X_WHEN + W_WHEN/2, Y0 + 0.29,
    'Sticky: once triggered, stays ON',
    fontsize=6.5, color='#7D6608')

# ════════════════════════════════════════════════════════════════════════════
# ④ WHERE — Dual Attention Probe
# ════════════════════════════════════════════════════════════════════════════
arrow(ax, X_WHEN + W_WHEN + 0.04, ARROW_Y,
          X_WHERE - 0.04, ARROW_Y,
      color=C['decision'])

fbox(ax, X_WHERE, Y0, W_WHERE, PH,
     fc='#F5F0FA', ec='#7B5EA7', lw=1.1)
section_header(ax, X_WHERE, Y0 + PH - 0.28, W_WHERE,
               'WHERE: Dual Attention Probe', '#7B5EA7')

# ── Text probe sub-box ──────────────────────────────────────
fbox(ax, X_WHERE + 0.10, Y0 + 1.90, (W_WHERE - 0.30)/2, 1.05,
     fc='#FFF5E6', ec=C['probe_text'], lw=0.8, radius=0.08)
TPX = X_WHERE + 0.10 + (W_WHERE - 0.30)/4
txt(ax, TPX, Y0 + 2.72, 'Text Probe',
    fontsize=7.5, fontweight='bold', color=C['probe_text'])
txt(ax, TPX, Y0 + 2.44,
    'Cross-Attention\nMap (tokens→space)',
    fontsize=6.5, color='#5a3000', ha='center')
heatmap_block(ax, X_WHERE + 0.14, Y0 + 1.97, (W_WHERE - 0.30)/2 - 0.08, 0.35,
              cmap='YlOrBr', alpha=0.8)

# ── Image probe sub-box ─────────────────────────────────────
IX0 = X_WHERE + 0.20 + (W_WHERE - 0.30)/2
fbox(ax, IX0, Y0 + 1.90, (W_WHERE - 0.30)/2, 1.05,
     fc='#F0EAF8', ec=C['probe_img'], lw=0.8, radius=0.08)
IPX = IX0 + (W_WHERE - 0.30)/4
txt(ax, IPX, Y0 + 2.72, 'Image Probe',
    fontsize=7.5, fontweight='bold', color=C['probe_img'])
txt(ax, IPX, Y0 + 2.44,
    'Self-Attn vs.\nCLIP Exemplars',
    fontsize=6.5, color='#3b006e', ha='center')
heatmap_block(ax, IX0 + 0.04, Y0 + 1.97, (W_WHERE - 0.30)/2 - 0.08, 0.35,
              cmap='Purples', alpha=0.8)

# ── Merge arrow down to mask ─────────────────────────────────
PROBE_MID_X = X_WHERE + W_WHERE/2
ax.annotate('', xy=(PROBE_MID_X - 0.58, Y0 + 1.88),
            xytext=(TPX, Y0 + 1.90),
            arrowprops=dict(arrowstyle='->', color=C['probe_text'],
                            lw=0.9, mutation_scale=8,
                            connectionstyle='arc3,rad=0.3'))
ax.annotate('', xy=(PROBE_MID_X + 0.02, Y0 + 1.88),
            xytext=(IPX, Y0 + 1.90),
            arrowprops=dict(arrowstyle='->', color=C['probe_img'],
                            lw=0.9, mutation_scale=8,
                            connectionstyle='arc3,rad=-0.3'))

# Union label
txt(ax, PROBE_MID_X - 0.28, Y0 + 1.74,
    r'$\mathcal{M} = \mathcal{M}_{\rm text} \cup \mathcal{M}_{\rm img}$',
    fontsize=7.5, color=C['mask'])

# Spatial mask block
mask_block(ax, X_WHERE + 0.45, Y0 + 0.78, W_WHERE - 0.90, 0.82)
txt(ax, X_WHERE + W_WHERE/2, Y0 + 0.60,
    'Spatial Mask  $\\mathcal{M}$',
    fontsize=7.5, fontweight='bold', color=C['mask'])

# Exemplar callout box at bottom
fbox(ax, X_WHERE + 0.10, Y0 + 0.10, W_WHERE - 0.20, 0.40,
     fc='#FDEBD0', ec=C['mask'], lw=0.7, radius=0.06)
txt(ax, X_WHERE + W_WHERE/2, Y0 + 0.30,
    r'Family-Grouped Exemplars: 4 families $\times$ 4 prompts',
    fontsize=6.5, color='#7E5109')

# Probe dominance note
txt(ax, X_WHERE + W_WHERE/2, Y0 + 1.60,
    'Text probe dominates: nudity (keywords)\n'
    'Image probe dominates: violence, disturbing',
    fontsize=6.0, color='#555555', ha='center',
    style='italic')

# ════════════════════════════════════════════════════════════════════════════
# ⑤ HOW — Guided Denoising
# ════════════════════════════════════════════════════════════════════════════
arrow(ax, X_WHERE + W_WHERE + 0.04, ARROW_Y,
          X_HOW - 0.04, ARROW_Y,
      color='#7B5EA7')

fbox(ax, X_HOW, Y0, W_HOW, PH,
     fc=C['mask_light'], ec=C['mask'], lw=1.1)
section_header(ax, X_HOW, Y0 + PH - 0.28, W_HOW,
               'HOW: Guided Denoising', C['mask'])

# Formula box
fbox(ax, X_HOW + 0.08, Y0 + PH*0.53, W_HOW - 0.16, 1.12,
     fc='white', ec=C['mask'], lw=0.9, radius=0.08)
txt(ax, X_HOW + W_HOW/2, Y0 + PH*0.53 + 0.85,
    r'$\varepsilon_{\rm safe} = (1{-}\mathcal{M})\,\varepsilon_{\rm cfg}$',
    fontsize=8.0, color=C['neutral'])
txt(ax, X_HOW + W_HOW/2, Y0 + PH*0.53 + 0.55,
    r'$+\;\mathcal{M}\,[\varepsilon_{\rm cfg} - s\,(\varepsilon_{\rm t}{-}\varepsilon_{\rm a})]$',
    fontsize=8.0, color=C['unsafe'])
txt(ax, X_HOW + W_HOW/2, Y0 + PH*0.53 + 0.22,
    r'$\varepsilon_{\rm a}$: anchor (safe concept)',
    fontsize=6.5, color='#666666')

# Two outcome mini-boxes
fbox(ax, X_HOW + 0.08, Y0 + 1.50, (W_HOW - 0.24)/2, 0.68,
     fc=C['safe_light'], ec=C['safe'], lw=0.7, radius=0.07)
txt(ax, X_HOW + 0.08 + (W_HOW - 0.24)/4, Y0 + 1.84,
    'Safe\nregion', fontsize=7, color=C['safe'], ha='center')
txt(ax, X_HOW + 0.08 + (W_HOW - 0.24)/4, Y0 + 1.57,
    r'$\varepsilon_{\rm cfg}$', fontsize=7.5, color=C['safe'])

fbox(ax, X_HOW + 0.12 + (W_HOW - 0.24)/2, Y0 + 1.50,
     (W_HOW - 0.24)/2, 0.68,
     fc=C['unsafe_light'], ec=C['unsafe'], lw=0.7, radius=0.07)
txt(ax, X_HOW + 0.12 + (W_HOW - 0.24)/2 + (W_HOW - 0.24)/4, Y0 + 1.84,
    'Unsafe\nregion', fontsize=7, color=C['unsafe'], ha='center')
txt(ax, X_HOW + 0.12 + (W_HOW - 0.24)/2 + (W_HOW - 0.24)/4, Y0 + 1.57,
    r'$\varepsilon_{\rm a}$', fontsize=7.5, color=C['unsafe'])

# Anchor label
fbox(ax, X_HOW + 0.08, Y0 + 0.78, W_HOW - 0.16, 0.55,
     fc='#EBF5EB', ec=C['safe'], lw=0.7, radius=0.07)
txt(ax, X_HOW + W_HOW/2, Y0 + 1.07,
    'Anchor: "clothed person,\nfully dressed figure"',
    fontsize=6.5, color='#1a5c1a', ha='center')

fbox(ax, X_HOW + 0.08, Y0 + 0.10, W_HOW - 0.16, 0.52,
     fc='#FFFDE7', ec='#F0C050', lw=0.7, radius=0.06)
txt(ax, X_HOW + W_HOW/2, Y0 + 0.36,
    r'Inpaint masked regions with $\varepsilon_{\rm anchor}$',
    fontsize=6.5, color='#7D6608')

# ════════════════════════════════════════════════════════════════════════════
# ⑥ Output image
# ════════════════════════════════════════════════════════════════════════════
arrow(ax, X_HOW + W_HOW + 0.04, ARROW_Y,
          X_OUTPUT - 0.04, ARROW_Y,
      color=C['safe'])

fbox(ax, X_OUTPUT, Y0, W_OUTPUT, PH,
     fc=C['safe_light'], ec=C['safe'], lw=1.0)

# Fake safe image: blue sky + green bottom
np.random.seed(3)
img_data = np.zeros((30, 20, 3))
img_data[:15, :, 0] = 0.53; img_data[:15, :, 1] = 0.75; img_data[:15, :, 2] = 0.95
img_data[15:, :, 0] = 0.60; img_data[15:, :, 1] = 0.82; img_data[15:, :, 2] = 0.45
img_data += np.random.randn(30, 20, 3) * 0.04
img_data = np.clip(img_data, 0, 1)
ax.imshow(img_data, extent=[X_OUTPUT+0.08, X_OUTPUT+W_OUTPUT-0.08,
                              Y0+0.90, Y0+PH-0.45],
          origin='upper', aspect='auto', zorder=4)
ax.add_patch(plt.Rectangle((X_OUTPUT+0.08, Y0+0.90),
                             W_OUTPUT-0.16, PH-1.35,
                             fill=False, edgecolor=C['safe'], lw=0.7, zorder=5))

txt(ax, X_OUTPUT + W_OUTPUT/2, Y0 + PH*0.74 + 0.05,
    '', fontsize=8)  # placeholder

txt(ax, X_OUTPUT + W_OUTPUT/2, Y0 + 0.60,
    'Safe\nOutput', fontsize=8.5, fontweight='bold', color=C['safe'])

# Green check mark
txt(ax, X_OUTPUT + W_OUTPUT/2, Y0 + 0.28,
    '[OK] Unsafe concept\n      removed',
    fontsize=7, color=C['decision'])

# Output label above
section_header(ax, X_OUTPUT, Y0 + PH - 0.28, W_OUTPUT,
               'Safe Image', C['safe'])

# ════════════════════════════════════════════════════════════════════════════
# ⑦ Top-level stage labels (above panels)
# ════════════════════════════════════════════════════════════════════════════
LABEL_Y = Y0 + PH + 0.22
for xc, lbl in [
    (X_INPUT  + W_INPUT/2,  ''),
    (X_UNet   + W_UNet/2,   ''),
    (X_WHEN   + W_WHEN/2,   ''),
    (X_WHERE  + W_WHERE/2,  ''),
    (X_HOW    + W_HOW/2,    ''),
    (X_OUTPUT + W_OUTPUT/2, ''),
]:
    if lbl:
        txt(ax, xc, LABEL_Y, lbl, fontsize=9, fontweight='bold',
            color=C['neutral'])

# ════════════════════════════════════════════════════════════════════════════
# ⑧ Legend row (bottom strip)
# ════════════════════════════════════════════════════════════════════════════
LEG_Y = 0.18
items = [
    (C['safe'],      'Safe / Anchor concept'),
    (C['unsafe'],    'Unsafe concept'),
    (C['decision'],  'CAS Decision Gate'),
    (C['probe_text'],'Text Attention Probe'),
    (C['probe_img'], 'Image Attention Probe'),
    (C['mask'],      'Spatial Mask'),
]
total_w = 13.0
step = total_w / len(items)
for i, (col, label) in enumerate(items):
    lx = PAD + i * step + 0.05
    fbox(ax, lx, LEG_Y - 0.08, 0.22, 0.22, fc=col, ec=col, lw=0,
         radius=0.04, zorder=4)
    txt(ax, lx + 0.33, LEG_Y + 0.03, label,
        fontsize=6.5, ha='left', color=C['neutral'])

# ════════════════════════════════════════════════════════════════════════════
# ⑨ Overall title (optional — comment out if journal page has caption)
# ════════════════════════════════════════════════════════════════════════════
txt(ax, 7.5, 5.38,
    'SafeGen: Training-Free Safe Image Generation via Dual-Probe Spatial Guidance',
    fontsize=11, fontweight='bold', color=C['neutral'])

# ════════════════════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════════════════════
OUT = '/mnt/home3/yhgil99/unlearning/docs/figures/fig_method_overview.pdf'
fig.savefig(OUT, format='pdf', bbox_inches='tight', dpi=300,
            facecolor=C['bg'], edgecolor='none')
print(f'Saved: {OUT}')

# Also save a PNG preview
OUT_PNG = OUT.replace('.pdf', '.png')
fig.savefig(OUT_PNG, format='png', bbox_inches='tight', dpi=200,
            facecolor=C['bg'])
print(f'Saved preview: {OUT_PNG}')
