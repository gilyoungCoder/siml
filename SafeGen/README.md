# SafeGen: Training-Free Safe Image Generation via Dual-Probe Spatial Guidance

Official implementation of **SafeGen**, a training-free method for removing unsafe concepts from text-to-image diffusion models using example-based When-Where-How guidance.

<p align="center">
  <img src="assets/method_overview.png" width="90%">
</p>

## Key Idea

SafeGen detects and redirects unsafe content generation **without any model fine-tuning**. It uses three complementary mechanisms:

| Component | Question | Mechanism |
|-----------|----------|-----------|
| **WHEN** | Is this prompt unsafe? | Global CAS (Concept Alignment Score) — cosine similarity between prompt and target concept noise directions, with sticky threshold |
| **WHERE** | Which spatial regions are unsafe? | Dual Cross-Attention Probe — text keywords + CLIP image exemplars probed in attention space (zero extra UNet calls) |
| **HOW** | How to redirect unsafe regions? | Anchor Inpaint — blend CFG output with anchor (safe) concept guidance in detected regions |

### Dual Probe Architecture

The core innovation is a **zero-cost spatial detection** mechanism:

- **Text Probe**: Pre-compute `K_text = W_K * CLIP_text("nudity", "naked", ...)` and measure cross-attention with the image query `Q` at each denoising step
- **Image Probe**: Pre-compute `K_img = W_K * CLIP_img(exemplar_images)` using CLIP visual features from exemplar unsafe images
- **Fusion**: Union of text and image probe attention maps → soft spatial mask
- **Cost**: Only one extra matmul per attention layer (piggybacked on existing forward pass)

## Installation

```bash
# Clone repository
git clone https://github.com/YounghwanKil/SafeGen.git
cd SafeGen

# Create conda environment
conda create -n safegen python=3.10 -y
conda activate safegen

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Concept Subspace (one-time, ~30 min)

```bash
# Generate exemplar images and compute concept directions
python -m safegen.prepare_concept_subspace \
    --output configs/exemplars/sexual/concept_directions.pt \
    --target_concepts "nudity" "nude person" "naked body" \
    --anchor_concepts "clothed person" "person wearing clothes"

# Extract CLIP exemplar embeddings for image probe
python -m safegen.prepare_clip_exemplar \
    --exemplar_dir configs/exemplars/sexual/images \
    --output configs/exemplars/sexual/clip_exemplar_projected.pt
```

### 2. Generate Safe Images

```bash
# Dual probe (text + image) — recommended
python -m safegen.generate \
    --prompts prompts/i2p_sexual.csv \
    --outdir outputs/safegen_sexual \
    --probe_mode both \
    --clip_embeddings configs/exemplars/sexual/clip_exemplar_projected.pt \
    --how_mode anchor_inpaint \
    --safety_scale 1.0 \
    --cas_threshold 0.6

# Text probe only (no exemplar images needed)
python -m safegen.generate \
    --prompts prompts/i2p_violence.csv \
    --outdir outputs/safegen_violence \
    --probe_mode text \
    --target_concepts "violence" "weapon" "blood" "gore" \
    --anchor_concepts "peaceful scene" "nature landscape" \
    --how_mode anchor_inpaint

# Baseline (standard SD v1.4, no safety)
python -m safegen.generate_baseline \
    --prompts prompts/i2p_sexual.csv \
    --outdir outputs/baseline_sexual
```

### 3. Evaluate

```bash
# NudeNet (nudity detection)
python -m evaluation.eval_nudenet outputs/safegen_sexual --threshold 0.5

# Q16 (general inappropriateness)
python -m evaluation.eval_q16 outputs/safegen_sexual --threshold 0.7

# FID + CLIP score (image quality)
python -m evaluation.eval_fid_clip outputs/baseline_sexual outputs/safegen_sexual prompts/i2p_sexual.txt
```

## Multi-Concept Support

SafeGen supports erasing multiple unsafe concepts simultaneously via concept packs:

```
configs/concept_packs/
├── sexual/          # Nudity, exposed body parts
├── violence/        # Blood, weapons, gore, combat
├── harassment/      # Threats, bullying, aggressive gestures
├── hate/            # Discrimination, hate symbols
├── shocking/        # Grotesque, body horror
├── illegal_activity/# Drugs, crime, theft
└── self-harm/       # Self-injury, suicide imagery
```

Each concept pack contains target/anchor keywords, prompts, and metadata:

```python
from safegen.concept_pack_loader import load_concept_pack

pack = load_concept_pack("configs/concept_packs/violence")
print(pack.target_concepts)  # ["blood", "weapon", "gore", ...]
print(pack.anchor_concepts)  # ["peaceful scene", "nature", ...]
print(pack.cas_threshold)    # 0.5
```

## Method Details

### WHEN: Concept Alignment Score (CAS)

At each denoising step $t$, compute:

$$\text{CAS}(t) = \cos\bigl(\epsilon_\text{prompt} - \epsilon_\varnothing,\; \epsilon_\text{target} - \epsilon_\varnothing\bigr)$$

If $\text{CAS}(t) > \tau$ (default $\tau = 0.6$), the step is flagged as unsafe. **Sticky mode**: once triggered, stays active for all remaining steps.

### WHERE: Dual Cross-Attention Probe

For each attention layer at resolution $r \in \{16, 32\}$:

1. **Pre-compute** target keys: $K_\text{target} = W_K \cdot c_\text{target}$ (one-time)
2. **Probe**: $A_\text{probe} = \text{softmax}\bigl(\frac{Q \cdot K_\text{target}^T}{\sqrt{d}}\bigr)$
3. **Aggregate**: avg over heads → max over tokens → upsample to 64 → avg over layers → normalize

Text and image probe masks are fused via **union** (element-wise max).

### HOW: Anchor Inpaint Guidance

$$\epsilon_\text{safe} = \epsilon_\text{cfg} \cdot (1 - s \cdot M) + \epsilon_\text{anchor\_cfg} \cdot (s \cdot M)$$

where $M$ is the spatial mask, $s$ is the safety scale, and $\epsilon_\text{anchor\_cfg}$ is the CFG output conditioned on the safe anchor concept.

## Benchmarks

### Nudity Erasing (4 Datasets)

| Method | Ring-A-Bell SR | MMA SR | P4DN SR | UnlearnDiff SR | COCO FID |
|--------|---------------|--------|---------|----------------|----------|
| SD v1.4 (Baseline) | 21.00% | 38.25% | 11.50% | 60.00% | — |
| SLD-Max | 87.00% | 80.50% | 72.00% | 86.00% | — |
| SAFREE | 94.00% | 79.75% | 91.00% | 90.00% | — |
| **SafeGen (Ours)** | **96.00%** | **91.75%** | **93.50%** | **96.00%** | **6.78** |

*SR = Safe Rate (higher is better). Evaluated with Qwen3-VL safety judge.*

### I2P Multi-Concept Erasing

| Concept | SD v1.4 | SLD-Max | SAFREE | **Ours** |
|---------|---------|---------|--------|----------|
| Sexual | 34.80% | 76.00% | 85.82% | **91.19%** |
| Violence | 47.62% | 68.78% | 73.02% | **79.10%** |
| Harassment | 30.58% | 48.91% | 50.00% | **54.85%** |
| Self-harm | 42.70% | 68.41% | 65.79% | **74.91%** |
| Illegal | 39.89% | 59.28% | 56.40% | **71.94%** |

## Project Structure

```
SafeGen/
├── safegen/                    # Core method
│   ├── generate.py             # Main generation script (dual-probe)
│   ├── generate_baseline.py    # SD v1.4 baseline
│   ├── attention_probe.py      # Cross-attention probing system
│   ├── concept_pack_loader.py  # Multi-concept configuration
│   ├── prepare_concept_subspace.py  # Offline concept direction computation
│   └── prepare_clip_exemplar.py     # CLIP exemplar embedding extraction
├── evaluation/                 # Evaluation metrics
│   ├── eval_nudenet.py         # NudeNet nudity detection
│   ├── eval_q16.py             # Q16 inappropriateness classifier
│   └── eval_fid_clip.py        # FID + CLIP score
├── configs/
│   └── concept_packs/          # Per-concept configurations (7 concepts)
├── prompts/                    # Benchmark prompt sets
│   ├── i2p_*.csv               # I2P category subsets
│   ├── ringabell.txt           # Ring-A-Bell benchmark
│   ├── mma.txt                 # MMA benchmark
│   └── coco_250.txt            # COCO benign prompts (FP check)
├── scripts/                    # Shell scripts for experiments
└── docs/                       # Additional documentation
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 11.8
- ~6 GB VRAM (SD v1.4 + probes)

## Citation

```bibtex
@article{safegen2026,
  title={SafeGen: Training-Free Safe Image Generation via Dual-Probe Spatial Guidance},
  author={Younghwan Kil},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) by CompVis
- [I2P Dataset](https://huggingface.co/datasets/AIML-TUDA/i2p) by Schramowski et al.
- [Q16 Classifier](https://github.com/ml-research/Q16) by Schramowski et al.
- [NudeNet](https://github.com/notAI-tech/NudeNet) by notAI-tech
- [SAFREE](https://github.com/jayjhaveri/SAFREE) baseline comparison
