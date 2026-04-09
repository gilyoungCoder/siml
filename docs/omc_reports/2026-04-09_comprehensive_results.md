# 2026-04-09 Comprehensive Results Report

## 1. Method: v27 Dual-Probe Safe Generation

**Architecture:**
- WHEN: Global Noise CAS (threshold=0.6, sticky)
- WHERE: Dual Cross-Attention Probe (text + image exemplar, union)
  - text_threshold=0.1, img_threshold=0.4
  - Auto target_words extraction per concept
- HOW: anchor_inpaint (best) / hybrid (ts=15, as=15)
- Image exemplars: SD-generated, 16 target + 16 anchor per concept (data-free)

## 2. Nudity Results (Qwen3-VL, nsamples=3)

| Dataset | both ainp SR% | text ainp SR% | both hybrid SR% | Full% | n |
|---------|-------------|-------------|----------------|-------|---|
| UnlearnDiff | **95.8%** | 95.8% | - | 2.1% | 142 |
| Ring-A-Bell | 90.3% | 90.7% | **94.1%** | 2.5% | 237 |
| P4DN | 92.7% | **93.4%** | - | 4.6-5.3% | 151 |
| I2P Sexual | **91.7%** | 91.6% | - | 3.0% | 931 |
| MMA | 78.2% | 78.3% | - | 18.9% | 1000 |
| **COCO FP** | Full=**0.0%** | | | | 250 |

## 3. Safety Concepts (Qwen3-VL)

| Concept | Best Config | SR% | Full% | n |
|---------|------------|-----|-------|---|
| Violence | both ainp ss=1.5 cas=0.4 | **97.0%** | 1.1% | 756 |
| Shocking | both ainp ss=0.8 cas=0.6 | **97.3%** | 2.3% | 856 |
| Harassment | both ainp ss=1.0 cas=0.4 | **82.8%** | 10.3% | 824 |
| Hate | multi_nude_hate ss=1.0 | 84.1% | 8.7% | 69 |

## 4. Artist Style Removal (Qwen3-VL 3-class eval)

Eval: OtherArt(성공) / Style(스타일 잔여=실패) / NotPainting(정합성 깨짐=실패)

| Artist | Best OtherArt% | Style% | NotPainting% | Config |
|--------|---------------|--------|-------------|--------|
| Picasso | **100%** | 0% | 0% | hybrid ts=15 |
| Warhol | **100%** | 0% | 0% | hybrid ts=20 |
| Van Gogh | **98.3%** | 1.7% | 0% | hybrid ts=10 |
| Rembrandt | **98.3%** | 1.7% | 0% | ainp ss=0.8 |
| Monet | **83.3%** | 16.7% | 0% | ainp ss=1.0 |
| Multi (3 artists) | 73.3% | 26.7% | 0% | ss=1.0 |

## 5. Multi-Concept Erasing

| Config | Dataset | SR% | NR% | Full% |
|--------|---------|-----|-----|-------|
| nude+violence ss=1.0 cas=0.4 | I2P Sexual | 87.9% | 7.3% | 4.8% |
| nude+violence ss=1.0 cas=0.5 | I2P Sexual | 87.0% | 7.9% | 5.0% |
| all_unsafe ss=1.0 | I2P Sexual | 87.0% | 1.9% | 11.1% |

## 6. Evaluation Validation

### 6.1 MJ-Bench (Qwen3-VL Judge Accuracy)
- **Dataset**: MJ-Bench safety subset (672 pairwise image pairs)
- **Metric**: Average accuracy (with/without ties), same as MJ-Bench leaderboard
- **Our Qwen3-VL-8B: 68.9%** (97.5% excl. ties, 40.3% incl. ties)
- **GPT-4o (paper): 63.9%** → Our Qwen3-VL **surpasses GPT-4o by +5.0pp**

| Model | Safety Accuracy |
|-------|----------------|
| **Our Qwen3-VL-8B** | **68.9%** |
| GPT-4o | 63.9% |
| GPT-4-vision | 63.2% |
| Claude 3 Opus | 60.7% |
| Gemini Ultra | 59.3% |
| Qwen-VL-Chat (v1) | 33.5% |

### 6.2 MJ-Bench 4-Class Alignment
Our 4-class evaluation aligns with MJ-Bench pairwise ground truth:

| MJ-Bench Role | Our Full% | Our Partial% | Our Safe% | Our NotRel% |
|---------------|----------|-------------|----------|------------|
| more_unsafe | **71.6%** | 16.5% | 9.7% | 2.2% |
| safer | 28.7% | **43.9%** | **24.7%** | 2.7% |

→ "more unsafe" images correctly classified as Full (71.6%)
→ "safer" images correctly classified as Partial+Safe (68.6%)

### 6.3 VQAScore Alignment (anchor_strict.csv, 33 prompts × 3 samples)

| Config | VQA(original)↓ | VQA(anchor)↑ | Gap↑ |
|--------|---------------|-------------|------|
| SD Baseline | 0.452 | 0.385 | **-0.067** |
| both ainp ss=1.0 | 0.316 | 0.321 | **+0.005** |
| both ainp ss=1.2 | 0.295 | 0.308 | **+0.013** |
| both hybrid ts=15 | 0.329 | 0.369 | **+0.040** |
| text ainp ss=1.2 | 0.294 | 0.308 | **+0.014** |

→ Baseline: harmful prompt에 가까움 (gap 음수)
→ Our method: safe anchor에 가까움 (gap 양수)

### 6.4 LLaVA Cross-Validation
(진행중 — siml-07 GPU 0)
Qwen3-VL과 LLaVA-NeXT 분류 일치율 비교 예정.

## 7. Files Created

| File | Description |
|------|------------|
| `CAS_SpatialCFG/generate_v27.py` | Main generation code (dual-probe) |
| `CAS_SpatialCFG/prepare_artist_exemplars.py` | Artist exemplar generation |
| `CAS_SpatialCFG/prepare_concept_exemplars.py` | Safety concept exemplar generation |
| `vlm/eval_mjbench_safety.py` | MJ-Bench pairwise eval |
| `vlm/eval_mjbench_4class.py` | MJ-Bench 4-class alignment analysis |
| `vlm/eval_vqascore_alignment.py` | VQAScore 3-way alignment eval |
| `vlm/opensource_vlm_i2p_all.py` | Qwen3-VL + LLaVA eval (all concepts + artists) |
