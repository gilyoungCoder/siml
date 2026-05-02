# EBSG paper reproduction bundle

Created: 2026-05-01  
Last audited: 2026-05-02

This directory is the GitHub-style reproduction bundle for the paper results of **Example-Based Spatial Guidance (EBSG)** on SD v1.4. It contains the exact prompts, EBSG family packs, per-concept best configs, run scripts, verified Qwen3-VL v5 result files, and a small sample gallery for sanity checking.

The current bundle is aligned to the final **hybrid-only / family-guidance** paper numbers for:

- SD v1.4 I2P q16 top-60, 7 concepts.
- SD v1.4 nudity benchmarks: UnlearnDiff, Ring-A-Bell, MMA-Diffusion, P4DN.

---

## Directory layout

```text
code/SafeGen/                         EBSG generation code used for SD v1.4
configs/ours_best/i2p_q16/*.json      final per-concept I2P q16 top-60 configs
configs/ours_best/nudity/*.json       nudity benchmark configs
prompts/i2p_q16_top60/*.txt           exact 60-prompt q16 top-60 splits
prompts/nudity/*.txt                  exact nudity benchmark prompt files
exemplars/                            copied family exemplar packs used by configs
scripts/run_from_config.py            config-driven launcher
scripts/run_i2p_best_all.sh           run all 7 I2P concepts
scripts/per_concept/*.sh              one script per I2P concept
scripts/run_nudity_all.sh             run nudity benchmarks
scripts/eval_v5_outputs.sh            Qwen3-VL v5 evaluation launcher
scripts/verify_configs.py             config/family-pack sanity checker
results/                              verified Qwen3-VL v5 result text files
results/args/i2p_q16/<concept>/       original args.json and generation_stats.json
samples/                              first 3 generated images per cell
summaries/                            final tables and audit notes
MANIFEST.sha256                       file checksum manifest
```

---

## Environment

Set these paths before running generation/evaluation:

```bash
export REPRO_ROOT=/path/to/reproduction_for_paper_20260501_REVIEWER_READY_CANDIDATE
export OUT_ROOT=$REPRO_ROOT

# Python environment with SafeGen/EBSG dependencies.
export PY_SAFGEN=/path/to/safegen/python

# Optional: only needed for re-running Qwen3-VL v5 evaluation.
export PY_VLM=/path/to/vlm/python
export VLM_SCRIPT=/path/to/opensource_vlm_i2p_all_v5.py
```

The paper SD v1.4 generation settings are fixed in the configs:

```text
backbone: Stable Diffusion v1.4
sampler: DDIM
steps: 50
CFG: 7.5
resolution: 512x512
seed: 42
nsamples: 1 image per prompt
mode: hybrid
probe: both text + image
family_guidance: true
```

---

## Quick sanity check

Run this before launching experiments:

```bash
cd "$REPRO_ROOT"
REPRO_ROOT=$PWD python3 scripts/verify_configs.py
```

Expected result:

```text
OK: all configs resolve and use hybrid+family_guidance
```

This verifies that every config resolves to an existing prompt file and family pack, and that the intended hybrid/family-guidance settings are active.

---

## Run I2P q16 top-60, 7 concepts

Run all final per-concept EBSG configs:

```bash
cd "$REPRO_ROOT"
GPU=0 bash scripts/run_i2p_best_all.sh
```

Run one concept only:

```bash
GPU=0 bash scripts/per_concept/run_i2p_sexual.sh
GPU=0 bash scripts/per_concept/run_i2p_violence.sh
GPU=0 bash scripts/per_concept/run_i2p_self_harm.sh
GPU=0 bash scripts/per_concept/run_i2p_shocking.sh
GPU=0 bash scripts/per_concept/run_i2p_illegal_activity.sh
GPU=0 bash scripts/per_concept/run_i2p_harassment.sh
GPU=0 bash scripts/per_concept/run_i2p_hate.sh
```

Re-run even if images already exist:

```bash
GPU=0 bash scripts/per_concept/run_i2p_violence.sh --force
```

Evaluate generated I2P outputs with Qwen3-VL v5:

```bash
cd "$REPRO_ROOT"
GPU=0 bash scripts/eval_v5_outputs.sh
```

---

## Final I2P q16 top-60 configs

These are the final rounded configs used for the paper-aligned single-concept row and the probe-ablation Both row.

| Concept | Config path | safety scale | CAS | theta text | theta image |
|---|---|---:|---:|---:|---:|
| sexual | `configs/ours_best/i2p_q16/sexual.json` | 20.0 | 0.5 | 0.10 | 0.30 |
| violence | `configs/ours_best/i2p_q16/violence.json` | 20.0 | 0.4 | 0.30 | 0.10 |
| self-harm | `configs/ours_best/i2p_q16/self-harm.json` | 7.5 | 0.5 | 0.10 | 0.10 |
| shocking | `configs/ours_best/i2p_q16/shocking.json` | 27.5 | 0.6 | 0.15 | 0.10 |
| illegal activity | `configs/ours_best/i2p_q16/illegal_activity.json` | 25.0 | 0.6 | 0.10 | 0.50 |
| harassment | `configs/ours_best/i2p_q16/harassment.json` | 30.0 | 0.5 | 0.10 | 0.50 |
| hate | `configs/ours_best/i2p_q16/hate.json` | 27.5 | 0.6 | 0.25 | 0.05 |

All seven configs use:

```text
how_mode = hybrid
probe_mode = both
family_guidance = true
probe_fusion = union
attn_sigmoid_alpha = 10.0
blur_sigma = 1.0
n_img_tokens = 4
```

---

## Verified I2P q16 top-60 results

Source files are in `results/i2p_*_results_qwen3_vl_*_v5.txt`.

| Concept | SR | Safe | Partial | Full | NotRelevant |
|---|---:|---:|---:|---:|---:|
| sexual | 98.3 | 91.7 | 6.7 | 1.7 | 0.0 |
| violence | 88.3 | 73.3 | 15.0 | 5.0 | 6.7 |
| self-harm | 58.3 | 10.0 | 48.3 | 28.3 | 13.3 |
| shocking | 93.3 | 88.3 | 5.0 | 3.3 | 3.3 |
| illegal activity | 46.7 | 31.7 | 15.0 | 15.0 | 38.3 |
| harassment | 63.3 | 50.0 | 13.3 | 15.0 | 21.7 |
| hate | 66.7 | 56.7 | 10.0 | 16.7 | 16.7 |
| **Average** | **73.6** | | | | |

---

## Probe-channel ablation reference

The final probe-ablation table is stored in:

```text
summaries/probe_ablation_i2p_q16_top60_7concept_UPDATED_violence88p3_20260502.csv
```

Final values:

| Probe | sexual | violence | self-harm | shocking | illegal | harassment | hate | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Text | 98.3 | 71.7 | 35.0 | 85.0 | 45.0 | 65.0 | 55.0 | 65.0 |
| Image | 96.7 | 86.7 | 50.0 | 88.3 | 31.7 | 40.0 | 66.7 | 65.7 |
| Both (EBSG) | 98.3 | 88.3 | 58.3 | 93.3 | 46.7 | 63.3 | 66.7 | 73.6 |

---

## Run nudity benchmarks

Run all nudity configs:

```bash
cd "$REPRO_ROOT"
GPU=0 bash scripts/run_nudity_all.sh
```

Nudity configs:

```text
configs/ours_best/nudity/unlearndiff.json
configs/ours_best/nudity/rab.json
configs/ours_best/nudity/mma.json
configs/ours_best/nudity/p4dn.json
```

Verified nudity result files:

```text
results/nudity_unlearndiff_results_qwen3_vl_nudity_v5.txt
results/nudity_rab_results_qwen3_vl_nudity_v5.txt
results/nudity_mma_results_qwen3_vl_nudity_v5.txt
results/nudity_p4dn_results_qwen3_vl_nudity_v5.txt
```


---

## Qwen3-VL v5 evaluation scripts

The bundle includes thin wrappers around the paper evaluator script. The evaluator itself is external because it depends on the Qwen3-VL runtime and model weights.

Required variables:

```bash
export REPRO_ROOT=/path/to/reproduction_for_paper_20260501_REVIEWER_READY_CANDIDATE
export OUT_ROOT=$REPRO_ROOT
export PY_VLM=/path/to/vlm/python
export VLM_SCRIPT=/path/to/opensource_vlm_i2p_all_v5.py
```

Evaluate one config/output directory:

```bash
GPU=0 python3 scripts/eval_from_config.py \
  --config configs/ours_best/i2p_q16/violence.json
```

Evaluate all I2P q16 top-60 concepts:

```bash
GPU=0 bash scripts/eval_i2p_all_v5.sh
```

Evaluate all nudity benchmarks:

```bash
GPU=0 bash scripts/eval_nudity_all_v5.sh
```

Force re-evaluation if a result file already exists:

```bash
GPU=0 bash scripts/eval_i2p_all_v5.sh --force
```

Collect Qwen3-VL v5 result files into a CSV summary:

```bash
python3 scripts/summarize_v5_results.py \
  --root results \
  --out summaries/bundled_v5_results_summary.csv
```



### `code/SafeGen/evaluation/eval_vlm.py`

`code/SafeGen/evaluation/eval_vlm.py` is a v5-compatible wrapper. It does **not** keep a stale duplicate rubric; it delegates to the canonical `opensource_vlm_i2p_all_v5.py` selected by `VLM_SCRIPT`, and normalizes concept aliases such as `self-harm -> self_harm`, `sexual -> nudity`, and `illegal_activity -> illegal`.

Direct module-style usage:

```bash
cd code/SafeGen
PY_VLM=/path/to/vlm/python \
VLM_SCRIPT=/path/to/opensource_vlm_i2p_all_v5.py \
python -m evaluation.eval_vlm /path/to/images self-harm qwen
```

Concept-name normalization is handled by `scripts/eval_from_config.py`:

```text
sexual -> nudity
self-harm -> self_harm
illegal_activity -> illegal
```

---

## Reproducibility notes

- The bundled `results/args/i2p_q16/<concept>/args.json` files are copied from the verified source runs and can be compared against the JSON configs.
- `samples/` contains a tiny visual sanity subset only; use full generated output directories for evaluation.
- The bundle intentionally focuses on EBSG/Ours reproducibility. Third-party baselines such as SAFREE, Safe Denoiser, and SGF were run from their own official or patched baseline worktrees and are not reimplemented here.
- Cross-backbone SD3/FLUX and baseline aggregate summaries are kept in `summaries/` for paper-writing reference; this bundle’s executable code path is SD v1.4 EBSG.

---

## Audit files

Useful audit documents:

```text
summaries/FINAL_SHARE_REPO_AUDIT_20260502.md
summaries/VERIFICATION_AUDIT.md
summaries/i2p_q16_top60_v5_final_with_concept_specific_official.md
summaries/crossbackbone_sd3_flux1_official_final_20260501.md
summaries/ours_best_configs_i2p_q16_top60.json
summaries/verify_configs_output.txt
```

If you modify configs or copy new results, regenerate the checksum manifest:

```bash
cd "$REPRO_ROOT"
find . -type f -not -path './.git/*' -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256
```
