# Baseline Runner

Shared runner layer for the UCE / RECE / SLD baseline experiments. It keeps the method repos unchanged and standardizes the experiment surface:

- one canonical prompt CSV format
- unique image filenames based on `prompt_id`, not source `case_number`
- model defaults: SD v1.4 `DDIM 50 / CFG 7.5 / 512`, SD3 `28 / CFG 7.0 / 1024`, FLUX `28 / guidance 3.5 / 512`
- one image per prompt by default
- per-image runtime and NFE-equivalent logging in `manifest.csv`
- single-concept and multi-concept UCE/RECE erasure plans

## 1. Standardize Prompts

CSV input:

```bash
python -m baseline_runner.standardize_prompts \
  --input prompts/CAS_SpatialCFG/p4dn_16_prompt.csv \
  --output runs/prompts/p4dn_sd14.csv \
  --dataset p4d-n \
  --concept nudity \
  --prompt-col prompt
```

TXT input, such as I2P Q16 Top-60 split files:

```bash
python -m baseline_runner.standardize_prompts \
  --input prompts/CAS_SpatialCFG/i2p_q16_top60/violence_q16_top60.txt \
  --output runs/prompts/i2p_q16_top60_violence_sd14.csv \
  --dataset i2p-q16-top60 \
  --concept violence
```

The output schema includes:

```text
prompt_id,prompt,evaluation_seed,steps,guidance_scale,height,width,dataset,concept,source_path,source_row,source_prompt_col,source_case_number
```

`prompt_id` is always a new contiguous index, so duplicate source `case_number` values cannot overwrite images.

## 2. Prepare Erased Models

Single-concept UCE nudity erasure:

```bash
python -m baseline_runner.prepare_erasure \
  --method uce \
  --concept-set nudity \
  --tag nudity \
  --output-dir runs/erasure/uce/sd14/nudity
```

Multi-concept UCE erasure for the I2P Q16 Top-60 concepts:

```bash
python -m baseline_runner.prepare_erasure \
  --method uce \
  --concept-set i2p-q16-6 \
  --tag i2p_q16_6 \
  --output-dir runs/erasure/uce/sd14/i2p_q16_6
```

RECE from a UCE safetensors checkpoint:

```bash
python -m baseline_runner.prepare_erasure \
  --method rece \
  --concept-set nudity \
  --uce-weights runs/erasure/uce/sd14/nudity/single_nudity.safetensors \
  --output-dir runs/erasure/rece/sd14/nudity
```

By default this only writes `erasure_plan.json` and `commands.sh`. Add `--run` when you want to actually run the generated commands in the current environment.

For UCE, safety concepts should use `concept_type=unsafe` and guide to
unconditional text by default. With `expand_prompts=false` and blank guides this
is equivalent to the older `object` path, but `unsafe` is clearer for records.

RECE training uses `--skip_eval` in the helper scripts so unsafe/multi-concept
checkpoints can be produced without the local Q16 evaluation import path.

## 3. Dry-Run A Job

```bash
python -m baseline_runner.run \
  --method sld \
  --sld-variant SLD-Weak \
  --prompts runs/prompts/i2p_q16_top60_violence_sd14.csv \
  --output-dir runs/baselines/sd14/i2p_q16_top60/violence/SLD-Weak \
  --dry-run
```

## 4. Run Baselines

Ready-to-launch scripts for the current SD v1.4 main experiments:

```bash
# SLD-Weak/Medium/Strong/Max and UCE for P4D-N + I2P Q16 Top-60.
GPUS="0 1 2 3" bash scripts/run_sd14_sld_uce.sh

# RECE checkpoint preparation: UCE safetensors -> full UNet .pt -> RECE train.py.
GPU=0 bash scripts/prepare_rece_sd14_ckpts.sh
```

`scripts/run_sd14_sld_uce.sh` defaults to `UCE_NUDITY_TAG=nudity_scale1` and
I2P single-concept UCE weights from `/mnt/home2/jhpark/uce_models`. To run the
I2P UCE baseline with one multi-concept checkpoint instead, first create
`runs/erasure_plans/sd14/uce/i2p-q16-6/multi_erasure.safetensors`, then run:

```bash
UCE_I2P_MODE=multi GPUS="0 1 2 3" bash scripts/run_sd14_sld_uce.sh
```

RECE training uses `--skip_eval` so it saves epoch checkpoints without running
NudeNet/Q16 evaluation during training. This is intended for producing the
generation checkpoint first; evaluation should be run separately on generated
images.

Single-concept I2P UCE/RECE baselines use one checkpoint per Q16 Top-60 split:

```bash
# Prepare six single-concept RECE checkpoints from the matching UCE weights.
GPUS="0 1 2 3 4 5" bash scripts/prepare_i2p_single_rece_sd14_ckpts.sh

# After checkpoint preparation, generate UCE_single and RECE_single images.
GPUS="0 1 2 3 4 5" bash scripts/run_sd14_i2p_single_uce_rece.sh
```

The six concepts are `violence`, `self-harm`, `shocking`, `illegal activity`,
`harassment`, and `hate`. Existing UCE weights are read from
`/mnt/home2/jhpark/uce_models`; set `MAKE_MISSING_UCE=1` if a single-concept UCE
weight needs to be recreated.

Multi-concept I2P UCE/RECE baselines use one 6-concept checkpoint shared across
all six Q16 Top-60 splits:

```bash
# Prepare 6-concept UCE and RECE checkpoints.
RUN_NUDITY=0 RUN_I2P=1 GPU=0 bash scripts/prepare_rece_sd14_ckpts.sh

# Generate UCE_multi and RECE_multi images on all six splits.
GPUS="0 1 2 3 4 5" bash scripts/run_sd14_i2p_multi_uce_rece.sh
```

SD3 / FLUX I2P baselines use the same standardized-runner interface with
model-family defaults:

```bash
# Write SD3 and FLUX prompt CSVs with the requested per-model settings.
bash scripts/prepare_sd3_flux_prompts.sh

# Run base SD3 and base FLUX over the six I2P Q16 Top-60 splits.
GPUS="0 1 2 3 4 5" bash scripts/run_sd3_flux_i2p.sh
```

The default model ids are `stabilityai/stable-diffusion-3-medium-diffusers` and
`black-forest-labs/FLUX.1-dev`; override `SD3_MODEL_ID` or `FLUX_MODEL_ID` if the
local cache uses a different path.

FLUX UCE checkpoints can be prepared separately:

```bash
GPU=0 bash scripts/prepare_flux_i2p_uce_ckpts.sh
RUN_FLUX_UCE=1 FLUX_UCE_MODE=multi GPUS="0 1 2 3" bash scripts/run_sd3_flux_i2p.sh
```

SD3/FLUX RECE and SLD are not wired by default. RECE is currently SD v1.4 UNet
checkpoint based, while the local SLD implementation subclasses the SD v1.x
pipeline. The runner does support loading SD3/FLUX UCE-style transformer
`.safetensors` via `--method uce --model-family sd3|flux` when such weights are
available.

MJA uses four local prompt files, each with 100 prompts:

```text
mja_disturbing.txt
mja_illegal.txt
mja_sexual.txt
mja_violent.txt
```

Prompt preparation writes `runs/prompts/{sd3,flux}/mja/{concept}.csv`.

```bash
bash scripts/prepare_sd3_flux_prompts.sh
```

For MJA UCE, first prepare 4-concept checkpoints:

```bash
GPU=0 bash scripts/prepare_sd3_mja_uce_ckpts.sh
GPU=0 bash scripts/prepare_flux_mja_uce_ckpts.sh
```

Then run UCE-only generation:

```bash
RUN_SD3_BASE=0 RUN_FLUX_BASE=0 RUN_SD3_UCE=1 RUN_FLUX_UCE=1 \
  GPUS="0 1 2 3" bash scripts/run_sd3_flux_mja.sh
```

Base SD:

```bash
python -m baseline_runner.run \
  --method sd \
  --prompts runs/prompts/p4dn_sd14.csv \
  --output-dir runs/baselines/sd14/p4d-n/SD
```

UCE:

```bash
python -m baseline_runner.run \
  --method uce \
  --uce-weights path/to/uce_nudity.safetensors \
  --prompts runs/prompts/p4dn_sd14.csv \
  --output-dir runs/baselines/sd14/p4d-n/UCE
```

RECE:

```bash
python -m baseline_runner.run \
  --method rece \
  --rece-ckpt path/to/rece_unet.pt \
  --prompts runs/prompts/p4dn_sd14.csv \
  --output-dir runs/baselines/sd14/p4d-n/RECE
```

SLD:

```bash
python -m baseline_runner.run \
  --method sld \
  --sld-variant SLD-Strong \
  --prompts runs/prompts/p4dn_sd14.csv \
  --output-dir runs/baselines/sd14/p4d-n/SLD-Strong
```

The SLD implementation path used by default is:

```text
/mnt/home3/datasets/jhpark/safety_experiments/safe-latent-diffusion
```

## 5. Outputs

Each run writes:

```text
run_config.json
manifest.csv
images/000000_0.png
images/000001_0.png
...
```

`manifest.csv` records `runtime_seconds`, `batched_unet_calls`, and `condition_forward_equivalents`. For SD v1.x/SD3 CFG generation, SD/UCE/RECE count as `2 * steps`; SLD counts as `3 * steps` because it adds the safety concept branch. FLUX `guidance_scale` is not true CFG in the default path, so it counts as `1 * steps` unless `--true-cfg-scale > 1` is set.
