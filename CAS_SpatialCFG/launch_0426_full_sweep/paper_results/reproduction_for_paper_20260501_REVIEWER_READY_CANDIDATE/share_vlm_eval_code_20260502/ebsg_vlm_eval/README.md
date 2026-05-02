# EBSG Qwen3-VL v5 evaluation code

This package contains the VLM evaluation code used for EBSG paper result tables.
It is intentionally self-contained for the evaluator logic: the wrapper scripts and the canonical v5 evaluator dependency chain are included together.

## Contents

```text
scripts/eval_from_config.py          Evaluate one output directory from a JSON config
scripts/eval_i2p_all_v5.sh           Evaluate all I2P q16 top-60 EBSG outputs if configs exist
scripts/eval_nudity_all_v5.sh        Evaluate all nudity benchmark outputs if configs exist
scripts/eval_v5_outputs.sh           Backward-compatible alias for I2P eval
scripts/summarize_v5_results.py      Collect *_v5.txt result files into CSV
code/SafeGen/evaluation/eval_vlm.py  Simple wrapper around canonical v5 evaluator
vlm/opensource_vlm_i2p_all_v5.py     Canonical v5 rubric/evaluator entrypoint
vlm/opensource_vlm_i2p_all_v4.py     Dependency of v5
vlm/opensource_vlm_i2p_all_v3.py     Dependency of v4/v5
vlm/opensource_vlm_i2p_all_v2.py     Base evaluator/model-loading implementation
vlm/result_paths.py                  Output filename helper
examples/bundled_v5_results_summary.csv  Example output summary format
```

## Environment

Use a Python environment with the Qwen3-VL/Qwen-VL dependencies installed. The original run used a dedicated VLM environment.

Typical packages:

```bash
pip install torch transformers accelerate pillow tqdm qwen-vl-utils
```

Depending on the Qwen3-VL checkpoint/version, you may also need the correct `transformers` version recommended by Qwen.

## Quick start: evaluate one image folder

```bash
tar -xzvf ebsg_vlm_eval_code_20260502.tar.gz
cd ebsg_vlm_eval

export PY_VLM=$(which python)
export VLM_SCRIPT=$PWD/vlm/opensource_vlm_i2p_all_v5.py

# Usage: <image_dir> <concept> [model]
python code/SafeGen/evaluation/eval_vlm.py /path/to/generated_images violence qwen
```

The output files are written inside `/path/to/generated_images`:

```text
categories_qwen3_vl_<concept>_v5.json
results_qwen3_vl_<concept>_v5.txt
```

## Supported concepts

Use these evaluator concept names:

```text
nudity
violence
harassment
hate
shocking
disturbing
illegal
self_harm
```

The wrapper also accepts these aliases:

```text
sexual -> nudity
self-harm -> self_harm
illegal_activity -> illegal
```

## Direct canonical v5 usage

You can also call the canonical evaluator directly:

```bash
python vlm/opensource_vlm_i2p_all_v5.py /path/to/generated_images violence qwen
python vlm/opensource_vlm_i2p_all_v5.py /path/to/generated_images self_harm qwen
python vlm/opensource_vlm_i2p_all_v5.py /path/to/generated_images nudity qwen
```

## Summarize result files

After running evaluations, collect result text files into one CSV:

```bash
python scripts/summarize_v5_results.py \
  --root /path/to/output_root \
  --out vlm_eval_summary.csv
```

CSV columns:

```text
concept_or_rubric,result_file,outdir,SR,Safe,Partial,Full,NotRelevant
```

## Config-based usage, optional

If you also have the EBSG reproduction configs, you can evaluate by config:

```bash
export REPRO_ROOT=/path/to/reproduction_bundle
export OUT_ROOT=$REPRO_ROOT
export PY_VLM=$(which python)
export VLM_SCRIPT=$PWD/vlm/opensource_vlm_i2p_all_v5.py

GPU=0 python scripts/eval_from_config.py \
  --config $REPRO_ROOT/configs/ours_best/i2p_q16/violence.json
```

For all I2P configs:

```bash
GPU=0 bash scripts/eval_i2p_all_v5.sh
```

For all nudity configs:

```bash
GPU=0 bash scripts/eval_nudity_all_v5.sh
```

## Notes

- This package contains evaluation code only, not generated images or model weights.
- The v5 rubric is the one used for the final EBSG paper numbers.
- `eval_vlm.py` is a wrapper; it delegates to `vlm/opensource_vlm_i2p_all_v5.py` through `VLM_SCRIPT` so there is no stale duplicate rubric.
