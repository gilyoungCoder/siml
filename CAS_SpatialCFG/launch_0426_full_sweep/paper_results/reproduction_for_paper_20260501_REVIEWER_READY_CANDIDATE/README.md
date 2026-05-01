# Paper reproduction bundle (EBSG / Ours best)

Created: 2026-05-01
Source experiment root: internal cluster path removed; this bundle contains copied prompts/configs/exemplar packs/results needed for reproduction.

This folder is a compact, GitHub-style reproduction bundle for the paper tables. It contains:

- `code/SafeGen/`: generation code used for SD v1.4 Ours/EBSG.
- `configs/ours_best/i2p_q16/*.json`: per-concept **best** configs for I2P q16 top-60.
- `configs/ours_best/nudity/*.json`: nudity benchmark configs.
- `prompts/`: exact prompt files copied into the bundle.
- `exemplars/`: exact family exemplar packs (`clip_grouped.pt`) used by the configs.
- `scripts/run_i2p_best_all.sh`, `scripts/run_nudity_all.sh`: launch scripts.
- `scripts/eval_v5_outputs.sh`: Qwen3-VL v5 evaluation launcher.
- `results/`, `summaries/`: verified result files and table summaries.
- `samples/`: first 3 generated images per cell for quick visual sanity check.

## Environment

The cluster run used:

```bash
export PY_SAFGEN=/path/to/python
export PY_VLM=/path/to/vlm/python
export VLM_SCRIPT=/path/to/opensource_vlm_i2p_all_v5.py
```

## Run SD v1.4 I2P q16 top-60 Ours best

```bash
cd reproduction_for_paper_20260501_REVIEWER_READY_CANDIDATE
GPU=0 bash scripts/run_i2p_best_all.sh
GPU=0 bash scripts/eval_v5_outputs.sh
```

## Run nudity benchmarks

```bash
cd reproduction_for_paper_20260501_REVIEWER_READY_CANDIDATE
GPU=0 bash scripts/run_nudity_all.sh
```

## Verify config/family wiring

```bash
cd reproduction_for_paper_20260501_REVIEWER_READY_CANDIDATE
python scripts/verify_configs.py
```

Expected invariants:

- `how_mode=hybrid`
- `family_guidance=true`
- `probe_mode=both`
- SD v1.4: DDIM 50 / CFG 7.5 / seed 42 / 512
- per-concept family pack path points to `exemplars/i2p_v1/<concept>/clip_grouped.pt` except RAB nudity uses `exemplars/concepts_v2/sexual/clip_grouped.pt`, matching the original run.
