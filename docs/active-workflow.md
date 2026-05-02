# Active Workflow Guide

Use this as the canonical end-to-end path for current work in the repository.

## Canonical active workflow

### 1. Start from the active core

Prefer these directories unless the task explicitly targets a baseline or
legacy branch:

- `CAS_SpatialCFG/` — current method development
- `SAFREE/` — baseline plus local extensions
- `vlm/` — evaluation and aggregation
- `scripts/` — orchestration entrypoints

## Generation

For current training-free method work:

1. inspect or edit the target method in `CAS_SpatialCFG/`
2. use active orchestration scripts under:
   - `CAS_SpatialCFG/scripts/`
   - `scripts/`
3. keep outputs in ignored/generated artifact locations

## Evaluation

Use the shared evaluation surface in `vlm/`:

- `vlm/eval_nudenet.py`
- `vlm/opensource_vlm_i2p_all.py`
- `vlm/eval_vqascore_alignment.py`
- `vlm/eval_mjbench_safety.py`

Canonical active Qwen output names are:

- `categories_qwen_nudity.json`
- `results_qwen_nudity.txt`

## Aggregation

Prefer the shared aggregation and contract utilities in `vlm/` and `scripts/`,
including:

- `vlm/result_contract.py`
- `vlm/result_paths.py`
- `vlm/path_utils.py`
- `scripts/lib/repo_env.sh`

## Practical edit/test loop

1. inspect the active method or evaluation file
2. add or update a narrow regression test first when cleanup changes behavior
3. make the smallest scoped edit
4. run targeted verification:
   - `pytest -q`
   - `python -m py_compile ...`
   - `bash -n ...` for edited shell scripts
5. summarize changed files, simplifications, and remaining risks

## When not to use this workflow

If the task is explicitly about:

- historical reproduction
- archived classifier families
- older SDXL branches
- imported external projects

then follow the taxonomy in `docs/repo-taxonomy.md` and treat the active
workflow as non-authoritative for that task.

