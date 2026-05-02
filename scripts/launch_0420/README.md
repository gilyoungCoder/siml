# launch_0420 — Per-GPU Queue Dispatch System

## Overview

15 workers across 4 servers, each running a nohup queue of generation + VLM eval jobs.

| Backbone | Servers | GPUs | Workers | Jobs/worker |
|---|---|---|---|---|
| SD1.4 | siml-01 | 0-7 | 8 | ~18 |
| SD3 | siml-06 | 4-7 | 4 | ~36 |
| FLUX1 | siml-08,siml-09 | 4,5 / 0 | 3 | ~19 |

**Total: ~342 generation jobs × 2 eval calls = ~684 eval invocations**

## How to Launch

```bash
# From siml-09 (or any host with SSH access to all servers):
cd /mnt/home3/yhgil99/unlearning
bash scripts/launch_0420/launch_all.sh
```

launch_all.sh does:
1. GPU pre-check (abort if foreign process >1000MB)
2. Dataset file verification
3. Concept pack verification
4. Build job manifests (15 CSV files in `manifests/`)
5. Smoke tests (1 prompt, 14 steps) for SD1.4/SD3/FLUX1
6. Launch 15 nohup workers, print PIDs

**If ANY pre-check or smoke test fails, the entire launch is aborted.**

## How to Check Status

```bash
bash scripts/launch_0420/status_all.sh
```

Prints a 15-line table: HOST | GPU | TOTAL | DONE | SKIPPED | FAILED | STATUS

## How to Abort One Worker

```bash
# Find the PID on the target host
ssh siml-01 "pgrep -f 'worker.sh siml-01 3'"

# Kill it
ssh siml-01 "pkill -f 'worker.sh siml-01 3'"

# The current generation job may still be running — kill it too if needed:
ssh siml-01 "nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 3"
ssh siml-01 "kill <pid>"
```

The lock file `<output_dir>/.lock` is cleaned up automatically on exit. If the worker
was killed mid-generation, the lock file may remain — remove it manually:
```bash
ssh siml-01 "find /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0420 -name '.lock' | xargs rm -f"
```

## Idempotency / Resume

Each worker checks before running each job:
- **Skip generation** if `output_dir` exists with ≥ N expected images
- **Skip eval_v2** if `categories_qwen3_vl_<concept>.json` exists and is non-empty
- **Skip eval_v3** if `.eval_v3_<concept>.done` sentinel exists

To resume a crashed worker:
```bash
ssh siml-01 "cd /mnt/home3/yhgil99/unlearning && nohup bash scripts/launch_0420/worker.sh siml-01 3 >> logs/launch_0420/worker_siml-01_g3.out 2>&1 &"
```

## Output Structure

```
CAS_SpatialCFG/outputs/launch_0420/
  baseline_sd14/<dataset>/        *.png + stats.json
  safree_sd14/<dataset>/          *.png
  ours_sd14/<dataset>/cas0.6_ss<ss>_thr<thr>_<how>_both/
  baseline_sd3/...
  safree_sd3/...
  ours_sd3/...
  baseline_flux1/...
  safree_flux1/...
  ours_flux1/...
```

Each output dir also contains (after eval):
- `categories_qwen3_vl_<concept>.json` — per-image classifications (v2)
- `results_qwen3_vl_<concept>.txt` — summary stats (v2)
- `.eval_v3_<concept>.done` — sentinel for v3 completion

## Logs

```
logs/launch_0420/
  launch_all_YYYYMMDD_HHMMSS.log   — orchestrator log
  worker_<host>_g<gpu>.log         — per-worker detailed log
  worker_<host>_g<gpu>.out         — nohup stdout/stderr
```

## GPU Constraints (DO NOT MODIFY)

- siml-06: ONLY GPU 4,5,6,7 — 0/1/2/3 belong to other users
- siml-08: ONLY GPU 4,5 — 0(giung2)/6(46GB)/7(16GB) are in use
- siml-09: ONLY GPU 0 (H100 97GB)
