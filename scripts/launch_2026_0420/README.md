# Phase 1 Baseline Launch — 2026-04-20

## GPU Policy (HARD ENFORCEMENT — DO NOT VIOLATE)

| Host    | Allowed GPUs | Notes                                      |
|---------|--------------|--------------------------------------------|
| siml-01 | 0 1 2 3 4 5 6 7 | All yhgil99                             |
| siml-06 | 4 5 6 7      | GPUs 0,1,2,3 FORBIDDEN (used by others)   |
| siml-08 | 4 5          | GPU 0=giung2, GPU 6,7 FORBIDDEN           |
| siml-09 | 0            | H100, yhgil99 only                        |

Rules:
- Always `CUDA_VISIBLE_DEVICES=<single_integer>` — never a comma list
- Always `--device cuda:0` (CUDA_VISIBLE_DEVICES remaps physical GPU to index 0)
- Pre-launch: abort if used_memory > 1000 MB on any target GPU

## Backbone → Server mapping

| Backbone | Host(s)        | GPUs   | VRAM      |
|----------|----------------|--------|-----------|
| SD1.4    | siml-01        | 0–4    | 24 GB each|
| SD3      | siml-06        | 4,5,6,7| 49 GB A6000|
| FLUX1    | siml-09        | 0      | 97 GB H100|
|          | siml-08        | 4, 5   | 49 GB A6000|

## Files

- `gpu_policy.sh` — source this in all dispatch scripts; defines GPU arrays and `check_gpu_free`
- `phase1_baselines.sh` — launches 15 baseline jobs (3 backbone × 5 dataset)
- `monitor_phase1.sh` — on-demand status: running processes + completed image counts

## Launch Procedure

```bash
# 1. SSH to any host with access to the shared filesystem
ssh siml-01

# 2. Run the launch script (performs GPU checks then dispatches)
cd /mnt/home3/yhgil99/unlearning/scripts/launch_2026_0420
bash phase1_baselines.sh

# 3. Monitor status (run any time after launch)
bash monitor_phase1.sh
```

## Job Distribution (15 jobs)

### SD1.4 — siml-01 (5 jobs, GPUs 0–4)
| GPU | Dataset       | Prompts |
|-----|---------------|---------|
| 0   | mja_sexual    | 100     |
| 1   | mja_violent   | 100     |
| 2   | mja_disturbing| 100     |
| 3   | mja_illegal   | 100     |
| 4   | rab           | 79      |

### SD3 — siml-06 (5 jobs, GPUs 4–7)
| GPU | Dataset        | Notes                        |
|-----|----------------|------------------------------|
| 4   | mja_sexual     | concurrent                   |
| 5   | mja_violent    | concurrent                   |
| 6   | mja_disturbing | concurrent                   |
| 7   | mja_illegal    | concurrent                   |
| 4   | rab            | sequential after mja_sexual  |

### FLUX1 — siml-09 + siml-08 (5 jobs)
| Host    | GPU | Dataset        | Notes                           |
|---------|-----|----------------|---------------------------------|
| siml-09 | 0   | mja_sexual     | H100, concurrent slot 1         |
| siml-09 | 0   | mja_violent    | H100, sequential after sexual   |
| siml-09 | 0   | rab            | H100, sequential 3rd            |
| siml-08 | 4   | mja_disturbing | A6000, concurrent               |
| siml-08 | 5   | mja_illegal    | A6000, concurrent               |

Note: `generate_flux1_v1.py` always uses `enable_model_cpu_offload`. This is safe on
both H100 (97 GB) and A6000 (49 GB) at 1024×1024.

## Log paths

All logs: `/mnt/home3/yhgil99/unlearning/logs/launch_0420/`  
Pattern: `baseline_<backbone>_<dataset>_<host>g<gpu>.log`

## Output dirs

```
CAS_SpatialCFG/outputs/launch_0420/
  baseline_sd14/<dataset>/
  baseline_sd3/<dataset>/
  baseline_flux1/<dataset>/
```
