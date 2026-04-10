# OMC Status Summary — 2026-04-10 04:40 KST

## Current server state
- siml-01: all GPUs idle at last check.
- siml-02: GPU 1 and GPU 2 active; others idle at last check.
- Current active jobs on siml-02 are SGF/Safe_Denoiser retry watchers.

## Main table snapshot (Qwen SR, provisional where noted)
| Method | RB | P4DN | UD | MMA | I2P | Vio | Har | Hate | Shock | Illegal | Self |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline-official | 38.2 | 22.9 | 62.9 | 33.5 | [931] | 82.1 | 74.2 | 86.1 | 93.2 | 47.5 | 56.0 |
| SLD-official | [79] | - | [24] | - | [24] | - | [24] | - | [12] | [12] | [8] |
| RECE-official | - | - | - | - | - | - | - | - | - | - | - |
| SAFREE-official | - | - | - | - | - | - | - | - | - | - | - |
| Ours-v27 | 94.9 | 94.7 | 97.2 | 79.6 | 93.1 | 97.1 | 83.3 | 84.8 | 97.8 | [0/18] | 83.4 |
| SDErasure-v12 | - | 86.2 | - | 96.9 | 92.8 | 80.2 | 72.5 | 79.6 | 94.9 | 58.2 | 65.0 |

### Legend
- plain number: SR available from current result file
- `[N]`: images exist but final eval not complete / only partial artifact exists
- `[0/18]`: 0 out of 18 sweeps completed
- `-`: nothing trustworthy collected yet

## Completion ratio
- Baseline-official: 10/11
- SLD-official: 0/11 trustworthy final evals
- RECE-official: 0/11 trustworthy final evals
- SAFREE-official: 0/11 trustworthy final evals
- Ours-v27: 10/11
- SDErasure-v12: 9/11

## Important caveats
- Many historical logs under `logs/official_rerun/` contain failures; these are not all superseded by successful reruns.
- SLD official reruns are not yet in a trustworthy completed state.
- SGF and Safe_Denoiser are currently retrying on siml-02 GPU 1/2 after compatibility + reference-data fixes.
- SAFREE artist/nudity official reruns were attempted repeatedly but are not yet counted as trustworthy completed rows.
- Current official-rerun table is incomplete; do not use unfinished rows in the paper yet.

## Immediate interpretation
- The only rows that are currently near paper-usable are:
  - Ours-v27 (except Illegal)
  - SDErasure-v12 (mostly)
  - Baseline-official (provisional, but farthest along among reruns)
- SLD / RECE / SAFREE official reruns still need stabilization + completion.
