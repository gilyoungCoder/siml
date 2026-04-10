# Meeting Status — 2026-04-10 04:40 KST

## Executive summary
- **Meeting-safe rows right now**: **Ours-v27**, **SDErasure-v12**, and **partially** the new **SD1.4 baseline rerun**.
- **Not meeting-safe yet**: **SLD-official**, **RECE-official**, **SAFREE-official**, plus most **SGF / Safe_Denoiser** official reruns.
- **Biggest blocker**: official reruns for SLD/RECE/SAFREE are incomplete or unstable; several older logs are failed and should not be shown as final numbers.

## Main table snapshot (meeting-safe interpretation)
| Method | RB | P4DN | UD | MMA | I2P | Vio | Har | Hate | Shock | Illegal | Self |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours-v27** | 94.9 | 94.7 | 97.2 | 79.6 | 93.1 | 97.1 | 83.3 | 84.8 | 97.8 | **pending** | 83.4 |
| **SDErasure-v12** | - | 86.2 | - | 96.9 | 92.8 | 80.2 | 72.5 | 79.6 | 94.9 | 58.2 | 65.0 |
| **Baseline-official (provisional)** | 38.2 | 22.9 | 62.9 | 33.5 | eval pending | 82.1 | 74.2 | 86.1 | 93.2 | 47.5 | 56.0 |
| **SLD-official** | partial only | - | partial only | - | partial only | - | partial only | - | partial only | partial only | partial only |
| **RECE-official** | not ready | not ready | not ready | not ready | not ready | not ready | not ready | not ready | not ready | not ready | not ready |
| **SAFREE-official** | not ready | not ready | not ready | not ready | not ready | not ready | not ready | not ready | not ready | not ready | not ready |

## What is safe to say in the meeting
1. **Our method is the strongest completed row**.
2. **SDErasure-v12 is the strongest completed baseline row** among stable runs.
3. **Official baseline rerun is mostly there**, but **official SLD / RECE / SAFREE reruns are not complete enough yet to claim final reproduction**.
4. **Illegal-activity for Ours-v27 is still pending**, so the multi-concept table is **not fully closed**.

## What should NOT be shown as final
- Any older SLD/RECE/SAFREE numbers from custom wrappers or patched unofficial runs.
- Any row where only image counts exist without final evaluation.
- Any “official” row that still has repeated failed logs.

## Current completion ratio
- Ours-v27: **10/11**
- SDErasure-v12: **9/11**
- Baseline-official: **10/11** (one eval still pending)
- SLD-official: **0/11 trustworthy final evals**
- RECE-official: **0/11 trustworthy final evals**
- SAFREE-official: **0/11 trustworthy final evals**

## Immediate next actions after meeting
1. Finish **Ours-v27 Illegal**.
2. Finish **Baseline-official I2P** eval.
3. Stabilize **SLD-official** reruns.
4. Stabilize **RECE-official** reruns.
5. Stabilize **SAFREE-official** nudity + artist reruns.
6. Continue **SGF / Safe_Denoiser** official retries.
7. Run missing **VQAScore** and ablation summary tables.

## Recommendation for the meeting deck
- Show **Ours-v27** and **SDErasure-v12** as completed.
- If needed, show **Baseline-official** as **provisional / rerunning**.
- Explicitly mark **SLD / RECE / SAFREE official reproduction as in progress**.
