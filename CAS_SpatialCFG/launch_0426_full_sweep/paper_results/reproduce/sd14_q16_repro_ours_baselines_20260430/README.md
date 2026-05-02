# SD1.4 Q16 reproducibility run (2026-04-30)

Purpose: reproduce/compare with the current SD1.4 Ours hybrid best configs using the **I2P Q16 top-60** prompt split, plus full nudity datasets. Also launches I2P Q16 top-60 SD1.4 baseline / SAFREE / SAFREE+SafeDenoiser / SAFREE+SGF.

Important prompt distinction:
- `i2p_sweep60/*_sweep.txt` is not Q16 top-60.
- This folder uses `prompts/i2p_q16_top60/*_q16_top60.txt` for all 7 I2P concepts.

Ours config source:
- I2P: `CAS_SpatialCFG/launch_0426_full_sweep/paper_results/single/*/args.json`, with only prompt/outdir changed to Q16 root. Hybrid only.
- Nudity: `paper_results_master/01_nudity_sd14_5bench/*_hybrid/args.json` for RAB/UD/P4DN/MMA. No sweep; one execution per dataset.

Code surfaces:
- `code/SafeGen/` copied from `/mnt/home3/yhgil99/unlearning/SafeGen`.
- `code/generate_baseline.py` copied from CAS_SpatialCFG.
- `code/official_repos/` copied from patched official Safe_Denoiser/SGF clones.
- Standalone SAFREE uses the existing `/mnt/home3/yhgil99/unlearning/SAFREE` repo.

Launch:
- `scripts/launch_siml01.sh`
- `scripts/launch_siml02.sh`

Status:
```bash
python scripts/status.py
```
