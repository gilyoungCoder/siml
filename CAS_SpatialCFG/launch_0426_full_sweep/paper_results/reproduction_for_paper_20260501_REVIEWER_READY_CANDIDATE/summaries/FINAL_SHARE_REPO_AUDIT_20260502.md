# Final share-repo audit (2026-05-02)

Scope: SD v1.4 I2P q16 top-60 Ours/EBSG single-concept reproduction bundle.

## Verdict

- Configs resolve under `REPRO_ROOT` and use `how_mode=hybrid`, `probe_mode=both`, `family_guidance=true`.
- Per-concept configs now match the final rounded paper row, not the older suspiciously precise sweep snapshot.
- Each final cell has 60 generated PNGs in the source experiment, an `args.json`, and a Qwen3-VL V5 result file. Result files and args evidence are copied into this bundle under `results/` and `results/args/`.
- Added per-concept launch scripts under `scripts/per_concept/`.

## Final per-concept configs

| concept | sh | CAS | theta_text | theta_img | SR | output path in bundle |
|---|---:|---:|---:|---:|---:|---|
| sexual | 20.0 | 0.5 | 0.1 | 0.3 | 98.3 | `outputs/ours_best/i2p_q16/sexual/hybrid_sh20_cas0.5_txt0.10_img0.30` |
| violence | 20.0 | 0.4 | 0.3 | 0.1 | 88.3 | `outputs/ours_best/i2p_q16/violence/hybrid_sh20_cas0.4_txt0.30_img0.10` |
| self-harm | 7.5 | 0.5 | 0.1 | 0.1 | 58.3 | `outputs/ours_best/i2p_q16/self-harm/hybrid_sh7.5_cas0.5_txt0.10_img0.10` |
| shocking | 27.5 | 0.6 | 0.15 | 0.1 | 93.3 | `outputs/ours_best/i2p_q16/shocking/hybrid_sh27.5_cas0.6_txt0.15_img0.10` |
| illegal_activity | 25.0 | 0.6 | 0.1 | 0.5 | 46.7 | `outputs/ours_best/i2p_q16/illegal_activity/hybrid_sh25_cas0.6_txt0.10_img0.50` |
| harassment | 30.0 | 0.5 | 0.1 | 0.5 | 63.3 | `outputs/ours_best/i2p_q16/harassment/hybrid_sh30_cas0.5_txt0.10_img0.50` |
| hate | 27.5 | 0.6 | 0.25 | 0.05 | 66.7 | `outputs/ours_best/i2p_q16/hate/hybrid_sh27.5_cas0.6_txt0.25_img0.05` |
| **Avg** | | | | | **73.6** | |

## Scripts

- Run all I2P concepts: `GPU=0 PY_SAFGEN=/path/to/python bash scripts/run_i2p_best_all.sh`
- Run one concept: `GPU=0 PY_SAFGEN=/path/to/python bash scripts/per_concept/run_i2p_<concept>.sh`
- Evaluate outputs: `GPU=0 PY_VLM=/path/to/vlm/python VLM_SCRIPT=/path/to/opensource_vlm_i2p_all_v5.py bash scripts/eval_v5_outputs.sh`
- Verify wiring: `REPRO_ROOT=$PWD python scripts/verify_configs.py`

## Notes

- The bundle intentionally reports rounded, defensible hyperparameters for self-harm/harassment/hate/violence, superseding older CANDIDATE values `self-harm sh=7.0`, `harassment sh=31.25`, `hate img=0.0375`, and violence `sh=19.5/img=0.225`.
- Nudity configs were already hybrid/family-wired and left unchanged in this pass.
