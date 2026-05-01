# OMC delivery note — final canonical paper data (2026-05-02)

Use this folder as the final curated handoff:

`/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_final_curated_hybrid_only_20260502`

Main handoff file:

`paper_final_curated_hybrid_only_20260502/summaries/FINAL_OMC_MLWRITER_HYBRID_ONLY_HANDOFF_20260502.md`

## Critical canonical decision

For **I2P cross-backbone Ours**, use the **image-backed** rows, not the older release-copy/status rows.

Canonical compact row:

| Benchmark | Method | SD3 Avg SR | FLUX1 Avg SR |
|---|---|---:|---:|
| I2P q16 top-60 | Ours hybrid/image-backed | 65.0 | 80.5 |

Source doc:

`paper_final_curated_hybrid_only_20260502/summaries/CROSSBACKBONE_OURS_I2P_IMAGE_BACKED_BREAKDOWN_20260501.md`

Manifest/validation:

- `paper_final_curated_hybrid_only_20260502/manifests/i2p_crossbackbone_ours_image_backed_manifest.json`
- `paper_final_curated_hybrid_only_20260502/checks/i2p_crossbackbone_image_backed_validation.txt`

Validation result: every canonical image-backed row has:

- exactly 60 PNG images,
- `args.json`,
- Qwen3-VL V5 result file,
- q16 top-60 prompt path,
- `how_mode=hybrid`,
- `probe_mode=both`.

The older `63.8 / 76.2` values came from a release-copy/status selection and should not be used for the final paper when full image-backed verification is required.

## MJA scope

MJA cross-backbone table is **SD3.0 and FLUX1.0 only**. Do not include SD1.4 SafeDenoiser/SGF MJA cells and do not mark them TBD; they are outside scope.

## Remaining true TBD

- Table 1 SLD-Weak/Medium/Strong/Max P4DN remains genuinely missing unless rerun.
- COCO FID/CLIP remains pending and should not be treated as final.
