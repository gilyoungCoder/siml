# Image exemplar saturation plan (I2P q16 top-60, SD1.4 Ours both-hybrid)

Goal: measure whether image-probe-dependent concepts saturate as K image exemplars per family increases.

Default concepts: violence, hate, shocking (image-only >= text-only or strong image gain in probe ablation). Optional add: self-harm.

Grid: K = 1, 2, 4, 8, 12, 16 images/family. For K>available images, stored CLIP image features are repeated; no SD exemplar regeneration is done. Text probe/family target-anchor words/CAS/seed/sampler remain fixed.

Run:
```bash
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
# prepare only
bash $ROOT/scripts/run_image_saturation_i2p_q16_20260502.sh prepare
# launch on available GPUs
CONCEPTS="violence hate shocking" K_LIST="1 2 4 8 12 16" GPUS="0 1 2 3" nohup bash $ROOT/scripts/run_image_saturation_i2p_q16_20260502.sh launch > $ROOT/logs/image_saturation_q16top60_20260502/launch_all.nohup.log 2>&1 &
# monitor / summarize
bash $ROOT/scripts/run_image_saturation_i2p_q16_20260502.sh status
bash $ROOT/scripts/run_image_saturation_i2p_q16_20260502.sh summarize
```

Outputs:
- configs: `configs/image_saturation_q16top60_20260502/<concept>/k<K>.json`
- modified packs: `exemplars/image_saturation_q16top60_20260502/<concept>/k<K>/clip_grouped.pt`
- images/results: `outputs/image_saturation_q16top60_20260502/<concept>/k<K>/`
- summary CSV: `summaries/image_saturation_q16top60_20260502_results.csv`
