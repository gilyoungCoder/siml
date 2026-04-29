# SAFREE + SafeDenoiser / SAFREE + SGF official-repo reproduction

Prepared on siml-09 under:
`/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/baseline_safree_official_20260429`

Official repos cloned into `official_repos/`:
- Safe_Denoiser: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/baseline_safree_official_20260429/official_repos/Safe_Denoiser`
- SGF nudity_sdv1: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/baseline_safree_official_20260429/official_repos/SGF/nudity_sdv1`

Runtime assets are symlinked from existing downloaded copies under `unlearning-baselines/*_official` because the fresh GitHub clones do not include Google Drive checkpoints/negative images/caches.

Targets:
- Nudity full set: RAB 79, UD 142, MMA 1000, P4D-N 151.
- I2P Top-60 7 concepts: sexual, violence, self-harm, shocking, illegal_activity, harassment, hate.

Caveat:
- Official SGF repo is primarily SD-v1.4 nudity/COCO. I2P-7 is run through the same `category=all` interface and Qwen VLM eval for comparability, but the official SGF config still uses the nudity/i2p_sexual repellency reference unless we add an all-concept negative reference dataset.
