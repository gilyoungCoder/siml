# Writer handoff — SGF P4DN breakdown + NFE config anchors (2026-05-02)

## 1) SGF P4DN nudity breakdown

Use the **new repaired full P4DN run** if we want breakdown-backed numbers rather than old SR-only handoff.

**SAFREE + SGF on P4DN (SD v1.4 nudity, Qwen3-VL v5, n=151):**

| Method | Dataset | n | SR | Safe | Partial | Full | NR/NotPeople |
|---|---|---:|---:|---:|---:|---:|---:|
| SAFREE + SGF | P4DN | 151 | **70.2** | 37.7 | 32.5 | 25.8 | 4.0 |

Important note: this supersedes/conflicts with older SR-only handoff `74.2` and old stale incomplete run `72.8 (n=147)`. The reliable, breakdown-backed repaired run is `70.2` with `n=151`.

Evidence paths:
- Result file: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/sgf/nudity/p4dn/all/results_qwen3_vl_nudity_v5.txt`
- Images: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/sgf/nudity/p4dn/all` (151 pngs)
- Official config snapshot: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/sgf/nudity/p4dn/config.yaml`
- Generation log: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/logs/repair_sgf_nudity_breakdown_20260501/p4dn_gpu2.log`
- Eval log: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/logs/repair_sgf_nudity_breakdown_20260501/p4dn_eval_gpu2.log`

Key generation config:
- Backbone: `CompVis/stable-diffusion-v1-4`
- Prompt CSV: `prompts/nudity_csv/p4dn.csv`
- Seed: `42`
- One image per prompt
- `guidance_scale=7.5`, `image_length=512`, `num_inference_steps=50`
- Official SGF repo: `code/official_repos/SGF/nudity_sdv1/generate_unsafe_sgf.py`
- Base config: `configs/base/vanilla/safree_neg_prompt_config.json`
- Task config: `configs/sgf/sgf.yaml`
- `erase_id=safree_neg_prompt_rep_time`
- `safe_level=MEDIUM`, `safree=true`, `self_validation_filter=true`, `latent_re_attention=true`
- Repellency: `method=grad_mmd`, `n_embed=8`, `scale=0.03`, `proj_ref_path=caches/sd_sgf/i2p_sexual/repellency_proj_ref.pt`
- `nudity_thr=0.6`, `valid_case_numbers=0,100000`

Suggested writer message:
> For Table 1 SGF P4DN, please use the repaired full n=151 breakdown-backed value: SR 70.2 (Safe 37.7 / Partial 32.5 / Full 25.8 / NotPeople 4.0). This replaces the older SR-only 74.2 handoff and resolves the previous n=147 inconsistency.

## 2) NFE sweep launched

A quick NFE/runtime+VLM sweep is launched on `siml-05`, explicitly excluding GPU1.

- GPUs used: baseline GPU0, SAFREE GPU2, SafeDenoiser GPU3, SGF GPU4, Ours GPU5.
- GPU1 is not used.
- Dataset: `prompts/runtime/runtime_sexual_10.{txt,csv}` (10 sexual/nudity prompts)
- Steps/NFE: `10, 20, 30, 40, 50`
- Evaluator: Qwen3-VL v5 nudity rubric
- Output root: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/nfe/sexual10_5methods_20260502`
- Logs: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/logs/nfe/sexual10_5methods_20260502`
- Summary target: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/summaries/nfe_sexual10_5methods_20260502.md`
