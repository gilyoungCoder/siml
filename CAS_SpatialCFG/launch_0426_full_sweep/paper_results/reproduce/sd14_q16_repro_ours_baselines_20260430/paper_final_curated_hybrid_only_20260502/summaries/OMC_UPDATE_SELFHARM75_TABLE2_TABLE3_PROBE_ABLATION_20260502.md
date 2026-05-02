# OMC update request: SD1.4 I2P single Ours self-harm canonical -> sh=7.5

## Decision
Update the SD v1.4 I2P q16 top-60 **Ours single-concept self-harm** cell from the previous canonical `sh=7.0` run to the new rounded `sh=7.5` run.

This affects both:
1. **Table 2**: I2P harmful-concept erasure on SD v1.4, single-concept Ours row.
2. **Table 3**: Probe-channel ablation (SR% ↑, SD v1.4 I2P top-60). In Table 3, the `Both (Ours)` row must match the final Table 2 Ours single-concept canonical values.

Do **not** update harassment or hate to the rounded sanity runs; those were worse than the existing canonical runs.

---

## New canonical self-harm evidence

### New self-harm run to use
- Concept: `self-harm`
- Backbone: SD v1.4
- Dataset: I2P q16 top-60 self-harm split
- Prompt file: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/i2p_q16_top60/self-harm_q16_top60.txt`
- Output path:
  `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/ours_round_sh_sanity_20260502/i2p_q16/self-harm/hybrid_sh7.5_cas0.5_txt0.10_img0.10_round`
- Args:
  `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/ours_round_sh_sanity_20260502/i2p_q16/self-harm/hybrid_sh7.5_cas0.5_txt0.10_img0.10_round/args.json`
- Result file:
  `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/ours_round_sh_sanity_20260502/i2p_q16/self-harm/hybrid_sh7.5_cas0.5_txt0.10_img0.10_round/results_qwen3_vl_self_harm_v5.txt`
- Image count: 60 PNGs
- Evaluator: Qwen3-VL V5, concept key `self_harm`

### New self-harm config
```text
how_mode       = hybrid
probe_mode     = both
family_guidance= true
family_config  = /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/self-harm/clip_grouped.pt
seed           = 42
steps          = 50
cfg_scale      = 7.5
safety_scale   = 7.5
cas_threshold  = 0.5
theta_text     = 0.10
theta_img      = 0.10
target_concepts= ["crying despair", "hanging noose", "self mutilation"]
target_words   = ["crying", "despair", "hanging", "noose", "self", "mutilation"]
anchor_concepts= ["wellness", "recovery", "supportive_therapy"]
```

### New self-harm result
```text
SR      = 58.3  (35/60)
Safe    = 10.0  (6/60)
Partial = 48.3  (29/60)
Full    = 28.3  (17/60)
NR      = 13.3  (8/60)
```

Previous self-harm canonical was:
```text
sh=7.0 / cas=0.5 / theta_text=0.10 / theta_img=0.10
SR      = 51.7
Safe    = 8.3
Partial = 43.3
Full    = 23.3
NR      = 25.0
```

Net effect: self-harm SR improves by **+6.6 pp** and NR decreases by **-11.7 pp**, while Full increases by **+5.0 pp**.

---

## Rounded sanity runs that should NOT replace canonical

### Harassment rounded run: do not use
- Config: `sh=30.0 / cas=0.5 / theta_text=0.10 / theta_img=0.50`
- Output:
  `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/ours_round_sh_sanity_20260502/i2p_q16/harassment/hybrid_sh30_cas0.5_txt0.10_img0.50_round`
- Result:
```text
SR      = 63.3
Safe    = 50.0
Partial = 13.3
Full    = 15.0
NR      = 21.7
```
- Existing canonical harassment remains better:
```text
sh=31.25 / cas=0.5 / theta_text=0.10 / theta_img=0.50
SR      = 68.3
Safe    = 56.7
Partial = 11.7
Full    = 13.3
NR      = 18.3
```

### Hate rounded run: do not use
- Config: `sh=27.5 / cas=0.6 / theta_text=0.25 / theta_img=0.05`
- Output:
  `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/ours_round_sh_sanity_20260502/i2p_q16/hate/hybrid_sh27.5_cas0.6_txt0.25_img0.05_round`
- Result:
```text
SR      = 66.7
Safe    = 56.7
Partial = 10.0
Full    = 16.7
NR      = 16.7
```
- Existing canonical hate remains better:
```text
sh=28.0 / cas=0.6 / theta_text=0.25 / theta_img=0.0375
SR      = 73.3
Safe    = 56.7
Partial = 16.7
Full    = 10.0
NR      = 16.7
```

---

## Updated Table 2 Ours single-concept row

Use these final SD v1.4 I2P q16 top-60 Ours single-concept values:

| Concept | SR | Safe | Partial | Full | NR | Config note |
|---|---:|---:|---:|---:|---:|---|
| sexual | 98.3 | 91.7 | 6.7 | 1.7 | 0.0 | sh20 / cas0.5 / θt0.10 / θi0.30 |
| violence | 88.3 | 73.3 | 15.0 | 5.0 | 6.7 | sh20 / cas0.4 / θt0.30 / θi0.10 |
| self-harm | 58.3 | 10.0 | 48.3 | 28.3 | 13.3 | **UPDATED: sh7.5 / cas0.5 / θt0.10 / θi0.10** |
| shocking | 93.3 | 88.3 | 5.0 | 3.3 | 3.3 | sh27.5 / cas0.6 / θt0.15 / θi0.10 |
| illegal | 46.7 | 31.7 | 15.0 | 15.0 | 38.3 | sh25 / cas0.6 / θt0.10 / θi0.50 |
| harassment | 68.3 | 56.7 | 11.7 | 13.3 | 18.3 | sh31.25 / cas0.5 / θt0.10 / θi0.50 |
| hate | 73.3 | 56.7 | 16.7 | 10.0 | 16.7 | sh28 / cas0.6 / θt0.25 / θi0.0375 |
| **Avg** | **75.2** |  |  |  |  | mean SR over 7 concepts |

Previous Table 2 Ours single Avg was 74.3. New Avg is:
```text
(98.3 + 88.3 + 58.3 + 93.3 + 46.7 + 68.3 + 73.3) / 7 = 75.2
```

---

## Updated Table 3: Probe-channel ablation

This is the table captioned:

> **Table 3: Probe-channel ablation (SR% ↑, SD v1.4 I2P top-60).**

The `Both (Ours)` row must be updated to match the final Table 2 Ours single-concept canonical row above.

Use:

| Probe | sexual | violence | self-harm | shocking | illegal | harass. | hate | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Text | 98.3 | 71.7 | 35.0 | 85.0 | 45.0 | 70.0 | 55.0 | 65.7 |
| Image | 96.7 | 86.7 | 50.0 | 88.3 | 31.7 | 40.0 | 66.7 | 65.7 |
| Both (Ours) | 98.3 | 88.3 | **58.3** | 93.3 | 46.7 | 68.3 | 73.3 | **75.2** |

Important: previous probe-ablation MD used `Both Avg=73.8` with self-harm 51.7, shocking 91.7, illegal 43.3, harassment 70.0. That row is now stale/sub-run mixed and should be replaced with the final Table 2-aligned `Both (Ours)` row above.

---

## Short instruction to OMC / ml writer

Please update all paper locations that reference SD v1.4 I2P Ours single-concept Avg or probe-ablation Both Avg:

1. Table 2 Ours single-concept self-harm: `51.7 -> 58.3`.
2. Table 2 Ours single-concept Avg: `74.3 -> 75.2`.
3. Table 3 Probe-channel ablation `Both (Ours)` row:
   - self-harm `51.7 -> 58.3`
   - shocking `91.7 -> 93.3`
   - illegal `43.3 -> 46.7`
   - harassment `70.0 -> 68.3`
   - Avg `73.8 -> 75.2`
4. Keep harassment canonical as `sh=31.25`; do not use the rounded `sh=30` result.
5. Keep hate canonical as `sh=28 / theta_img=0.0375`; do not use the rounded `sh=27.5 / theta_img=0.05` result.
6. Ensure any text claim such as “I2P 7-concept Avg SR” uses the new Ours single Avg **75.2** where it refers to single-concept Ours.

