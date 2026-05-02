# OMC NFE configs: SD1.4 I2P q16 top-60, Ours per-concept best

Common: SD v1.4, seed 42, CFG 7.5, resolution 512, NFE/steps grid {10,20,30,40,50}, mode=hybrid, probe=both, family_guidance=on.

| concept | safety_scale | cas_threshold | theta_text | theta_img | family pack | prompt |
|---|---:|---:|---:|---:|---|---|
| sexual | 20.0 | 0.5 | 0.10 | 0.30 | `$CAS/exemplars/i2p_v1/sexual/clip_grouped.pt` | `$CAS/prompts/i2p_q16_top60/sexual_q16_top60.txt` |
| violence | 20.0 | 0.4 | 0.30 | 0.10 | `$CAS/exemplars/i2p_v1/violence/clip_grouped.pt` | `$CAS/prompts/i2p_q16_top60/violence_q16_top60.txt` |
| self-harm | 7.0 | 0.5 | 0.10 | 0.10 | `$CAS/exemplars/i2p_v1/self-harm/clip_grouped.pt` | `$CAS/prompts/i2p_q16_top60/self-harm_q16_top60.txt` |
| shocking | 27.5 | 0.6 | 0.15 | 0.10 | `$CAS/exemplars/i2p_v1/shocking/clip_grouped.pt` | `$CAS/prompts/i2p_q16_top60/shocking_q16_top60.txt` |
| illegal_activity | 25.0 | 0.6 | 0.10 | 0.50 | `$CAS/exemplars/i2p_v1/illegal_activity/clip_grouped.pt` | `$CAS/prompts/i2p_q16_top60/illegal_activity_q16_top60.txt` |
| harassment | 31.25 | 0.5 | 0.10 | 0.50 | `$CAS/exemplars/i2p_v1/harassment/clip_grouped.pt` | `$CAS/prompts/i2p_q16_top60/harassment_q16_top60.txt` |
| hate | 28.0 | 0.6 | 0.25 | 0.0375 | `$CAS/exemplars/i2p_v1/hate/clip_grouped.pt` | `$CAS/prompts/i2p_q16_top60/hate_q16_top60.txt` |

Baseline/SAFREE/SafeDenoiser/SGF NFE should use the same seed/CFG/resolution/step grid. SafeDenoiser/SGF should reuse the final concept-specific negative/reference setup used for `safedenoiser_cs` and `sgf_cs`.
