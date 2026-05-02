# Verified Ours best config audit

All SD3/FLUX official generation/eval cells under `outputs/crossbackbone_0501` had missing count 0 before this bundle was created.

## SD1.4 I2P q16 top-60 Ours best

| Concept | Best variant | SR | Config |
|---|---|---:|---|
| sexual | hybrid_best_tau05_cas0.5 | 98.3 | `configs/ours_best/i2p_q16/sexual.json` |
| violence | hybrid_best_img075_img0.225 | 81.7 | `configs/ours_best/i2p_q16/violence.json` |
| self-harm | hybrid_best_tau05_cas0.5 | 51.7 | `configs/ours_best/i2p_q16/self-harm.json` |
| shocking | hybrid_best_ss125_ss27.5 | 93.3 | `configs/ours_best/i2p_q16/shocking.json` |
| illegal_activity | hybrid_best_ss125_ss25.0 | 46.7 | `configs/ours_best/i2p_q16/illegal_activity.json` |
| harassment | hybrid_best_ss125_ss31.25 | 68.3 | `configs/ours_best/i2p_q16/harassment.json` |
| hate | hybrid_best_img075_img0.0375 | 73.3 | `configs/ours_best/i2p_q16/hate.json` |

## SD1.4 nudity Ours hybrid

- unlearndiff: 97.2% (`configs/ours_best/nudity/unlearndiff.json`)
- rab: 96.2% (`configs/ours_best/nudity/rab.json`)
- mma: 84.2% (`configs/ours_best/nudity/mma.json`)
- p4dn: 97.4% (`configs/ours_best/nudity/p4dn.json`)

## Family/anchor-target check

Run `python scripts/verify_configs.py` to print the family metadata from each `clip_grouped.pt`. The copied configs preserve the exact target/anchor concepts and tuned args from the verified run.

