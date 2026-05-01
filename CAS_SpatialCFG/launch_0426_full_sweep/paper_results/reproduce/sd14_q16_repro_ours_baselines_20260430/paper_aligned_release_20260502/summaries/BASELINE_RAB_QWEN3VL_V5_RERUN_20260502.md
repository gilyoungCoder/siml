# Baseline RAB Qwen3-VL v5 rerun (2026-05-02)

Rerun requested on siml-09 GPU0 for:

```text
/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/baselines_v2/rab
```

Command used:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 \
  /mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py \
  /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/baselines_v2/rab nudity qwen
```

Result file:

```text
/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/baselines_v2/rab/results_qwen3_vl_nudity_v5.txt
```

Result:

| Dataset | n | SR | Safe | Partial | Full | NotPeople |
|---|---:|---:|---:|---:|---:|---:|
| RAB baseline_v2 rerun v5 | 79 | 45.6 | 20.3 | 25.3 | 50.6 | 3.8 |

Interpretation:

- This does **not** match the old non-v5 `baselines_v2/rab/results_qwen3_vl_nudity.txt` value (27.8 SR).
- It is close to, but not exactly the same as, the paper main Table baseline RAB SR 48.1.
- It does not match the older appendix Table 9 row RAB SR 21.5.
- Therefore, if using strict Qwen3-VL v5 rerun on `baselines_v2/rab`, the baseline RAB breakdown is 45.6 / 20.3 / 25.3 / 50.6 / 3.8.
