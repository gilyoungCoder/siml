# Current Grid-Search Best Config Summary

- Generated: `2026-04-04T03:29:18`
- Outputs root: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs`
- Versions scanned: `v14, v15, v16, v17, v18, v19`

## Coverage by version

| Version | Configs | With NudeNet | With SR | Pareto points |
| --- | ---: | ---: | ---: | ---: |
| v14 | 72 | 72 | 67 | 6 |
| v15 | 24 | 24 | 21 | 2 |
| v16 | 0 | 0 | 0 | 0 |
| v17 | 144 | 144 | 117 | 6 |
| v18 | 218 | 218 | 0 | 0 |
| v19 | 0 | 0 | 0 | 0 |

## Best configs by version

| Version | Best NN | NN% | SR% | Best SR | NN% | SR% | Best balanced rank-sum | NN% | SR% |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| v14 | image_dag_adaptive_ss5.0_st0.2 | 3.2 | 13.6 | both_dag_adaptive_ss2.0_st0.2 | 15.8 | 69.0 | both_dag_adaptive_ss3.0_st0.4 | 11.1 | 66.8 |
| v15 | text_dag_adaptive_ss5.0_st0.2_np16 | 3.8 | 27.2 | text_dag_adaptive_ss3.0_st0.4_np16 | 19.9 | 72.8 | text_dag_adaptive_ss5.0_st0.4_np16 | 13.0 | 70.6 |
| v16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| v17 | both_dag_adaptive_ss5.0_st0.2_fused | 2.9 | N/A | image_hybrid_ss5.0_st0.3_fused | 28.5 | 77.8 | image_hybrid_ss5.0_st0.2_fused | 27.9 | 76.6 |
| v18 | both_dag_adaptive_ss5.0_st0.2_none_sb0.5 | 2.9 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| v19 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## Global Pareto frontier (NN low, SR high)

| Version | Config | NN% | SR% | Relevant_SR% |
| --- | --- | ---: | ---: | ---: |
| v17 | text_dag_adaptive_ss5.0_st0.3_fused | 2.9 | 10.4 | 82.5 |
| v14 | image_dag_adaptive_ss5.0_st0.2 | 3.2 | 13.6 | 87.8 |
| v14 | image_dag_adaptive_ss5.0_st0.3 | 3.5 | 14.6 | 86.8 |
| v17 | image_dag_adaptive_ss1.0_st0.3_probe_only | 3.8 | 30.4 | 88.1 |
| v14 | image_dag_adaptive_ss1.0_st0.3 | 4.1 | 34.2 | 90.0 |
| v14 | image_dag_adaptive_ss1.0_st0.4 | 4.8 | 37.3 | 89.4 |
| v14 | both_dag_adaptive_ss5.0_st0.4 | 6.0 | 63.0 | 89.2 |
| v14 | both_dag_adaptive_ss3.0_st0.4 | 11.1 | 66.8 | 85.4 |
| v15 | text_dag_adaptive_ss5.0_st0.4_np16 | 13.0 | 70.6 | 85.8 |
| v17 | text_dag_adaptive_ss5.0_st0.4_probe_only | 13.0 | 70.6 | 85.8 |
| v15 | text_dag_adaptive_ss3.0_st0.4_np16 | 19.9 | 72.8 | 81.3 |
| v17 | text_dag_adaptive_ss3.0_st0.4_probe_only | 19.9 | 72.8 | 81.3 |
| v17 | image_hybrid_ss5.0_st0.2_fused | 27.9 | 76.6 | 79.9 |
| v17 | image_hybrid_ss5.0_st0.3_fused | 28.5 | 77.8 | 81.5 |
