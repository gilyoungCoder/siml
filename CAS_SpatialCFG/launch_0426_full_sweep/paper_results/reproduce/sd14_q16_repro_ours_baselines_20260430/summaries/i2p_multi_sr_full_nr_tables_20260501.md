# Table 3 candidate: SD v1.4 multi-concept I2P

Use as appendix or main multi-concept table.  
Metric: Qwen3-VL v5 over I2P q16 top-60.  
Each cell is **SR / Full / NR** in %, where SR = Safe + Partial and NR = NotRelevant. Lower Full and lower NR are better; higher SR is better.

## 2-concept: sexual + violence

| Method | sexual | violence | Avg SR | Avg Full | Avg NR |
|---|---:|---:|---:|---:|---:|
| SAFREE multi | — | — | 0.0 | 0.0 | 0.0 |
| SAFREE + SafeDenoiser multi | 88.3/0.0/11.7 | 66.7/28.3/5.0 | 77.5 | 14.2 | 8.3 |
| SAFREE + SGF multi | 86.7/1.7/11.6 | 58.3/33.3/8.3 | 72.5 | 17.5 | 9.9 |
| **Ours multi** | 90.0/8.3/1.7 | 63.3/28.3/8.3 | **76.7** | **18.3** | **5.0** |

## 3-concept: sexual + violence + shocking

| Method | sexual | violence | shocking | Avg SR | Avg Full | Avg NR |
|---|---:|---:|---:|---:|---:|---:|
| SAFREE multi | — | — | — | 0.0 | 0.0 | 0.0 |
| SAFREE + SafeDenoiser multi | 85.0/1.7/13.3 | 60.0/31.7/8.3 | 61.7/38.3/0.0 | 68.9 | 23.9 | 7.2 |
| SAFREE + SGF multi | 90.0/1.7/8.3 | 53.3/38.3/8.3 | 40.0/60.0/0.0 | 61.1 | 33.3 | 5.5 |
| **Ours multi (C2_ss130)** | 90.0/6.7/3.3 | 76.7/16.7/6.7 | 78.3/21.7/0.0 | **81.7** | **15.0** | **3.3** |

## 7-concept multi

| Method | sexual | violence | self-harm | shocking | illegal | harassment | hate | Avg SR | Avg Full | Avg NR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAFREE multi | — | — | — | — | — | — | — | 0.0 | 0.0 | 0.0 |
| SAFREE + SafeDenoiser multi | 90.0/0.0/10.0 | 58.3/35.0/6.7 | 40.0/11.7/48.3 | 60.0/38.3/1.7 | 45.0/6.7/48.3 | 28.3/41.7/30.0 | 33.3/46.7/20.0 | 50.7 | 25.7 | 23.6 |
| SAFREE + SGF multi | 86.7/0.0/13.3 | 46.7/38.3/15.0 | 43.3/8.3/48.3 | 50.0/50.0/0.0 | 38.3/11.7/50.0 | 36.7/15.0/48.3 | 36.7/26.7/36.7 | 48.3 | 21.4 | 30.2 |
| **Ours multi (C2_ss130)** | 88.3/1.7/10.0 | 85.0/3.3/11.7 | 66.7/5.0/28.3 | 88.3/1.7/10.0 | 65.0/3.3/31.7 | 60.0/13.3/26.7 | 58.3/26.7/15.0 | **73.1** | **7.9** | **19.1** |

Notes:

- Ours 3c and 7c rows use the q16 top-60 `C2_ss130` / scale-1.3 multi-concept configuration.
- SafeDenoiser/SGF 7c balanced-ref rerun is queued separately; until it finishes, the 7c baseline rows are the completed non-balanced official runs.
- For compact paper tables, use SR/Full/NR cells; detailed Safe/Partial counts remain in the per-output `results_qwen3_vl_*_v5.txt` files.

Suggested caption language:

> Multi-concept I2P q16 top-60 evaluation on SD v1.4. Each cell reports SR / Full / NR (%), where SR = Safe + Partial and NR = NotRelevant under the Qwen3-VL four-class rubric. For 2c, 3c, and 7c, a single run suppresses all listed concepts simultaneously. Ours uses the C2_ss130 scale-1.3 multi-concept configuration.

---
