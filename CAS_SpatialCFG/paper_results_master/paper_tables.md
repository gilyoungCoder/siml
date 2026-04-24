# Paper Tables — Final

## Table 1: SD1.4 Nudity Benchmarks (SR / Full %)

| Method | RAB | UnlearnDiff | MMA | P4DN | MJA-Sexual |
|---|---|---|---|---|---|
| SAFREE | 77.2/12.7 | 90.1/2.1 | 75.5/20.2 | -- | -- |
| Ours (anchor) | 93.3/6.7 | 92.9/7.1 | 77.0/23.0 | 91.2/8.8 | 90.0/10.0 |
| Ours (hybrid) | 97.4/2.6 | 98.6/1.4 | 84.7/15.3 | 97.4/2.6 | 96.7/3.3 |

## Table 2: I2P top60 (SR / Full %, SD1.4, 60 prompts/concept)

| Concept | Baseline | SAFREE | Ours (anchor) | Ours (hybrid) |
|---|---|---|---|---|
| violence | 57/43 | 77/10 | 88.3/3.3 | 91.7/8.3 |
| self-harm | 55/35 | 53/3 | 68.3/18.3 | 61.7/6.7 |
| shocking | 28/72 | 75/5 | 78.3/18.3 | 88.3/11.7 |
| illegal_activity | 35/27 | 33/7 | 46.7/20.0 | 41.7/25.0 |
| harassment | 25/53 | 25/12 | 71.7/18.3 | 46.7/30.0 |
| hate | 30/65 | 33/32 | 55.0/26.7 | 66.7/16.7 |

## Table 3: MJA Cross-Backbone (SR / Full %)

| Concept | Backbone | Baseline | SAFREE | Ours (anchor) | Ours (hybrid) |
|---|---|---|---|---|---|
Traceback (most recent call last):
  File "<stdin>", line 84, in <module>
NameError: name 'base' is not defined. Did you mean: 'False'?

## Table 3: MJA Cross-Backbone (SR / Full %)

| Concept | Backbone | Baseline | SAFREE | Ours (anchor) | Ours (hybrid) |
|---|---|---|---|---|---|
| sexual | SD1.4 | 43/57 | 71/29 | 90.0/10.0 | 96.7/3.3 |
| sexual | SD3 | 51/49 | 64/36 | 81.0/19.0 | 84.0/16.0 |
| sexual | FLUX1 | 62/38 | 73/27 | 96.0/4.0 | 97.0/3.0 |
| violent | SD1.4 | 10/86 | 55/27 | 56.0/26.0 | 69.0/16.0 |
| violent | SD3 | 0/100 | 6/94 | 58.0/42.0 | -- |
| violent | FLUX1 | 2/98 | 3/97 | 89.0/11.0 | 67.0/20.0 |
| illegal | SD1.4 | 51/40 | 73/10 | 76.0/8.0 | -- |
| illegal | SD3 | 19/80 | 20/77 | 53.0/42.0 | 67.0/16.0 |
| illegal | FLUX1 | 32/65 | 34/64 | 86.0/13.0 | 58.0/35.0 |
| disturbing | SD1.4 | 49/51 | 82/10 | 89.0/0.0 | -- |
| disturbing | SD3 | 35/65 | 63/37 | 86.0/14.0 | 90.0/10.0 |
| disturbing | FLUX1 | 51/49 | 46/54 | 98.0/2.0 | 96.0/4.0 |

## Table 4: Multi-Concept Erasure (SD1.4)

### MJA multi (sexual + violent erased simultaneously)
| Concept | Single best (anchor) | Multi (anchor) |
|---|---|---|
| sexual | 90.0/10.0 | 73.0/27.0 |
| violent | 56.0/26.0 | 71.0/23.0 |

### I2P multi (6 concepts erased simultaneously, hybrid)
| Concept | Single best (hybrid) | Multi (hybrid) |
|---|---|---|
| violence | 91.7/8.3 | 60.0/40.0 |
| self-harm | 61.7/6.7 | 50.0/28.3 |
| shocking | 88.3/11.7 | 43.3/56.7 |
| illegal_activity | 41.7/25.0 | 46.7/25.0 |
| harassment | 46.7/30.0 | 33.3/50.0 |
| hate | 66.7/16.7 | 36.7/48.3 |

## Table 5: Probe Mode Ablation (SD1.4 I2P top60)

| Concept | txt-only | img-only | both |
|---|---|---|---|
| violence | 86.7/8.3 | 86.7/5.0 | 91.7/8.3 |
| self-harm | 55.0/18.3 | 55.0/28.3 | 61.7/6.7 |
| shocking | 60.0/28.3 | 78.3/20.0 | 88.3/11.7 |
| illegal_activity | 43.3/23.3 | 38.3/23.3 | 41.7/25.0 |
| harassment | 38.3/35.0 | 46.7/35.0 | 46.7/30.0 |
| hate | 51.7/33.3 | 60.0/18.3 | 66.7/16.7 |
