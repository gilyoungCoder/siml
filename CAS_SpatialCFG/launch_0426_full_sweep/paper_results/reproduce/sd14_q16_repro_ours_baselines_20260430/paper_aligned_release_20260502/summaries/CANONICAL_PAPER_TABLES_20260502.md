# Canonical paper-aligned experiment bundle (2026-05-02)

Use this folder as the source of truth for paper updates. It supersedes stale violence-best files that reported 81.7; final paper-aligned violence single-concept Ours is 88.3 from `sh20_tau04_txt030_img010`.

## SD1.4 I2P q16 top-60 single-concept (SR%)

| concept | Baseline | SAFREE | SAFREE+SafeDenoiser-CS | SAFREE+SGF-CS | Ours best | Ours config |
|---|---:|---:|---:|---:|---:|---|
| sexual | 68.3 | 83.3 | 91.7 | 96.7 | **98.3** | sh=20.0, cas=0.5, text=0.10, img=0.30 |
| violence | 36.7 | 73.3 | 40.0 | 40.0 | **88.3** | sh=20.0, cas=0.4, text=0.30, img=0.10 |
| self-harm | 43.3 | 36.7 | 38.3 | 36.7 | **51.7** | sh=7.0, cas=0.5, text=0.10, img=0.10 |
| shocking | 15.0 | 81.7 | 21.7 | 18.3 | **93.3** | sh=27.5, cas=0.6, text=0.15, img=0.10 |
| illegal_activity | 31.7 | 35.0 | 41.7 | 30.0 | **46.7** | sh=25.0, cas=0.6, text=0.10, img=0.50 |
| harassment | 25.0 | 28.3 | 26.7 | 23.3 | **68.3** | sh=31.25, cas=0.5, text=0.10, img=0.50 |
| hate | 25.0 | 43.3 | 28.3 | 21.7 | **73.3** | sh=28.0, cas=0.6, text=0.25, img=0.0375 |
| **Avg** | **35.0** | **54.5** | **41.2** | **38.1** | **74.2** | — |

## Ours per-class breakdown (SR/Safe/Partial/Full/NR)

| concept | Ours best breakdown |
|---|---:|
| sexual | 98.3 / 91.7 / 6.7 / 1.7 / 0.0 |
| violence | 88.3 / 73.3 / 15.0 / 5.0 / 6.7 |
| self-harm | 51.7 / 8.3 / 43.3 / 23.3 / 25.0 |
| shocking | 93.3 / 88.3 / 5.0 / 3.3 / 3.3 |
| illegal_activity | 46.7 / 31.7 / 15.0 / 15.0 / 38.3 |
| harassment | 68.3 / 56.7 / 11.7 / 13.3 / 18.3 |
| hate | 73.3 / 56.7 / 16.7 / 10.0 / 16.7 |

## Probe ablation, q16 top-60 incl. sexual

| Probe | sexual | violence | self-harm | shocking | illegal | harassment | hate | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Text | 98.3 | 71.7 | 35.0 | 85.0 | 45.0 | 70.0 | 55.0 | 65.7 |
| Image | 96.7 | 86.7 | 50.0 | 88.3 | 31.7 | 40.0 | 66.7 | 65.7 |
| Both (Ours) | 98.3 | **88.3** | 51.7 | 91.7 | 43.3 | 70.0 | 73.3 | **73.8** |

## Multi-concept q16 top-60

See `summaries/i2p_multi_sr_full_nr_tables_20260501.md`. Main rows:
- 2c sexual+violence: SAFREE 65.0, SafeDenoiser 77.5, SGF 72.5, Ours 76.7.
- 3c sexual+violence+shocking: SAFREE 52.8, SafeDenoiser 68.9, SGF 61.1, Ours 81.7.
- 7c all: SAFREE 41.9, SafeDenoiser 50.7, SGF 48.3, Ours 73.1.

## Nudity Table 1 notes

Use `TABLE1_NUDITY_BREAKDOWN_RELIABLE_HANDOFF_20260501.md` plus `WRITER_SGF_P4DN_AND_NFE_CONFIGS_20260502.md` for repaired SGF P4DN n=151 breakdown. Repaired SGF P4DN: SR 70.2 / Safe 37.7 / Partial 32.5 / Full 25.8 / NotPeople 4.0.

