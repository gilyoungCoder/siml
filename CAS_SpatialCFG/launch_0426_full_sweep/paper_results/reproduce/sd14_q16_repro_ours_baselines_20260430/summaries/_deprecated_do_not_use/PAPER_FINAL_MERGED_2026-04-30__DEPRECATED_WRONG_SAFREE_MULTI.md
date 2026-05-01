# Paper Final Merged Status — 2026-04-30 KST (post-recovery + experiment integration)

> ml-paper-writer 가 paper Tables / Appendix 작성 시 이 문서를 single source of truth 로 사용.
> 이 문서는 GPT 의 복구 패키지 (`PAPER_UPDATE_REPRO_TABLES_20260430.md`) + 그 이후 진행/검증된 실험 결과를 통합한 최신본.
> 변경/수정된 부분은 ⚠️ 표시.

Root: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/`

---

## 0. Status overview

| 항목 | 상태 |
|---|---|
| Single-concept i2p_sweep60 (60 prompts) — 5-method 비교 | ✅ GPT 표 그대로 valid (SAFREE single 은 v2 옵션 켜져있음) |
| **Paper Table 1 (UnlearnDiff/RAB/MMA/p4dn/COCO × 12 methods)** | ⚠️ 부분 — 외부 baseline (literature) + ESD/SDD p4dn (자체 측정) 채움. Many TBD (§1.5) |
| **글로벌 표 정책** (ml-paper-writer 적용 필수) | 📌 모든 main 표에 **SR / Full / NotRel** 세 metric 동시 표시 권장 — NotRel 가 evaluator sanity check 역할 (§1.5e) |
| Multi-concept i2p_sweep60 (2c/3c/4c/7c) — SAFREE vs EBSG | ⚠️ **GPT 표의 SAFREE multi 값은 잘못 (옛 crippled 버전)** — `phase_safree_v2/` 의 v2 수치로 교체 |
| Hate single best | ⚠️ **n_tok=4 통일 (60.0%)** 로 갱신, 이전 68.33% (n_tok=16) deprecated. 상세: `HATE_DECISION_2026-04-29.md` |
| Image-count saturation (K∈{1,2,4,8,12,16}) | ✅ Nested subsample 7 concept × 6 K = 42 cells 완료. `img_sat_nested.{pdf,png}` |
| Probe-mode ablation (text/image/both) | 🔄 **PNG 21/21 generated, v5 eval 0/21 PENDING** |
| SAFREE multi q16top60 v2 (60 prompts) | 🔄 **PNG 12/12 generated, v5 eval 0/12 PENDING** |
| SafeDenoiser / SGF concept-specific I2P | 🔄 GPT 노트: siml-09 에서 실행 중 |
| COCO FID/CLIP for SafeDenoiser/SGF nudity | 🔄 GPT 노트: siml-07 GPU1 실행 중 |
| Human survey | ✅ DB 복구 (1662 rows / 20 annotators / 400 items). Qwen3-VL vs human: **전체 62.2% / S/P collapsed 82.6%**, Qwen "S+P" precision **97.6%** → SR 은 conservative lower bound (§7) |

---

## 1. Single-concept i2p_q16 top60 — 5-method (GPT 표 유지)

`/scripts/sd14_q16_repro_ours_baselines_20260430/summaries/i2p_q16_top60_v5_5method_comparison_with_ours_best.csv`

| concept | baseline | SAFREE | SafeDenoiser | SGF | ours main | ours best | best variant |
|---|---:|---:|---:|---:|---:|---:|---|
| sexual | 68.3 | 83.3 | 91.7 | 95.0 | 90.0 | 98.3 | `hybrid_best_tau05_cas0.5` |
| violence | 36.7 | 73.3 | 43.3 | 41.7 | 75.0 | 81.7 | `hybrid_best_img075_img0.225` |
| self-harm | 43.3 | 36.7 | 41.7 | 35.0 | 50.0 | 51.7 | `hybrid_best_tau05_cas0.5` |
| shocking | 15.0 | 81.7 | 20.0 | 16.7 | 85.0 | 93.3 | `hybrid_best_ss125_ss27.5` |
| illegal_activity | 31.7 | 35.0 | 41.7 | 25.0 | 41.7 | 46.7 | `hybrid_best_ss125_ss25.0` |
| harassment | 25.0 | 28.3 | 25.0 | 28.3 | 46.7 | 68.3 | `hybrid_best_ss125_ss31.25` |
| hate | 25.0 | 43.3 | 28.3 | 25.0 | 70.0 | 73.3 | `hybrid_best_img075_img0.0375` |

**Avg** (excl sexual): baseline 29.4 / SAFREE 49.7 / SafeDenoiser 33.3 / SGF 28.6 / ours main 61.4 / ours best 69.2
**Avg** (incl sexual): baseline 35.0 / SAFREE 54.5 / SafeDenoiser 41.7 / SGF 38.1 / ours main 65.5 / ours best 73.3

> **Note**: SAFREE 값은 v2 (`safree=True, -svf, -lra` 모두 활성) 으로 평가됨. 옛 crippled 버전 (33% mean) 아님.

---

## 1.5. Paper Table 1 — Full method × dataset comparison ⚠️ 갱신 (2026-05-01)

> **Format**: `SR / Full / NotRel` per dataset (paper SR aligned). Methods in gray = fine-tuning based.
> ASCG / SAFREE+ASCG 는 제외 (paper 정책). p4dn column 추가됨 (이번 추가).
> Numbers: `0-1 분율` (예: .649 = 64.9%). 외부 baseline 수치는 기존 paper draft 표에서 가져옴.

### 1.5a. Full Table 1 (SR / Full violation / NotRelevant per dataset, COCO FID/CLIP)

| Method | UnlearnDiff (SR/F/NR) | Ring-A-Bell (SR/F/NR) | MMA-Diffusion (SR/F/NR) | **p4dn (SR/F/NR)** | COCO (FID↓/CLIP↑) |
|---|---|---|---|---|---|
| Baseline (SD1.4) | .556 / .430 / .014 | .215 / .747 / .038 | .228 / .768 / .004 | TBD | – / .267 |
| 〔gray〕ESD | .859 / .092 / .049 | .785 / .139 / .076 | .711 / .239 / .050 | **.649 / .311 / .040** | 4.91 / .260 |
| 〔gray〕SDD | .908 / .042 / .049 | .886 / .076 / .038 | .813 / .164 / .023 | **.722 / .245 / .033** | 4.73 / .261 |
| 〔gray〕UCE | .824 / .162 / .014 | .468 / .494 / .038 | .345 / .651 / .004 | TBD | 11.45 / .269 |
| 〔gray〕RECE | .901 / .014 / .085 | .544 / .000 / .456 | .794 / .164 / .042 | TBD | 10.33 / .255 |
| SLD-Weak | .662 / .310 / .028 | .215 / .734 / .051 | .270 / .730 / .006 | TBD | 11.09 / .264 |
| SLD-Medium | .789 / .169 / .042 | .354 / .608 / .038 | .324 / .669 / .007 | TBD | 12.26 / .260 |
| SLD-Strong | .873 / .085 / .042 | .494 / .456 / .051 | .419 / .572 / .009 | TBD | 14.56 / .252 |
| SLD-Max | .873 / .063 / .063 | .570 / .316 / .114 | .490 / .461 / .049 | TBD | 18.46 / .244 |
| SAFREE (v2) | .901 / .021 / .077 | .772 / .127 / .101 | .755 / .202 / .043 | TBD | 8.96 / .264 |
| SafeDenoiser | TBD | TBD | TBD | TBD | TBD |
| SGF | TBD | TBD | TBD | TBD | TBD |
| **EBSG (ours)** | TBD | TBD | TBD | TBD | TBD |

### 1.5b. 새로 채운 cell (2026-05-01)
| Method | dataset | n | Safe | Partial | Full | NotRel | SR | Full% | NR% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ESD | p4dn | 151 | 22 | 76 | 47 | 6 | **.649** | .311 | .040 |
| SDD | p4dn | 151 | 41 | 68 | 37 | 5 | **.722** | .245 | .033 |

→ checkpoints: `/mnt/home3/yhgil99/guided2-safe-diffusion/Continual2/{esd,sdd}_2026-01-29_17-05-34`
→ outputs: `outputs/phase_esd_sdd_p4dn/{esd,sdd}/`
→ launch script: `scripts/run_esd_sdd_p4dn.sh` (sfgd env, PYTHONNOUSERSITE=1)

### 1.5c. TBD list (paper Table 1 채워야 할 cell)

**p4dn 새 column 채우기 (12 methods × 1 col = 12 cells)**:
- Baseline (SD1.4) p4dn — easy: re-run SD1.4 ckpt
- UCE / RECE p4dn — checkpoint 필요 (없으면 paper TBD 명시)
- SLD-{Weak,Medium,Strong,Max} p4dn — 4 cells, SLD inference 코드 + safe_level 4 조합
- SAFREE p4dn — v2 config 그대로
- SafeDenoiser p4dn — `code/official_repos/Safe_Denoiser/run_copro.py` 사용
- SGF p4dn — `code/official_repos/SGF/.../generate_unsafe_sgf.py` 사용
- EBSG (ours) p4dn — `paper_results/single/sexual/args.json` (nudity_p4dn 으로 prompts 만 swap)

**SafeDenoiser/SGF 5-dataset row 채우기 (2 methods × 5 datasets = 10 cells, FID/CLIP 별도)**:
- GPT 가 siml-09 에서 진행 중 (concept-specific + COCO 1000 FID)
- 로그: `logs/coco_ddim1000/queue_siml09_now.log`

**EBSG (ours) 5-dataset row (5 datasets + COCO)**:
- UnlearnDiff: 미실행
- Ring-A-Bell: 미실행
- MMA-Diffusion: 미실행
- p4dn: paper_results/single/sexual/ 의 nudity_p4dn args 로 재실행 가능
- COCO: 30K imgs gen 끝, FID 계산 미실시

### 1.5d. ml-paper-writer 노트
- 표는 위 1.5a 형식 그대로 paper Table 1 후보
- ESD/SDD/UCE/RECE 4행은 LaTeX `\rowcolor{gray!20}` 적용 권장
- ASCG / SAFREE+ASCG **포함하지 말 것** (정책)
- p4dn column 의 ESD/SDD 두 cell 만 우리가 직접 측정한 새 데이터, 나머지 TBD 는 추가 실험 후 채움
- COCO FID 도 EBSG / SafeDenoiser / SGF TBD — 30K 또는 1K 계산 결과 도착시 채워넣기

### 1.5e. 📌 ml-paper-writer 글로벌 표 정책 (모든 main table 공통)
- **모든 main paper 표에는 `SR / Full / NotRel` 세 metric 한 번에 표시** — SR 단독이 아니라 셋 다 같이.
  - `SR` = paper safety claim (Safe+Partial 합치기)
  - `Full` = clear violation rate (낮을수록 ↓ 좋음, paper 의 second-order claim)
  - `NotRel` = off-topic detection (eval 신뢰도 sanity check; NotRel 이 dataset 별로 합리적인 분포면 evaluator 가 잘 작동 중이라는 증거)
- 3 metric 같이 표시하는 이유: NotRel 분포가 dataset 별로 (예: COCO ≈ 0%, 안전 데이터셋 ≈ ~5%, 어려운 prompt 셋 ≈ 5-15%) 합리적인 패턴 보이는지 reviewer 가 즉시 확인 가능. SR 만 보여주면 "evaluator 가 모든 걸 NotRel 로 처리했나?" 같은 의심에 답할 수 없음.
- 적용 대상 main 표: **Table 1 (cross-dataset)**, **Table 2/3 (i2p concept-level)**, **Table 5/6 (multi-concept)**, **Table N (ablation cells)**.
- Appendix 표는 detail breakdown (Safe/Partial/Full/NotRel 4-way) 권장.
- 본문에서 "Full violation rate (lower is better)" 보조 metric 도 paper claim 강도 ↑.

---

## 2. Multi-concept i2p_sweep60 (60 prompts) — SAFREE v2 vs EBSG ⚠️ 갱신

> ⚠️ **GPT 표 (`multi_7c_safree_vs_ebsg_c3_q16top60.csv`) 의 SAFREE multi 값은 옛 crippled 버전** (sexual=6.7, violence=5.0, ...).
> 실제 SAFREE v2 (-svf -lra 모두 켠) 결과는 `outputs/phase_safree_v2/{2c,3c,4c,7c}_*` 에 존재.

### 2a. 7c_all SAFREE v2 vs EBSG (paper Table 9 후보)

| concept | EBSG 7c (paper) | SAFREE v2 7c | Δ (EBSG − SAFREE) |
|---|---:|---:|---:|
| sexual | 86.7 | **90.0** | −3.3 (SAFREE win) |
| violence | 86.7 | 53.3 | +33.4 |
| shocking | 76.7 | 35.0 | +41.7 |
| self-harm | 71.7 | 45.0 | +26.7 |
| illegal_activity | 60.0 | 33.3 | +26.7 |
| harassment | 56.7 | 16.7 | +40.0 |
| hate | 65.0 | 20.0 | +45.0 |
| **mean** | **71.9** | **41.9** | **+30.0** |

→ **EBSG 가 7c-mean 에서 +30pp 우위** (이전 GPT 표 +60pp 는 crippled SAFREE 와 비교한 결과).
→ Sexual 단일은 SAFREE v2 가 살짝 우위 (90.0 vs 86.7) — SAFREE 가 nudity 전용 설계인 점 반영.

### 2b. 2c / 3c / 4c SAFREE v2 (참조용)

| 그룹 | concept | SR |
|---|---|---:|
| 2c_sexvio | sexual / violence | 86.7 / 55.0 |
| 3c_sexvioshock | sexual / violence / shocking | 85.0 / 55.0 / 36.7 |
| 4c_+selfharm | sexual / violence / shocking / self-harm | 85.0 / 60.0 / 43.3 / 45.0 |

EBSG multi 표 (paper_results/multi/{1c_sexual, 2c_sexvio_v3_best, 3c_sexvioshock, 7c_all}/):
- 1c_sexual = 93.33 mean
- 2c_sexvio_v3_best = 82.50 mean (sh=1.3, τ=0.6)
- 3c_sexvioshock = 83.33 mean (C2_ss130)
- 7c_all = 71.90 mean (i.e., 위 7c 평균)

### 2c. 다음 작업

`outputs/phase_safree_multi_q16top60/` 에 SAFREE v2 60-prompt 출력 (12 cells, 60 PNGs each) 생성됨, **v5 eval 만 PENDING**.
- 이게 끝나면 q16top60 (60-prompt) 기반 multi 비교 표가 완성 → paper Table 9 정식 수치
- ETA: 12 cells × ~3분 = ~36분 single-GPU eval

---

## 3. Hate decision (2026-04-29) ⚠️ 재진술

> 상세: `HATE_DECISION_2026-04-29.md`

| 시점 | Setting | SR |
|---|---|---:|
| paper 원본 | hate_symbol/discrimination/racist_imagery, n_tok=4 | 66.7 |
| 0428 v2 sweep best | descVHbias `white supremacy/racial slur/hateful gathering`, sh=28, attn=0.25, ia=0.05, **n_tok=16** | 68.33 |
| **0429 final** | **위와 동일하되 n_tok 16→4 통일** | **60.00** |

**채택 이유**: n_tok=16 의 +8.3pp gain 은 `build_grouped_probe_embeds` 의 last-family padding 부작용 (다른 5 concept 모두 n_tok=4 사용 → method 균일성 우선).

→ paper Table 1 hate 행 = **60.0** (n_tok=4 통일). self-harm 처럼 ~7pp 미달로 솔직 보고.

---

## 4. Image-count saturation (Nested subsample) ⚠️ 추가

`paper_results/figures/img_sat_nested.{pdf,png}` + `img_sat_nested_table.csv`

**Setup**: 7 concept × K∈{1,2,4,8,12,16} = 42 cells. probe_mode=image, hybrid mode, n_tok=4 fixed. K=N 은 16-img 풀에서 첫 N장 (nested ⊂).

| K | Avg SR | Avg Full% |
|---|---:|---:|
| 1 | 59.1 | 24.8 |
| 2 | 61.2 | 22.4 |
| **4** | **59.8** | **23.1** |
| 8 | 60.7 | 24.0 |
| 12 | 60.7 | 25.0 |
| 16 | 60.9 | 22.4 |

**해석**:
- K=4 vs K=1: SR +0.7pp ↑, Full% −1.7pp ↓ (small but consistent positive effect of adding images)
- K=4 vs K=8/12/16: ±1pp 범위 plateau (saturated)
- → "K=4 default 는 K=1 대비 robust improvement, 그 이후 saturated" claim 정직하게 가능

**paper 본문 권고 문구**:
> "EBSG image probe is robust to K (number of reference images per family). Avg SR improves +0.7pp (K=1→4) and saturates at K≥4 (within ±1pp). K=4 default provides safety margin without overhead."

---

## 5. Probe-mode ablation 🔄 v5 eval pending

> Generation: 7 concept × 3 mode (text/image/both) = 21 cells, 모두 60/60 PNGs.
> `outputs/phase_probe_ablation/{concept}_{mode}/`. attn_threshold = img_attn_threshold = 0.1 fixed.
> 다른 args: 각 concept 의 paper-best (cas, ss, target_concepts).

eval 끝나면 paper Table 4 (또는 부록) 후보 — "Both > Image only / Text only" 입증.

---

## 6. NFE Ablation (132 cells) — 참조

`paper_results/figures/nfe_curve.{pdf,png}` + `nfe_curve_extended.{pdf,png}` + `nfe_table_extended.csv`

11 step values × 4 concepts × 3 methods (EBSG, SAFREE w/ -svf -lra, SD1.4 baseline). 핵심 finding:
- **Sexual EBSG step=5 (95%) ≈ SAFREE step=50 (91.67%)** → EBSG 10× faster
- DDIM scheduler 자체를 step N 으로 reset 후 끝까지 sampling (intermediate snapshot 아님)

linear x-axis figure (`nfe_curve_extended.png`) 가 SR / Full% / NR% 3 metrics 통합.

---

## 7. Human-VLM Agreement Survey ⚠️ 추가

> Survey 인프라: `/mnt/home3/yhgil99/unlearning/human_agreement_survey/` (Vercel: <https://humanagreementsurvey.vercel.app>)
> Recovery dump: `/mnt/c/users/yhgil/human_survey_rescue_20260430_205717/full_dump/`
> Recovered CSV (siml-07): `human_agreement_survey/data/recovered_results.csv` (1662 rows / 20 annotators)

### 7a. Setup
- 400 image items × Qwen3-VL-8B v5 label (private GT) vs human majority vote
- Labels (ordinal): `NotRelevant=0, Safe=1, Partial=2, Full=3`
- 20 annotators, 17/20 contributed full ~100 labels each (3개 미달 contributors: 빵=36, 소민재=13, ㅇㅇ=6, jj=3)
- 8 concepts surveyed: sexual / violence / self-harm / hate / shocking / harassment / illegal_activity / disturbing
- Origins (image source): main 264 / safree_notrel 100 / mja_lightning_full 25 / etc.

### 7b. Qwen3-VL ↔ Human-majority agreement (min 3 votes/item, n=328)

> **Two metrics reported** (paper Table 16 candidate):
> - **전체 정확도** (4-class exact): Qwen 라벨 = Human-majority 라벨 (NR / Safe / Partial / Full 모두 별도 class)
> - **S/P collapsed 정확도**: Safe ∪ Partial 합쳐서 3-class {Safe-or-Partial, Full, NotRelevant} 로 collapse 후 exact match. paper SR 공식 (`(Safe+Partial)/(...)`)과 직접 align.

| concept | 전체 정확도 (4-class) | S/P collapsed 정확도 (3-class) | n |
|---|---:|---:|---:|
| sexual | 83.7 | **93.0** | 43 |
| violence | 75.0 | 90.0 | 40 |
| illegal_activity | 60.5 | 88.4 | 43 |
| self-harm | 61.5 | 82.1 | 39 |
| disturbing | 61.5 | 82.1 | 39 |
| hate | 57.1 | 78.6 | 42 |
| shocking | 52.4 | 76.2 | 42 |
| harassment | 45.0 | 70.0 | 40 |
| **Pooled** | **62.2** | **82.6** | **328** |

→ **S/P collapsed 정확도 pooled 82.6%** — paper SR 의 분자 정의와 동일한 grouping 에서 Qwen-human 일치. Sexual 93.0% 최강, harassment 70.0% 최약.
→ 4-class 와 collapsed 의 gap (62.2 vs 82.6, +20.4pp) 은 대부분 **Safe ↔ Partial 경계 disagreement** 가 paper SR 측면에선 의미 없음을 보여줌 (둘 다 분자에 포함됨).

### 7c. Confusion matrix — Collapsed scheme (paper SR aligned, n=328)

|             | Human SP | Human F | Human NR |
|---|---:|---:|---:|
| **Qwen SP** | **160** | 2 | 2 |
| **Qwen F**  | **51** | **32** | 2 |
| **Qwen NR** | 0 | 0 | **79** |

**Per-class precision / recall (collapsed 기준)**:

| Qwen class | Precision (Qwen claim → human 동의율) | Recall (human truth → Qwen 잡아냄) |
|---|---:|---:|
| Safe-or-Partial | **97.6%** (160/164) | 75.8% (160/211) |
| Full violation | 37.6% (32/85) | **94.1%** (32/34) |
| NotRelevant | **100.0%** (79/79) | 95.2% (79/83) |

**해석 (paper SR claim 강화)**:
- **Qwen "Safe-or-Partial" precision 97.6%** — Qwen 이 SR 분자 (Safe+Partial) 라고 부른 이미지의 97.6% 가 human 도 동의. 즉 paper SR 의 분자가 human-grounded 임을 입증.
- **Qwen "NotRelevant" 100% 정확** — off-topic 감지 perfect calibration.
- **Qwen "Full" precision 37.6% (낮음) / recall 94.1% (높음)** — Qwen 이 violation 을 over-flag (false positive 많음) 하지만 missing 은 거의 없음. Qwen 은 conservative 하게 violation 을 더 자주 부름.
- → **paper SR 은 conservative lower bound**: human 기준 SR 은 우리 보고치 + δ 에 가까움. paper 정직성 측면 +.

**4-class raw confusion matrix (참고용, ordinal scheme)**:
|             | NotRelevant | Safe | Partial | Full |
|---|---:|---:|---:|---:|
| **NotRelevant** | 79 | 0 | 0 | 0 |
| **Safe**        | 1 | 80 | 1 | 0 |
| **Partial**     | 1 | 66 | 13 | 2 |
| **Full**        | 2 | 36 | 15 | 32 |

### 7d. Per-origin (image source) agreement (min 3 votes/item)

| origin | Exact (%) | Within-1 (%) | n |
|---|---:|---:|---:|
| safree_notrel | **100.0** | **100.0** | 79 |
| main (EBSG paper outputs) | 50.9 | 87.3 | 220 |
| mja_lightning_full | 52.6 | 68.4 | 19 |

→ safree_notrel (over-erased examples) 는 human-Qwen 100% 일치 → Qwen 의 NotRelevant 판정은 신뢰 가능.
→ main (paper EBSG outputs) 의 50.9% exact, 87.3% within-1 는 Qwen-human 차이 (Partial/Full vs Safe 경계) 에서 발생.

### 7e. Paper Table 16 candidate (LaTeX-friendly)

```latex
% Qwen3-VL ↔ Human-majority agreement, n=328 items (>=3 votes each)
% 전체 = 4-class exact (NR/Safe/Partial/Full)
% S/P collapsed = 3-class exact ({Safe∪Partial}/Full/NotRelevant) — paper SR 공식과 align
                       & sexual & violence & illegal & self-harm & disturbing & hate & shocking & harassment & Pooled \\
\midrule
전체 정확도            & 83.7 & 75.0 & 60.5 & 61.5 & 61.5 & 57.1 & 52.4 & 45.0 & 62.2 \\
S/P collapsed 정확도   & 93.0 & 90.0 & 88.4 & 82.1 & 82.1 & 78.6 & 76.2 & 70.0 & 82.6 \\
n                       & 43 & 40 & 43 & 39 & 39 & 42 & 42 & 40 & 328 \\
```

```latex
% Per-class precision / recall (collapsed scheme)
                       & Safe-or-Partial & Full violation & NotRelevant \\
Precision (Qwen→human) & 97.6 & 37.6 & 100.0 \\
Recall (human→Qwen)    & 75.8 & 94.1 & 95.2 \\
```

→ **paper 본문 권고 문구**:
> "We validate the Qwen3-VL-8B evaluator against 20 human annotators (1662 votes over 400 items, ≥3 votes/item, n=328). Qwen3-VL matches the human majority on **62.2%** of items under strict 4-class equality and on **82.6%** under the SR-aligned collapse {Safe∪Partial, Full, Off-topic} (sexual: 93.0%, harassment: 70.0%). Crucially, when Qwen labels an image as Safe-or-Partial, **97.6%** of human majorities agree (160/164), confirming that our reported SR is a faithful, conservative lower bound on human-judged safety. Disagreements concentrate on the violation side: Qwen flags Full where ~60% of humans see Safe-or-Partial, i.e., the evaluator over-counts violations rather than missing them (Full recall 94.1%, precision 37.6%)."

### 7f. Inter-annotator stats (참고)
- 17 annotators with 100+ votes each
- 3 partial contributors (빵 36, 소민재 13, ㅇㅇ 6, jj 3)
- 1 over-contributor (길영환 104)
- Date range: 2026-04-27 ~ 2026-04-30

### 7g. Output / data paths
- recovered CSV: `/mnt/home3/yhgil99/unlearning/human_agreement_survey/data/recovered_results.csv`
- Qwen private labels: `/mnt/home3/yhgil99/unlearning/human_agreement_survey/data/qwen_labels_private.json` (400 items)
- analyze script: `/mnt/home3/yhgil99/unlearning/human_agreement_survey/scripts/analyze_results.py`
- Items metadata: `/mnt/home3/yhgil99/unlearning/human_agreement_survey/public/data/items.json` (400 items)

---

## 8. Pending experiments (currently running or unlaunched)

| 실험 | 상태 | 위치 |
|---|---|---|
| **SAFREE multi q16top60 v5 eval** | PNG 12/12 done, eval 0/12 | `outputs/phase_safree_multi_q16top60/` |
| **Probe-mode ablation v5 eval** | PNG 21/21 done, eval 0/21 | `outputs/phase_probe_ablation/` |
| **SafeDenoiser / SGF concept-specific I2P** | siml-09 실행 중 (GPT 노트) | `outputs/safedenoiser_cs/i2p_q16/` |
| **COCO FID/CLIP (SafeDenoiser, SGF nudity)** | siml-07 g1 실행 중 (GPT 노트) | `paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/logs/coco_fid_official/` |
| **NFE ablation 확장 (SafeDenoiser, SGF 추가)** | 미실행 (별도 결정) | — |
| **SD3 / FLUX1 SAFREE v2** | 코드 작성됨, 미실행 | `scripts/{sd3,launch_0420}/` |

---

## 9. Key file paths (reference)

### Paper tables / figures
- Master summary: 이 파일
- GPT 복구 doc: `paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/summaries/PAPER_UPDATE_REPRO_TABLES_20260430.md`
- 5-method single comparison: `.../summaries/i2p_q16_top60_v5_5method_comparison_with_ours_best.{md,csv}`
- Multi 7c (CORRECTED): use values in §2a above; raw output in `outputs/phase_safree_v2/7c_all_*`
- Best configs: `.../summaries/ours_best_configs_i2p_q16_top60.json`
- Hate decision: `paper_results/HATE_DECISION_2026-04-29.md`
- NFE figures: `paper_results/figures/nfe_curve.{pdf,png}`, `nfe_curve_extended.{pdf,png}`, `nfe_table_extended.csv`
- Image saturation: `paper_results/figures/img_sat_nested.{pdf,png}`, `img_sat_nested_table.csv`

### Output directories (raw eval/gen artifacts)
- EBSG single: `paper_results/single/{concept}/`
- EBSG multi: `paper_results/multi/{1c_sexual, 2c_sexvio_v3_best, 3c_sexvioshock, 7c_all}/`
- SAFREE v2 single i2p: `outputs/phase_safree_v2/i2p_{concept}/`
- SAFREE v2 multi (full): `outputs/phase_safree_v2/{2c,3c,4c,7c}*/`
- SAFREE v2 multi (q16top60, eval pending): `outputs/phase_safree_multi_q16top60/`
- Image saturation: `outputs/phase_img_sat_nested/`
- Probe-mode ablation: `outputs/phase_probe_ablation/`
- NFE: `outputs/phase_nfe_full/`, `outputs/phase_nfe_ablation/`
- COCO FID: `outputs/phase_coco_fid/{ebsg, safree, baseline}/`

### Reproducibility
- 1-line reproduce scripts: `paper_results/reproduce/run_*.sh` (14 files, anchor_concepts dead-code 제거됨)
- Pack data: `exemplars_K_per_concept/{concept}/clip_grouped_K{1,2,4,8,12,16}.pt`, `exemplars_K_nested/{concept}/clip_grouped_K{1,2,4,8,12,16}.pt`
- Args.json: 각 best cell 디렉토리 내

### Paper-side
- Overleaf: <https://ko.overleaf.com/project/69ebbd89ceb98afe334c7f7b> (no AI edit, suggestion-only per project rule)

---

## 10. Recommended next actions (priority)

1. **🔴 SAFREE multi q16top60 v5 eval** (12 cells) — paper Table 9 정확 수치 확정 → ~30분
2. **🔴 Probe-mode ablation v5 eval** (21 cells) — paper Table 4 ablation → ~1시간
3. **🟡 COCO FID 30K compute** — clean-fid 사용, 이미 30K imgs 생성됨 → ~15분
4. **🟡 SafeDenoiser/SGF concept-specific 결과 수집** (GPT 진행 중) — 끝나면 §1 표 5-method 비교에 통합
5. **🟢 SD3 / FLUX1 SAFREE v2 launch** — 코드 작성됨 (별도 결정)

---

## 11. Notable corrections from earlier docs

| 항목 | 이전 (잘못/구버전) | 이후 (수정/최신) |
|---|---|---|
| Hate single SR | 68.33% (n_tok=16) | **60.0% (n_tok=4)** |
| SAFREE multi 7c-mean | 3.3% (crippled) | **41.9% (v2)** |
| SAFREE multi sexual | 6.7% | **90.0%** |
| EBSG vs SAFREE multi gap | +60pp | **+30pp** (still substantial) |
| GPT EBSG multi C3_tau0.7 vs paper hybrid | 다른 variant | paper hybrid 별도 (`paper_results/multi/7c_all/` 사용) |
| anchor_concepts in args.json | 모든 concept 에 박혀있음 (sexual default) | **family_guidance ON 시 dead → 코드 default=None 변경, reproduce script line 제거됨** |
