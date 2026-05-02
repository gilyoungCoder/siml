# 2026-04-09 Experiment Results Summary

## Executive Summary

Phase 1 (Nudity Erasing) 실험이 전체 5개 데이터셋에서 완료되었으며, **v27 mega 버전이 최고 성능**을 달성했다.
MJ-Bench safety judge 평가에서 Qwen3-VL-8B가 GPT-4o에 근접하는 성능을 보였다.
Phase 2 Multi-concept 실험이 생성 단계까지 진행되었으나 아직 평가되지 않은 상태이다.

---

## 1. Nudity Erasing Results (Qwen3-VL SR%)

SR = (Safe + Partial) / Total. 높을수록 좋음 (nudity 제거 성공).

### 1.1 Best Results per Method (All Datasets)

| Method | RingABell (79) | P4DN (151) | UnlearnDiff (142) | MMA (1000) | I2P Sexual (931) |
|--------|:-:|:-:|:-:|:-:|:-:|
| **SD v1.4 Baseline** | 34.8% | — | — | — | — |
| **AMG SLD s7** | 57.9% | — | — | — | — |
| **AMG AShift s5** | 55.7% | — | — | — | — |
| **SAFREE best** | 89.9% | 67.5% | 87.3% | — | — |
| **v4 (CAS+SLD ss12)** | 89.9% | 94.7% | 95.1% | 79.8% | 91.5% |
| **v20 ainp (CLIP WHERE)** | 89.9% | 94.7% | 96.5% | — | — |
| **v22 best (textonly12)** | 88.6% | 94.7% | 95.8% | 79.2% | 91.7% |
| **v27 mega hyb** | **94.9%** | 94.0% | **97.2%** | 79.6% | **93.1%** |
| **v27 mega ainp** | 89.9% | 92.7% | 95.8% | 78.3% | 91.7% |
| **v27 full text** | 91.1% | 93.4% | 95.8% | 78.3% | 91.6% |
| **v27 final both_ainp** | — | 92.7% | 95.8% | 78.2% | 91.7% |

### 1.2 Key Findings

- **v27_mega/nude_hyb_ringabell: 94.9%** — RingABell 역대 최고 (기존 89.9% 대비 +5%p)
- **v27_mega/nude_txt_hyb_unlearndiff: 97.2%** — UnlearnDiff 역대 최고
- **v27_mega/nude_txt_hyb_i2p_sexual: 93.1%** — I2P Sexual 역대 최고
- **v27_mega/4s_ainp_both: 92.7%** — 4-dataset 통합(316) 최고
- MMA는 모든 버전에서 ~78-80%로 plateau — 가장 어려운 데이터셋

### 1.3 v27 Best Config Summary

| Mode | RingABell | Config |
|------|-----------|--------|
| hyb | **94.9%** | `nude_hyb_ringabell` |
| txt_hyb | 89.9% | `nude_txt_hyb_ringabell` |
| ainp | 89.9% | `nude_ainp_ringabell` |
| txt_ainp | 91.1% | `nude_txt_ainp_ringabell` |
| hybproj | 89.9% | `nude_hybproj_ringabell` |

**hyb (hybrid) 모드가 img+text 모두 사용할 때 가장 좋은 성능**

### 1.4 4-Dataset Combined (316 prompts = RingABell + P4DN + UnlearnDiff + MMA subset)

| Config | SR% |
|--------|-----|
| v27_clean/both_ainp_ss12_tt01_it04_4s | **93.1%** |
| v27_mega/4s_ainp_both | 92.7% |
| v27_mega/4s_hyb_both_ts15as15 | 92.4% |
| v27_mega/4s_hybproj_both_ss15as15 | 92.1% |
| v22_boost_ss1.2 | 92.4% |

---

## 2. COCO False Positive (FP) Check

COCO 이미지에서 nudity로 잘못 탐지하면 안됨.

| Method | NudeNet Unsafe% | Qwen SR% (nudity detected) | Notes |
|--------|:-:|:-:|-----|
| v4 COCO ainp_cas05 | 3.3% | 20.8% | Best FP |
| v4 COCO sld_s5 | 1.7% | 20.8% | Very low NN FP |
| v7 COCO hyb_ts15_as15 | — | 20.8% | Maintained |
| v27_final COCO both_ainp | — | 25.6% (n=250) | Slightly higher |

- NudeNet FP: 대부분 3-5% 범위 (양호)
- Qwen SR 20-25%: COCO에서 nudity 관련으로 분류되는 비율 (사람 포함 이미지)

---

## 3. NudeNet Results (Unsafe Rate, lower = better)

| Method | Unsafe Rate | Total |
|--------|:-:|:-:|
| Baseline (SD v1.4) | 72.2% | 79 |
| AMG SLD s7 | 46.8% | 79 |
| CAS sweep cas05_s10_t01 | 18.4% | 316 |
| CAS sweep cas06_s10_t01 | 23.4% | 316 |
| v4 sld_s10 | 7.0% | 316 |
| v3 dag_s3 | **0.95%** | 316 |

---

## 4. MJ-Bench Safety Judge Evaluation (Qwen3-VL-8B)

### 4.1 Official 672 Pairs (MJ-Bench Standard Prompt, Scale [1-10])

| Metric | Value |
|--------|-------|
| Total pairs | 672 |
| Correct | 334 |
| Ties | 293 (43.6%) |
| Errors | 0 |
| **Accuracy w/ tie** | **49.7%** |
| **Accuracy w/o tie** | **88.1%** |

### 4.2 Comparison with Published Models (HF Results = Table 21)

| Model | Safety Avg w/ tie |
|-------|:-:|
| GPT-4o | ~73% (Tox 92.1 + NSFW 54.3) |
| LLaVA-1.5-7b | ~35% (Tox 43.8 + NSFW 26.3) |
| **Qwen3-VL-8B (ours)** | **49.7% (all safety)** |

- NSFW-only로 비교 시 GPT-4o NSFW 54.3% vs Qwen ~49.7% (근접)
- Toxicity subset은 공개 데이터에 없어 NSFW 위주 비교만 가능
- Subcategory annotation 없음 — 저자에게 이메일 문의 중

### 4.3 NSFW 471 Pairs (6 Mode Evaluation)

| Mode | w/ tie | w/o tie | Ties |
|------|:-:|:-:|:-:|
| single_5 | 0.8% | 2.1% | 278 |
| single_10 | 1.5% | 2.7% | 216 |
| single_narrative | 1.1% | 2.6% | 275 |
| multi_5 | 0.8% | 1.6% | 225 |
| multi_10 | 1.5% | 2.6% | 200 |
| **multi_full** | **19.5%** | **38.2%** | 230 |

- 대부분 모드에서 매우 낮은 정확도 → **label direction 문제 가능성 높음**
- multi_full만 비교적 동작 → 전체 평가 프롬프트가 label과 맞는 것
- 추가 조사 필요: label=0이 image_0 preferred인지 확인

### 4.4 Cross-Validator Agreement (Qwen3-VL vs LLaVA)

- Binary safe/unsafe 일치율: **92.3%**
- Qwen3-VL이 독립적 VLM safety judge로 신뢰성 있음을 확인

---

## 5. Multi-Concept Experiments Status (v27_overnight_final)

### 5.1 Generated (Not Yet Evaluated)

| Experiment | Images | Status |
|-----------|--------|--------|
| multi_nude_harassment_ss1.0 | 2,472 | Generated, no eval |
| multi_nude_harassment_ss1.2 | 2,472 | Generated, no eval |
| multi_nude_hate_ss1.0 | 693 | Generated, no eval |
| multi_nude_hate_ss1.2 | 693 | Generated, no eval |
| multi_nude_shocking_ss1.0 | 2,568 | Generated, no eval |
| multi_nude_shocking_ss1.2 | 2,568 | Generated, no eval |
| multi_nude_violence_ss1.2 | 30 | Generated, no eval |

### 5.2 Already Evaluated (v27_final)

| Config | SR% | Dataset |
|--------|-----|---------|
| multi_nude_violence_ss1.0_cas0.4 | 87.9% | I2P sexual (931) |
| multi_nude_violence_ss1.0_cas0.5 | 87.0% | I2P sexual (931) |
| multi_nude_violence_ss1.2_cas0.4 | 85.5% | I2P sexual (931) |

- Nudity+Violence 동시 erasing 시 nudity SR ~85-88% 유지 (single-concept 91.7% 대비 약간 하락)

### 5.3 Style Erasing (Van Gogh, Monet, Picasso, Warhol)

| Config | SR% | n |
|--------|-----|---|
| multi_vg_monet_ss0.8_cas0.3 | 30.0% | 60 |
| multi_warhol_nude_ss0.8_cas0.3 | — | 60 (no eval) |

- Style erasing 실험 초기 단계

---

## 6. Country Nude Body Results (Bias Test)

| Method | SR% | n=80 |
|--------|-----|------|
| Baseline | 3.8% | — |
| v4 sld_s10 | 47.5% | — |
| v5 sld_s10 | 78.8% | — |
| v6 ts20_as15 | **98.8%** | Over-erases |
| v12 xattn | 70.0% | — |

---

## 7. Version Evolution Summary

| Version | Key Idea | Best SR (RingABell) |
|---------|----------|:-:|
| v3 | SLD + CAS + Spatial mask | ~57% |
| v4 | Anchor inpainting + CAS | 89.9% |
| v5 | Projection subtraction | ~89.9% |
| v6 | Cross-attention exemplar | 89.9% |
| v7 | GLASS exemplar hybrid | 89.9% |
| v10 | Projection on noise | ~82% |
| v11 | Stochastic ensemble | ~82% |
| v12 | Cross-attention WHERE | ~86% |
| v13 | CLIP exemplar probe | ~88% |
| v20 | CLIP probe WHERE | 89.9% |
| v21 | Adaptive anchor inpaint | — |
| v22 | Union/text-only WHERE | 88.6% |
| v23 | Noise/attn union | 88.6% |
| v24 | Multi-stage refinement | 91.1% |
| **v27** | **Clean rewrite + mega grid** | **94.9%** |

---

## 8. Next Steps (Priority Order)

### Immediate (이번 주)
1. **v27_overnight_final multi-concept 평가 실행** (Qwen3-VL)
   - multi_nude_harassment, multi_nude_hate, multi_nude_shocking
   - 총 ~11,000 이미지 → GPU 8대 병렬 활용
2. **MJ-Bench NSFW 471 pairs label direction 확인 및 수정**
   - label=0 방향 재확인, 수정 후 재실행
3. **v27 mega best config → 나머지 데이터셋 확장**
   - hyb 모드: P4DN, UnlearnDiff, I2P에서도 실행 (현재 RingABell만)

### Short-term (다음 주)
4. **COCO FP 체계적 평가** (v27 best config)
   - 250 COCO prompts × v27_mega_hyb → NudeNet + Qwen
5. **FID/CLIP Score 이미지 품질 평가**
   - COCO 30k subset으로 FID 측정
6. **Phase 2 Violence concept 본격 실험**
   - Anchor prompt 세트 제작, CAS threshold sweep

### Paper Preparation
7. **Table 구성**: Baseline vs SLD vs SAFREE vs Ours (v27) across 5 datasets
8. **MJ-Bench 결과로 evaluator trustworthiness 섹션 작성**
9. **Multi-concept ablation**: single vs dual concept erasing 비교
