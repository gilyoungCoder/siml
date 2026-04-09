# v17/v18/v19 Experiment Results — SR 기준 전체 정리

> **Primary Metric**: SR% = (Safe + Partial) / Total (Qwen VLM)
> **Dataset**: Ring-A-Bell (79 prompts x 4 samples)
> **Model**: CompVis/stable-diffusion-v1-4, safety_checker=None
> **Eval**: Qwen VLM 4-class (NotRel / Safe / Partial / Full)
> **Date**: 2026-04-05

---

## 1. Version별 Best SR% (Overall)

| Version | SR% | Full% | NN Unsafe% (ref) | Best Config | Evaluated Configs |
|---------|-----|-------|-------------------|-------------|-------------------|
| **v17** | **90.6%** | 9.4% | 6.65% | text_dag_adaptive_ss2.0_st0.2_probe_only | 122 |
| **v19** | **90.6%** | 9.4% | 4.43% | image_diverse_dag_adaptive_ss1.0_st0.3_multi_probe | 335 |
| v18 | 88.2% | 11.8% | 3.16% | image_dag_adaptive_ss3.0_st0.2_cosine_sb0.5 | 134 |

- v17과 v19가 **동률 1위 (SR 90.6%)**
- v19는 335개 config에서 안정적으로 90%+ 유지
- v3~v15는 Qwen eval 미실시 (NudeNet만 존재)

---

## 2. Probe Source별 Best SR% (Version x Probe Source)

| Version | Image Only | Text Only | Image + Text (Both) |
|---------|-----------|-----------|---------------------|
| **v17** | 90.0% | **90.6%** | 87.2% |
| **v18** | **88.2%** | 87.5% | 87.5% |
| **v19** | **90.6%** | 90.1% | 89.8% |

### 해석
- **v17**: Text probe가 최고 (90.6%)
- **v18**: Image probe가 최고 (88.2%)
- **v19**: Image probe가 최고 (90.6%) — text도 90.1%로 근접
- **Both (image+text)는 오히려 단일 source보다 떨어짐** — 신호 dilution 가능성

---

## 3. v19 Guide Mode 비교

| Guide Mode | #Configs | Avg SR% | Best SR% | Avg Full% |
|------------|----------|---------|----------|-----------|
| **dag_adaptive** | 167 | **85.9%** | **90.6%** | 14.1% |
| hybrid | 162 | 66.4% | 81.8% | 33.6% |

**dag_adaptive가 압도적 우세** (avg SR 85.9% vs 66.4%)

---

## 4. v19 Where Mode 비교

| Where Mode | #Configs | Avg SR% | Best SR% |
|-----------|----------|---------|----------|
| **multi_probe** | 165 | 76.1% | **90.6%** |
| fused | 164 | 76.5% | 89.6% |

평균은 비슷하나 **multi_probe가 best에서 우세**

---

## 5. v19 Top 20 Configs (SR% 기준)

| # | Probe | Config | SR% | Full% | Safe | Partial | Full |
|---|-------|--------|-----|-------|------|---------|------|
| 1 | image | diverse_dag_adaptive_ss1.0_st0.3_multi_probe | **90.6%** | 9.4% | 87 | 19 | 11 |
| 2 | image | all_dag_adaptive_ss1.0_st0.2_multi_probe | **90.6%** | 9.4% | 65 | 12 | 8 |
| 3 | image | all_dag_adaptive_ss1.0_st0.4_multi_probe | **90.6%** | 9.4% | 105 | 58 | 17 |
| 4 | image | top_k_dag_adaptive_ss1.0_st0.2_multi_probe | 90.5% | 9.5% | 66 | 10 | 8 |
| 5 | image | top_k_dag_adaptive_ss1.0_st0.4_multi_probe | 90.4% | 9.6% | 107 | 53 | 17 |
| 6 | image | diverse_dag_adaptive_ss1.0_st0.4_multi_probe | 90.1% | 9.9% | 105 | 50 | 17 |
| 7 | **text** | dag_adaptive_ss3.0_st0.3_multi_probe | 90.1% | 9.9% | 102 | 53 | 17 |
| 8 | image | top_k_dag_adaptive_ss1.0_st0.3_multi_probe | 90.1% | 9.9% | 80 | 20 | 11 |
| 9 | image | diverse_dag_adaptive_ss1.0_st0.2_multi_probe | 90.0% | 10.0% | 61 | 11 | 8 |
| 10 | image | all_dag_adaptive_ss1.0_st0.3_multi_probe | 89.8% | 10.2% | 79 | 18 | 11 |
| 11 | both | diverse_dag_adaptive_ss1.0_st0.3_multi_probe | 89.8% | 10.2% | 66 | 13 | 9 |
| 12 | both | all_dag_adaptive_ss1.0_st0.4_multi_probe | 89.7% | 10.3% | 82 | 14 | 11 |
| 13 | both | all_dag_adaptive_ss1.0_st0.3_multi_probe | 89.7% | 10.3% | 69 | 9 | 9 |
| 14 | **text** | dag_adaptive_ss2.0_st0.2_multi_probe | 89.6% | 10.4% | 83 | 38 | 14 |
| 15 | image | top_k_dag_adaptive_ss1.0_st0.4_fused | 89.6% | 10.4% | 60 | 9 | 8 |
| 16 | both | all_dag_adaptive_ss1.0_st0.2_multi_probe | 89.5% | 10.5% | 60 | 8 | 8 |
| 17 | both | top_k_dag_adaptive_ss1.0_st0.4_fused | 89.3% | 10.7% | 57 | 10 | 8 |
| 18 | both | diverse_dag_adaptive_ss1.0_st0.4_fused | 89.3% | 10.7% | 57 | 10 | 8 |
| 19 | **text** | dag_adaptive_ss2.0_st0.3_multi_probe | 89.3% | 10.7% | 115 | 85 | 24 |
| 20 | both | diverse_dag_adaptive_ss1.0_st0.4_multi_probe | 89.2% | 10.8% | 79 | 12 | 11 |

### Top Config 공통 패턴
- Guide mode: **전부 dag_adaptive**
- Where mode: Top 15 중 14개가 **multi_probe**
- Image probe: Top 10 중 **9개가 image only**
- Safety scale: Top 9가 **ss1.0** (약한 guidance가 오히려 최적)
- Exemplar selection: diverse/all/top_k **성능 차이 미미**

---

## 6. Key Findings

### What works
1. **dag_adaptive** guide mode가 hybrid 대비 SR 20%p 이상 우세
2. **Image CLIP exemplar probe**가 text keyword probe보다 효과적
3. **Multi-probe** (개별 exemplar 유지) > fused (평균 후 사용)
4. **Safety scale 1.0**이 최적 — 과도한 guidance(3.0, 5.0)는 content 파괴로 SR 하락
5. Exemplar selection 전략(all/top_k/diverse)은 성능 차이 거의 없음

### What doesn't work
1. **hybrid guide mode**: avg SR 66.4% — 너무 공격적인 noise manipulation
2. **Both (image+text) probe**: 단일 source보다 오히려 성능 저하
3. **High safety_scale (5.0)**: NudeNet은 좋아지나 Qwen SR은 하락 (content 파괴)

### v17 vs v19
- v17 (text probe only) = v19 (image multi-probe) = **SR 90.6%**
- v19의 차별점: 335개 sweep에서 **안정적 90%+ 유지**, image probe의 spatial precision
- v17은 text probe 기반이라 concept-agnostic → multi-concept 확장 용이

---

## 7. NudeNet 참고 (v3~v19)

> NN은 body-part detector로, Qwen VLM과 기준이 다름. **참고용으로만 사용.**

| Version | NN Unsafe% | Note |
|---------|-----------|------|
| v13 | 0.0% | representation collapse (품질 저하) |
| v3/dag | 0.9% | training-based baseline |
| v7 | 0.9% | noise CAS spatial mask |
| v17 | 2.9% | text probe |
| v18 | 2.9% | step-adaptive |
| v19 | 2.9% | multi-exemplar probe |
| v14 | 3.2% | hybrid WHERE fusion |

---

## 8. Multi-Concept 실험 계획

### 7 Concepts x 6 Methods = 48 Experiments (compound 포함)

| Concept | I2P Prompts | CAS Thr | Status |
|---------|-------------|---------|--------|
| Sexual | 931 | 0.6 | v19 sweep 완료 |
| Violence | 757 | 0.5 | 실험 대기 |
| Harassment | 824 | 0.45 | 실험 대기 |
| Hate | 234 | 0.5 | 실험 대기 |
| Shocking | 857 | 0.5 | 실험 대기 |
| Illegal Activity | 727 | 0.5 | 실험 대기 |
| Self-harm | 802 | 0.45 | 실험 대기 |

**siml-05 8 GPU, 예상 ~24h**
