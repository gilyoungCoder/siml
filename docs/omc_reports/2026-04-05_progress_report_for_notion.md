# 2026-04-05 Progress Report — v20/v21 실험 결과 및 Multi-Concept 확장

## Executive Summary
- v3~v19 전체 Qwen3-VL 재평가 완료 → **v4 anchor_inpaint (SR 96.5%)가 최고**
- **v20**: CLIP image exemplar probe로 WHERE mask 보강 → **v4 baseline 대비 +3.8% SR 개선** (1-sample)
- **v21**: Adaptive anchor inpainting (area dampening, temporal decay, mask gamma) → **area dampening은 역효과, v4 baseline이 여전히 최강**
- Best config 확정 후 **전체 nudity 5개 데이터셋 + 6개 concept I2P 실험 시작**

---

## 1. Qwen3-VL 재평가 (v3~v19)

기존 Qwen2-VL 결과와 Qwen3-VL 결과가 크게 다름 (Qwen3-VL이 더 엄격):

| Version | Best Config | Qwen3-VL SR% | Qwen3-VL NR% | Qwen3-VL Full% |
|---------|------------|-------------|-------------|---------------|
| **v4** | ainp_s1.0_t0.1 | **96.5%** | 3.5% | 0.0% |
| **v3** | dag_s3 | 92.1% | 7.9% | 0.0% |
| v18 | image_dag_ss3.0_st0.2 | 86.7% | 4.4% | 8.9% |
| v17 | text_dag_ss2.0_st0.2 | 74.1% | 10.1% | 15.8% |
| v19 | image_diverse_dag_ss1.0 | 73.4% | 3.5% | 23.1% |
| v14 | fused_dag_ss3.0 | 72.5% | - | - |

**핵심 발견**: 
- Noise spatial CAS (WHERE) + Anchor inpainting (HOW)가 cross-attention probe 기반 방식들보다 압도적
- DAG adaptive < anchor inpainting (HOW 비교)
- Cross-attention probe를 WHERE로 쓰면 오히려 성능 하락

---

## 2. v20: CLIP Image Exemplar WHERE Enhancement

**설계**: v4 base + CLIP image exemplar cross-attention probe로 WHERE mask 보강

**Image embedding strategies**:
- `cls_mean`: K개 exemplar CLS features 평균 → 1 probe token
- `cls_multi`: 각 exemplar CLS를 개별 token → K probe tokens (union)

**Fusion modes**:
- `noise_only`: 순수 v4 (probe 없음)
- `multiply`: noise × attn (intersection)
- `noise_boost`: noise × (1 + α·attn)

**HOW modes**: anchor_inpaint, dag_adaptive, hybrid

### v20 Results (Ring-A-Bell, 1 sample, 79 images)

| Config | SR% | NR% | Full% |
|--------|-----|-----|-------|
| **cls_multi_boost05_ainp** | **88.6%** | 1.3% | 10.1% |
| cls_mean_boost_ainp | 87.3% | 2.5% | 10.1% |
| cls_multi_boost10_ainp | 87.3% | 1.3% | 11.4% |
| cls_multi_boost_ainp | 87.3% | 2.5% | 10.1% |
| cls_mean_boost10_ainp | 86.1% | 2.5% | 11.4% |
| **v4_baseline** | **84.8%** | 2.5% | 12.7% |
| cls_mean_mul_ainp | 83.5% | 1.3% | 15.2% |
| cls_multi_mul_ainp | 81.0% | 1.3% | 17.7% |
| hybrid 계열 | 40-59% | - | 50%+ |
| dag 계열 | 15-16% | 81%+ | - |

**핵심 발견**:
1. **CLIP image probe + noise_boost가 v4를 +3.8% 개선!** (84.8% → 88.6%)
2. `cls_multi` (개별 exemplar) > `cls_mean` (평균) — 다양성 보존이 중요
3. `boost_alpha=0.5`가 최적 (gentle boost)
4. `multiply` (intersection)은 너무 제한적
5. **hybrid/dag는 anchor_inpaint보다 크게 열등** — ainp 확정

---

## 3. v21: Adaptive Anchor Inpainting

**목표**: v4의 NR(over-erasure) 3.5% → <2% 줄이면서 Full=0% 유지

**v21 innovations**:
1. Area dampening: 큰 마스크 → safety_scale 줄여서 사람 소실 방지
2. Temporal decay: 후반 step에서 가벼운 guidance → 디테일 보존
3. Mask gamma: mask^gamma로 고확신 영역에 집중
4. Min preserve: 원본 최소 보존 비율

### v21 Results (Ring-A-Bell, 4 samples, 316 images)

| Config | SR% | NR% | Full% |
|--------|-----|-----|-------|
| **v4_ref_ss10_st01** | **93.7%** | 3.8% | 2.5% |
| gamma_15 | 92.7% | 4.4% | 2.8% |
| gamma_20 | 92.4% | 4.4% | 3.2% |
| preserve_015 | 91.5% | 2.8% | 5.7% |
| temporal_f05 | 90.5% | 3.5% | 6.0% |
| temporal_f03 | 88.9% | **2.2%** | 8.9% |
| area_damp_d05 | 64.9% | 1.9% | 33.2% |
| area_damp_d07 | 50.0% | 1.6% | 48.4% |
| area_damp_d09 | 43.4% | 1.9% | 54.7% |

**핵심 발견**:
1. **Area dampening은 역효과** — NR 줄지만 Full 폭증 (under-erasure)
2. **v4 baseline이 여전히 최강** — SR/Full 트레이드오프에서 최적점
3. Gamma mask (1.5)은 거의 동등한 성능
4. Temporal decay는 NR 약간 감소하나 Full 증가 트레이드오프

---

## 4. Best Config 확정 및 Full Evaluation

**확정 config**: v20 `cls_multi_boost05_ainp`
- v4 anchor_inpaint base + CLIP multi-exemplar noise_boost (alpha=0.5)
- 1-sample 기준 SR 88.6% → 4-sample에서 더 높을 것으로 예상

### 실행 중인 Full Evaluation (siml-02, 8 GPUs)

**Phase 2 — Nudity (best config × 5 datasets)**:
| Dataset | Prompts | Status |
|---------|---------|--------|
| Ring-A-Bell | 78 | 실행 중 |
| MMA | 1,000 | 실행 중 |
| P4DN | 151 | 실행 중 |
| UnlearnDiff | 142 | 실행 중 |
| I2P Sexual | 931 | 실행 중 |

**Phase 3 — Multi-Concept (v4-style × 6 concepts)**:
| Concept | I2P Prompts | Status |
|---------|-------------|--------|
| Violence | 756 | 실행 중 |
| Harassment | 824 | 실행 중 |
| Hate | 231 | 실행 중 |
| Shocking | 856 | 실행 중 |
| Illegal Activity | 727 | 실행 중 |
| Self-harm | 801 | 실행 중 |

예상 완료: ~2.5시간 → Qwen3-VL 평가 → 결과 집계

---

## 5. 생성된 파일

| File | Description |
|------|------------|
| `CAS_SpatialCFG/generate_v20.py` | v4 + CLIP image probe WHERE |
| `CAS_SpatialCFG/generate_v21.py` | Adaptive anchor inpainting |
| `scripts/nohup_v20_siml02.sh` | v20 실험 runner |
| `scripts/nohup_v21_siml02.sh` | v21 실험 runner |
| `scripts/run_full_eval.sh` | Phase 2/3 통합 runner |
| `scripts/collect_v20_results.py` | v20 결과 수집 |

---

## 6. Next Steps

1. Full evaluation 완료 대기 (~2.5h)
2. Qwen3-VL 평가 실행 (`bash scripts/run_full_eval.sh --eval`)
3. 결과 집계 (`bash scripts/run_full_eval.sh --results`)
4. COCO benign FP 체크 (false positive)
5. NeurIPS 논문용 표 생성
6. Concept별 CAS threshold 최적화 (violence, harassment 등)
