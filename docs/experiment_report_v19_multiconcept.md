# Experiment Report: v19 Grid Search Results & Multi-Concept Plan

**Date**: 2026-04-05  
**Author**: Auto-generated  
**Dataset**: Ring-A-Bell (79 prompts, 4 samples/prompt = 316 images)  
**Model**: CompVis/stable-diffusion-v1-4, safety_checker=None  
**Eval**: NudeNet (body-part detection) + Qwen VLM (semantic 4-class: NotRel/Safe/Partial/Full)

---

## 1. Version Evolution Summary

### Best Config per Version (Ring-A-Bell)

| Version | NN Unsafe% | Qwen Full% | Qwen SR% | Best Config | #Configs Tested |
|---------|-----------|------------|----------|-------------|-----------------|
| **v3** (SLD baseline) | **0.9%** | - | - | dag_s5 | 16 |
| v4 | 3.3% | - | - | COCO_sld_s5 | 74 |
| v5 | 5.0% | - | - | COCO_ainp_cas05 | 64 |
| v6 (CrossAttn WHERE) | 6.7% | - | - | COCO_v6_crossattn | 11 |
| **v7** (Noise CAS WHERE) | **0.9%** | - | - | v7_hyb_ts15_as15_cas05 | 38 |
| v8 (Projection) | 5.0% | - | - | COCO_v8_dual_ts15_as15 | 12 |
| v9 (Direct Exemplar) | 5.0% | - | - | COCO_v9_exhyb_ts15_as15 | 13 |
| v10 (Projection v2) | 6.7% | - | - | v10_proj_ts2_as1 | 3 |
| v12 (CrossAttn WHERE v2) | 9.5% | - | - | v12_xattn_proj_ts2_as1 | 1 |
| v13 (CLIP Exemplar Probe) | **0.0%** | - | - | ringabell79_fn_hybproj_ss20_st02_a15 | 169 |
| v14 (Hybrid WHERE Fusion) | 3.2% | - | - | ringabell_image_dag_adaptive_ss5.0_st0.2 | 72 |
| v15 (CLIP Patch Token) | 3.8% | - | - | ringabell_text_dag_adaptive_ss5.0_st0.2_np16 | 24 |
| **v17** (Text Probe Only) | 2.9% | **9.4%** | **90.6%** | text_dag_adaptive_ss2.0_st0.2_probe_only | 122 |
| v18 (Step-adaptive) | 2.9% | 11.8% | 88.2% | image_dag_adaptive_ss3.0_st0.2_cosine_sb0.5 | 134 |
| **v19** (Multi-Exemplar Probe) | 2.9% | **9.4%** | **90.6%** | image_diverse_dag_adaptive_ss1.0_st0.3_multi_probe | **329** |

### Key Takeaways
- **v3/dag (SLD baseline)**: NN 0.9% — training-based 방법이라 직접 비교 주의
- **v13**: NN 0.0%이나 representation collapse로 이미지 품질 저하 (Qwen eval 미실시)
- **v17 = v19 최고 성능 동률**: Qwen Full% 9.4%, SR 90.6%
- **v19의 장점**: 329개 config sweep 완료, 안정적으로 9-10% Full 유지

---

## 2. v19 Detailed Results (329 configs evaluated)

### Guide Mode 비교

| Guide Mode | #Configs | Avg Full% | Avg SR% | Best Full% |
|------------|----------|-----------|---------|------------|
| **dag_adaptive** | 167 | **14.1%** | **85.9%** | **9.4%** |
| hybrid | 162 | 33.6% | 66.4% | 18.2% |

**결론: dag_adaptive가 hybrid 대비 압도적으로 우수** (avg Full% 14.1 vs 33.6)

### Probe Source 비교

| Probe Source | #Configs | Avg Full% | Best Full% |
|-------------|----------|-----------|------------|
| image | 143 | 23.5% | **9.4%** |
| both | 138 | 23.5% | 10.2% |
| text | 48 | 24.9% | 9.9% |

**결론: image probe가 근소하게 best, 평균은 image=both > text**

### Where Mode 비교

| Where Mode | #Configs | Avg Full% | Best Full% |
|-----------|----------|-----------|------------|
| fused | 164 | 23.5% | 10.4% |
| multi_probe | 165 | 23.9% | **9.4%** |

**결론: multi_probe가 best에서 약간 우세, 평균은 비슷**

### Top 10 v19 Configs

| Rank | Config | Full% | SR% |
|------|--------|-------|-----|
| 1 | image_diverse_dag_adaptive_ss1.0_st0.3_multi_probe | **9.4%** | 90.6% |
| 2 | image_all_dag_adaptive_ss1.0_st0.2_multi_probe | **9.4%** | 90.6% |
| 3 | image_all_dag_adaptive_ss1.0_st0.4_multi_probe | **9.4%** | 90.6% |
| 4 | image_top_k_dag_adaptive_ss1.0_st0.2_multi_probe | 9.5% | 90.5% |
| 5 | image_top_k_dag_adaptive_ss1.0_st0.4_multi_probe | 9.6% | 90.4% |
| 6 | image_diverse_dag_adaptive_ss1.0_st0.4_multi_probe | 9.9% | 90.1% |
| 7 | text_dag_adaptive_ss3.0_st0.3_multi_probe | 9.9% | 90.1% |
| 8 | image_top_k_dag_adaptive_ss1.0_st0.3_multi_probe | 9.9% | 90.1% |
| 9 | image_diverse_dag_adaptive_ss1.0_st0.2_multi_probe | 10.0% | 90.0% |
| 10 | image_all_dag_adaptive_ss1.0_st0.3_multi_probe | 10.2% | 89.8% |

### Top Config 패턴 분석
- **Guide mode**: Top 10 전부 `dag_adaptive`
- **Safety scale**: Top 9이 `ss1.0` (가장 약한 guidance) — 과도한 guidance가 오히려 해로움
- **Probe source**: Top 6이 `image` — CLIP image exemplar가 text keyword보다 효과적
- **Where mode**: Top 10 전부 `multi_probe`
- **Exemplar selection**: `diverse`, `all`, `top_k` 모두 비슷한 성능

---

## 3. v19 Grid Search 현황

| 항목 | 수량 |
|------|------|
| 총 config 수 | 336 |
| 생성 완료 (stats.json) | 333 |
| NudeNet 평가 완료 | 331 |
| Qwen 평가 완료 | 329 |
| 미완료 | 3 (현재 siml-05에서 진행중) |

**Sweep 변수:**
- `probe_source`: text, image, both (3)
- `exemplar_selection`: all, top_k, diverse (3)
- `guide_mode`: dag_adaptive, hybrid (2)
- `safety_scale`: 1.0, 2.0, 3.0, 5.0 (4)
- `spatial_threshold`: 0.2, 0.3, 0.4 (3)
- `where_mode`: multi_probe, fused (2)

---

## 4. Multi-Concept Expansion Plan

### 목표
동일 방법론(v14, v19)을 nudity 이외 6개 I2P concept으로 확장 실험.

### Concepts & Datasets

| Concept | I2P Prompts | CAS Threshold | Status |
|---------|-------------|---------------|--------|
| Sexual (nudity) | 931 | 0.6 | v19 grid search 완료 |
| Violence | 757 | 0.5 | 준비 완료, 실험 대기 |
| Harassment | 824 | 0.45 | 준비 완료, 실험 대기 |
| Hate | 234 | 0.5 | 준비 완료, 실험 대기 |
| Shocking | 857 | 0.5 | 준비 완료, 실험 대기 |
| Illegal Activity | 727 | 0.5 | 준비 완료, 실험 대기 |
| Self-harm | 802 | 0.45 | 준비 완료, 실험 대기 |

### Methods to Compare

| Method | Type | Description |
|--------|------|-------------|
| SD 1.4 Baseline | Baseline | Vanilla generation, no safety |
| SAFREE | Baseline | Training-free safety (self-validation + re-attention) |
| v14 (dag_adaptive) | Ours | Hybrid WHERE Fusion (crossattn × noise CAS) |
| v14 (hybrid) | Ours | Hybrid WHERE Fusion, hybrid guide mode |
| v19 (dag_adaptive) | Ours | Multi-Exemplar Diverse Probe Ensemble |
| v19 (hybrid) | Ours | Multi-Exemplar, hybrid guide mode |

### Compound Concept Experiments

| Combination | Concepts | Prompts | Purpose |
|------------|----------|---------|---------|
| 2-concept | sexual + violence | 931 | Pairwise interference test |
| 3-concept | sexual + violence + harassment | 931 | Multi-concept scaling |
| All 7 | All concepts | 931 | Full multi-concept framework |

### Execution Plan

| Phase | Task | Est. Time | GPU |
|-------|------|-----------|-----|
| 1 | Single-concept generation (42 experiments) | ~24h | siml-05 × 8 GPU |
| 2 | Compound concept generation (6 experiments) | ~24h (parallel with Phase 1) | siml-05 × 8 GPU |
| 3 | Qwen VLM evaluation (all outputs) | ~4-6h | siml-05 × 8 GPU |
| 4 | Result aggregation & analysis | <1h | Local |

**총 48 experiments, 예상 벽시계 ~24h (8 GPU parallel)**

### Evaluation
- **Primary metric**: Qwen VLM SR% = (Safe + Partial) / Total
- **Secondary metric**: Qwen Full% (unsafe rate, lower = better)
- NudeNet은 nudity 전용이므로, 다른 concept은 VLM 평가가 주력

---

## 5. Next Steps

1. [x] v19 Grid Search 완료 (336 configs, Ring-A-Bell)
2. [ ] siml-05 v19 잔여 3개 완료 대기
3. [ ] Multi-concept 실험 launch (48 experiments, ~24h)
4. [ ] Qwen VLM 평가 실행
5. [ ] 결과 집계 및 cross-concept 분석
6. [ ] NeurIPS 논문 Table 구성
7. [ ] 발표 자료 업데이트
