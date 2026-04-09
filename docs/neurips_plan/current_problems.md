# Current Problems & Solutions

## 핵심 문제 요약

```
                    SR (Content Preservation)
                    ↑
            96.5% · · · · · v4 (ainp)
            91.4% · · · · · v6 (crossattn) · · · v3/dag (90.5%)
                  ·
                  · · · · · · · · · · · · · · · GAP
                  ·
            54.1% · · · · · v7 (hybrid)
            41.7% · · · · · v10
                  ·
                  +----+----+----+----+----+----→ NudeNet% (Safety, lower=better)
                  0    5    10   15   20   25

문제: SR과 NN을 동시에 달성하는 방법이 없음
- v7: 최고 NN (5.06%) but SR 파괴 (54.1%)
- v6: 최고 SR (91.4%) but nudity leak (15.5%)
- v3/dag: 양쪽 다 최강 (SR 90.5%, NN 0.95%) — 우리 방법의 dag_adaptive HOW 모드
```

---

## Problem 1: v7의 Content Destruction (SR 54.1%)

### 근본 원인
**Noise-based spatial CAS의 mask over-coverage**

| Version | WHERE Method | Avg Mask Area | SR% | NN% |
|---------|-------------|---------------|-----|-----|
| v6 | Cross-attention probe (text) | **0.31** | 91.4 | 15.5 |
| v7 | Noise spatial CAS | **0.88** | 54.1 | 5.06 |

- v7의 noise-space cosine similarity는 nudity prompt에 대해 **globally correlated**
- 즉, 거의 모든 pixel에서 `cos(d_prompt[h,w], d_target[h,w]) > threshold`
- Mask가 이미지의 88%를 커버 → hybrid guidance가 전체 이미지를 왜곡
- 결과: NotRelevant 40-75% (사람이 아예 사라지거나 무관한 이미지 생성)

### 해결 방향
1. **v14 (Hybrid WHERE Fusion)**: v6의 focused mask × v7의 CAS 확인 = 교차점만 (~0.15-0.25)
2. **v18 (Timestep Adaptive)**: 초반에 강하게, 후반에 약하게 → layout 단계에서 nudity 방지

---

## Problem 2: v6의 Nudity Leakage (NN 15.5%)

### 근본 원인
**Text-based probe의 한계: keyword matching 실패**

현재 `target_words = ["nude", "naked", "nudity", "nsfw", "bare", "body"]`

- Ring-A-Bell의 간접적 nudity 표현에서 token match 실패
  - "sultry temptress reclining on silk" — "nude" 단어 없음
  - "woman emerging from bath" — 직접적 nudity keyword 부재
- Cross-attention probe가 matching token을 못 찾으면 → mask = 0 → no guidance → nudity 통과

### 해결 방향
1. **target_words 확장**: `["skin", "breast", "topless", "underwear", "lingerie", "exposed", "undress", "erotic", "sensual"]` 추가
   - 위험: COCO FP 증가 ("bare tree", "body of water" 등)
2. **v15 (CLIP Patch Token)**: 텍스트 keyword 의존 제거 → 이미지 패턴 직접 매칭
3. **v16 (Contrastive Direction)**: nude-clothed CLIP 차이 벡터로 concept 특화

---

## Problem 3: v13의 Representation Collapse

### 근본 원인
**CLIP CLS token 반복 → cross-attention 선택성 상실**

```
prepare_clip_exemplar.py 과정:
  16 exemplar images → CLIP get_image_features() → 16 × [768]
  → Average: 1 × [768]  ← 모든 이미지 정보 1개 벡터로 압축
  → 4번 반복: [BOS, concept, concept, concept, concept, EOS, PAD...]
  → K_target = to_k(repeated_vector)
  → 4개 key 열이 거의 동일 → softmax attention ≈ uniform
  → Probe mask = 공간 선택성 없음
```

v13 best NN = 8.86% — v7 (5.06%) 보다 나쁨, v6 (15.5%) 보다 약간 나음

### 해결 방향
1. **v15 (CLIP Patch Tokens)**: CLS 대신 256개 patch token 사용 → 공간 정보 보존
2. **v17 (IP-Adapter)**: Pre-trained Resampler로 최적 projection
3. **v19 (Multi-Exemplar)**: 평균하지 않고 16개 개별 probe → union

---

## Problem 4: v3/dag_adaptive의 성능을 v4+ 파이프라인에 통합하지 못한 문제

### 현실
| Config | NN% | SR% | VQA | 방법 |
|--------|-----|-----|-----|------|
| v3/dag_s5 | **0.95** | **90.5** | 0.324 | **Ours: CAS + dag_adaptive HOW** |
| v7_hyb_ts15_as15 | 5.06 | 54.1 | 0.408 | Ours: noise spatial CAS + hybrid HOW |
| v6_ts20_as15 | 15.5 | 91.4 | 0.469 | Ours: crossattn probe + hybrid HOW |

### v3/dag_adaptive는 우리 독자적 방법
> **확인 완료**: dag_adaptive는 DAG 논문(Li et al., 2025)과 **별개의 독자적 구현**이다.
> DAG 논문의 area-adaptive scaling "직관"에서 영감을 받았으나, 핵심이 완전히 다름:
> - DAG: annotated 데이터 기반 **Guideline Token Optimization** + connected component analysis (학습 필요)
> - 우리: **CAS text gating** + cross-attention map + area/magnitude adaptive scaling (완전 training-free)
> - DAG와 공유하는 것: base weight `5/(H*W)`, magnitude range [1,5] 상수뿐
> - 탐지(WHEN), 위치(WHERE), 전체 파이프라인 구조가 모두 독자 설계
> → 상세 비교: `related_work/comparison_dag_vs_ours.md` 참고

### dag_adaptive HOW가 왜 최강인가
1. **Area-adaptive scaling**: mask 면적에 비례하여 guidance 강도 조절 → 작은 영역이면 강하게, 넓으면 약하게
2. **Magnitude-based weighting**: spatial CAS score 크기에 따라 가중치 → 확실한 곳만 강하게 개입
3. **CAS gating (tau=0.3)**: 낮은 threshold로 대부분의 nudity prompt 잡음
4. 이 조합이 v4-v7의 fixed-scale guidance보다 자연스러운 이유: **pixel-wise adaptive** scaling

### 왜 v4 이후로 이 방식을 활용하지 못했나?
- v4부터 CAS threshold를 0.6으로 올림 (COCO FP 감소 목적)
- v4+의 WHERE가 더 복잡 (spatial CAS, crossattn probe)해지면서 dag_adaptive와 조합 미검토
- **Action item**: v14/v15에 dag_adaptive HOW mode를 반드시 조합하여 테스트!
  - v6 WHERE (focused crossattn mask) + dag_adaptive HOW → SR 유지 + NN 개선 기대
  - v14 WHERE (hybrid fusion) + dag_adaptive HOW → 최적 조합 탐색

### 논문에서의 위치
- **v3/dag_adaptive를 우리 방법론의 핵심 HOW 모드로 포함** (baseline이 아님)
- 논문 Table 1에서 HOW ablation: sld / hybrid / dag_adaptive 비교
- dag_adaptive가 최강 HOW임을 보여주고, WHERE 개선(v6/v14/v15)과 조합하여 전체 성능 향상
- DAG 논문은 Related Work에서 "training-based method"로 비교 대상에 포함

---

## Problem 5: 평가 파이프라인 미비

### 현재 상태 vs 필요 상태

| 평가 항목 | 현재 구현 | NeurIPS 필요 | GAP |
|----------|----------|-------------|-----|
| NudeNet | ✅ 구현됨 | ✅ | - |
| Qwen3-VL SR | ✅ 구현됨 | ✅ | - |
| VQAScore | ✅ 구현됨 | ✅ | - |
| FID/CLIP Score | ✅ 구현됨 | ✅ | COCO FID 3.2% 커버리지 |
| SigLIP2 Safety | ✅ 구현됨 | ✅ | - |
| I2P 7-category IP | ❌ 미구현 | ✅ | **eval_ip_metric.py 필요** |
| LPIPS | ❌ 미구현 | ✅ | **eval_lpips.py 필요** |
| Artist Style Removal | ❌ 미구현 | 🔶 선택 | 파이프라인 전체 필요 |
| Human Evaluation | ❌ 미구현 | ✅ | 프로토콜 설계 필요 |
| Baseline 비교 | ❌ 통합 안됨 | ✅ | **SAFREE/SLD/SGF 동일 벤치마크** |
| Multi-dataset | 🔶 일부 | ✅ | MMA/P4DN/UnlearnDiff 완전 커버 |

### 해결 순서
1. **P0**: Baseline 재현 (SAFREE, SLD on Ring-A-Bell + COCO)
2. **P0**: Best config → 4 datasets 전체 실험
3. **P1**: I2P 7-category 생성 + IP metric 구현
4. **P1**: LPIPS 구현 (erased/unrelated)
5. **P2**: Human evaluation 프로토콜 설계 + 실행
6. **P2**: Artist style removal (stretch goal)

---

## Problem 6: CVPR Workshop 리뷰어 피드백

### 지적 1: VLM 평가 신뢰성
> "human annotation과의 alignment 검증이 없다"

**대응**:
- 50-100개 이미지에 대해 3명 annotator human evaluation
- Full/Partial/Safe/NotRelevant 4-category (Qwen과 동일 taxonomy)
- Cohen's kappa ≥ 0.7 목표
- MJ-Bench의 Evident/Subtle/Evasive 분류로 정당화 (reference)
- SigLIP2 cross-validation으로 보강

### 지적 2: Generalization
> "현재 nudity에만 한정, 다른 concept 확장 불분명"

**대응**:
- I2P 7-category evaluation (Table 2):
  - Harassment, Hate, Illegal Activity, Self-harm, Sexual, Shocking, Violence
  - IP (Inappropriate Probability) metric + CLIP Score
- 최소 2개 concept (violence + hate) 에서 pilot 실험
- Artist style removal (SAFREE Table 4 형식)

### 지적 3: Overhead 분석
> "classifier 사용 및 추가 연산에 따른 계산 비용 분석 부족"

**대응**:
| Component | Extra Cost | Note |
|-----------|-----------|------|
| CAS computation | ~1ms/step | cosine sim of existing UNet outputs |
| Cross-attention probe | ~0.5ms/step | single matmul during existing forward pass |
| CLIP exemplar prep | ~2sec (1회) | offline, amortized |
| Exemplar anchor | 0 | pre-computed, no UNet call |
| **총 overhead** | **< 3% wall-clock** | vs standard SD generation |

---

## Problem 7: Image Embedding Injection 방식 미확정

### v13의 접근 (pooling vector)
```
Image → CLIP ViT-L/14 → CLS token [768] → repeat → cross-attention key
한계: spatial 정보 완전 손실, token diversity 없음
```

### 후보 대안 (v15-v19)

| 방식 | 장점 | 단점 | 복잡도 |
|------|------|------|--------|
| CLIP Patch Tokens (v15) | Spatial 보존, 256 diverse tokens | Vision/text space alignment 불확실 | 중 |
| Contrastive Direction (v16) | Concept-specific, 원리적으로 깔끔 | 차이 벡터 magnitude 작을 수 있음 | 중 |
| IP-Adapter Resampler (v17) | 학습된 최적 projection | "Training-free" 주장과 긴장 | 높 |
| Timestep Adaptive (v18) | 어떤 방식과도 결합 가능 | 단독으로는 해결 안됨 | 낮 |
| Multi-Exemplar (v19) | 다양한 패턴 개별 탐지 | Union mask 너무 넓을 수 있음 | 중 |

### 확정된 방향 (미팅 03-27)
"Image + Text 둘 다 사용. Clear concept은 text, ambiguous concept은 image."
→ **v15 (CLIP patch) + v14 (hybrid WHERE)가 가장 부합**

---

## Problem 8: 발표 자료에 실제 이미지 없음

### 현재
- 6개 presentation HTML 모두 `<img>` 태그 **0개**
- 모든 결과가 텍스트/테이블로만 표현
- 시각적 증거 (생성 이미지, attention map, GradCAM) 부재

### meeting_pack에는 있음
- `meeting_pack/outputs/` 폴더에 실제 이미지 파일 존재:
  - `compare_*.png` — 방법간 생성 이미지 비교
  - `gradcam_v4_v13_*.png` — GradCAM 비교
  - `mask_compare_*.png` — Mask overlay 비교
  - `mask_v4_overlay_*.png`, `mask_v13_overlay_*.png`

### 해결
- 다음 presentation에는 `<img>` 태그로 실제 이미지 포함
- 상대 경로 사용: `<img src="CAS_SpatialCFG/meeting_pack/outputs/compare_0000.png">`
- 또는 base64 inline 임베딩 (meeting_20260320.html이 이 방식 사용, 파일 매우 큼)
- **권장**: 별도 `outputs/presentation_assets/` 폴더에 축소된 이미지 저장, 상대 경로 참조
