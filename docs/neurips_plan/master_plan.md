# Master Plan: NeurIPS 2026 Submission

## 전체 워크플로우 개략도

```
[Phase A: 방법론 확정]          [Phase B: 실험]              [Phase C: 평가]           [Phase D: 발표/논문]
 ┌─────────────────┐      ┌──────────────────┐      ┌──────────────────┐      ┌─────────────────┐
 │ v14-v19 구현     │ ──→  │ Ring-A-Bell 79p  │ ──→  │ NudeNet + Qwen   │ ──→  │ HTML 발표자료    │
 │ (CLIP 이미지     │      │ Grid Search      │      │ VQAScore         │      │ 결과표/시각화    │
 │  임베딩 변형)    │      │                  │      │ SigLIP2          │      │                  │
 └────────┬────────┘      │ MMA/P4DN/UDiff   │      │ I2P 7-category   │      │ NeurIPS 논문     │
          │               │ COCO FP          │      │ LPIPS            │      │ 드래프트         │
          │               │ I2P sexual       │      │ Human eval       │      └─────────────────┘
          │               └──────────────────┘      └──────────────────┘
          │
 ┌────────┴────────┐
 │ Baseline 재현    │
 │ SAFREE/SLD/SGF  │
 │ SD baseline     │
 └─────────────────┘
```

## 확정된 방법론

**미팅 03-27 확정**: Image + Text Exemplar 기반 Example-based When + Where + How

| 구성 | 메커니즘 | 비고 |
|------|---------|------|
| **WHEN** | CAS (Concept Alignment Score) | `cos(d_prompt, d_target) > tau` (tau=0.6) |
| **WHERE** | Image/Text Exemplar Cross-Attention Probe | CLIP 이미지 임베딩 + 텍스트 키워드 |
| **HOW** | **dag_adaptive** (Area+Magnitude Adaptive Guidance) | `eps_cfg - s * area_factor * mag_scale * M * d_target` (v3에서 NN 0.95%, SR 90.5%) |

**핵심 차별점**: Training-free, Example-based (image+text 둘 다 활용), Selective guidance (When+Where), Pixel-wise adaptive HOW (dag_adaptive)

## 실행 순서 (우선순위순)

### Step 1: 방법론 구현 (2-3일)

| 순서 | 버전 | 설명 | 예상 시간 | 우선순위 |
|------|------|------|----------|---------|
| 1 | **v14** | v6 CrossAttn WHERE + v7 noise CAS 확인 (Hybrid WHERE Fusion) | 1-2h | **HIGH** |
| 2 | **v18** | v14에 timestep-adaptive mask/guidance 추가 | 30min | **HIGH** |
| 3 | **v15** | CLIP Patch Token Probe (256 spatial tokens, CLS 대신) | 4h | **HIGH** |
| 4 | **v16** | Contrastive Image Direction (nude-clothed 차이 벡터) | 4h | **MED-HIGH** |
| 5 | **v17** | IP-Adapter Image Projection (pre-trained Resampler 활용) | 1일 | **MED** |
| 6 | **v19** | Multi-Exemplar Diverse Probe Ensemble (16개 개별 프로브) | 4h | **MED** |

**Kill Criterion**: Ring-A-Bell 전체 79 prompts에서 NN < 15% AND SR > 80%

### Step 2: Grid Search + 최적 Config 선정 (2-3일)

각 버전별 Ring-A-Bell에서 grid search:
- `safety_scale`: [0.5, 1.0, 1.5, 2.0, 3.0]
- `spatial_threshold`: [0.15, 0.2, 0.25, 0.3, 0.4]
- `sigmoid_alpha`: [5, 10, 15, 20]
- `guide_mode`: dag_adaptive (우선), hybrid, sld, proj_anchor

**프로세스**:
1. Quick smoke test: 20 prompts로 빠르게 탈락 판정
2. Full eval: 통과한 configs만 79 prompts 전체로
3. NudeNet + Qwen SR 평가
4. Best 3 configs 선정

### Step 3: Multi-Dataset 확장 (1-2일)

Best config으로 4개 데이터셋 전체 실험:
| Dataset | Prompts | 특성 |
|---------|---------|------|
| Ring-A-Bell | 79 | 주요 벤치마크 |
| MMA | ~100 | Adversarial (white-box), 텍스트에 nudity 명시 없음 |
| P4DN | ~80 | 다양한 nudity 프롬프트 |
| UnlearnDiff | ~50 | Fine-tuning 방법론 비교용 |

+ COCO 30 prompts (FP + FID)
+ I2P sexual subset (931 prompts)

### Step 4: Baseline 비교 (1-2일)

| Baseline | 소스 | 필요 작업 |
|----------|------|----------|
| SD v1.4 baseline | `generate_baseline.py` | Ring-A-Bell만 재실행 |
| SLD-Medium | 기존 v3 SLD 모드 | 이미 있을 수 있음 확인 |
| SAFREE | `SAFREE/` 폴더 | Ring-A-Bell + COCO 실행 |
| SGF | `SGF/` 폴더 | Ring-A-Bell 실행 |
| v3/dag_adaptive | 기존 `outputs/v3/` | **우리 방법 — HOW ablation 기준점 (dag_adaptive only)** |

### Step 5: 종합 평가 (2-3일)

모든 방법 x 모든 데이터셋 x 모든 메트릭:

**Primary Metrics** (논문 Table 1):
- NudeNet Unsafe% (threshold=0.8)
- Qwen SR% = (Safe + Partial) / Total
- VQAScore (prompt alignment)

**Secondary Metrics**:
- COCO FID + CLIP Score (benign preservation)
- SigLIP2 NSFW rate
- I2P 7-category IP (inappropriate probability)
- LPIPS (추가 구현 필요)

**New Evaluations** (논문 차별화):
- I2P concept-wise evaluation (Table 2 스타일)
- VQAScore on anchor-friendly subset
- Artist style removal (SAFREE Table 4 스타일, stretch goal)

### Step 6: 시각 자료 생성 (1일)

- GradCAM / Cross-attention map 비교 (text probe vs image probe vs hybrid)
- Image generation comparison grids (5 methods x 5 prompts)
- Pareto front chart (SR vs NN)
- Ablation tables
- Pipeline architecture diagram

### Step 7: 발표 자료 작성 (1일)

`presentation_next.html` (Reveal.js, `presentation_dag_pbe_v13.html` 기반):
- 18-22 slides
- KaTeX math rendering
- 실제 이미지 비교 포함 (`<img>` 태그)
- 영어 (논문 submission 대비)

### Step 8: 논문 드래프트 (병렬 진행)

- Related Work + Method 섹션: Step 1-2와 병렬
- Results + Tables: Step 5 완료 후
- Introduction + Abstract: 최종

## 리스크 및 대응

| 리스크 | 확률 | 대응 |
|--------|------|------|
| v14-v19 모두 v7보다 나쁨 | 중 | v4+v7 조합으로 fallback, Pareto 분석 논문 |
| v3/dag_adaptive + 새 WHERE 조합 미검증 | 중 | v14/v15 WHERE와 dag_adaptive HOW 조합 실험으로 해결 |
| SR과 NN 동시 달성 불가 | 중 | Pareto front 제시, 각 축에서 best 보여주기 |
| 평가 스크립트 미구현 | 저 | I2P/LPIPS 스크립트 먼저 작성 |
| GPU 부족/충돌 | 저 | nvidia-smi 확인 후 실행, nohup 활용 |

## CVPR Workshop 리뷰어 피드백 대응

| 지적 | 대응 | 상태 |
|------|------|------|
| VLM 평가 신뢰성 | Human eval + VLM-human agreement (Cohen's kappa) | 미시작 |
| Generalization | I2P 7-category + multi-concept (violence, hate) | 계획 |
| Overhead 분석 | Wall-clock time + memory usage 측정 | 미시작 |

## 주간 미팅 보고 체크리스트

- [ ] Best config 성능 (NN%, SR%, VQA)
- [ ] Baseline 대비 비교표
- [ ] Cross-attention / GradCAM 시각화
- [ ] 새 평가 결과 (I2P, VQAScore, etc.)
- [ ] 다음 주 계획
