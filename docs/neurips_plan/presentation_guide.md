# Presentation Guide: 다음 발표 자료 가이드

## 템플릿 기본 정보

- **Framework**: Reveal.js v4.6.1 (CDN)
- **Base template**: `presentation_dag_pbe_v13.html` (가장 최신, KaTeX 포함)
- **Math**: KaTeX v0.16.9 (auto-render)
- **크기**: width=1200, height=700, margin=0.06
- **언어**: English (논문 submission 대비)
- **슬라이드 수**: 18-22개 목표

## CSS Design System

```css
:root {
  --heading: #1a1a2e;   /* 제목 텍스트 */
  --accent: #2563eb;    /* 파란색 강조 */
  --accent2: #7c3aed;   /* 보라색 강조 */
  --safe: #059669;      /* 녹색 (안전) */
  --danger: #dc2626;    /* 빨간색 (위험) */
  --warn: #d97706;      /* 주황색 (경고) */
  --teal: #0f766e;      /* 청록색 */
  --new: #ea580c;       /* 새로운 내용 강조 */
}
```

### 주요 CSS 클래스

| 클래스 | 용도 | 예시 |
|--------|------|------|
| `.title-slide` | 타이틀 슬라이드 (중앙 정렬, gradient h1) | 첫 슬라이드 |
| `.divider` | 섹션 구분 (어두운 gradient 배경) | Part 구분 |
| `.box` / `.box-green` / `.box-red` / `.box-warn` / `.box-purple` / `.box-teal` / `.box-new` | 정보 박스 (왼쪽 border 색상) | 인사이트, 결과 |
| `.two-col` / `.three-col` | CSS Grid 레이아웃 | 비교 |
| `.tag-blue` / `.tag-green` / `.tag-red` / `.tag-purple` / `.tag-orange` | 둥근 pill 뱃지 | 태그 |
| `.formula` / `.formula-hl` / `.formula-new` | 수식 박스 (KaTeX) | WHEN/WHERE/HOW |
| `.pipe .s` + `.pipe .a` | 파이프라인 흐름도 | WHEN→WHERE→HOW |
| `.nc` | 번호 원형 뱃지 | 순서 표시 |
| `.best` / `.hl` / `.new-row` / `.gray` | 테이블 행 하이라이트 | 결과 테이블 |
| `.mode-card` / `.mc-g` / `.mc-r` / `.mc-p` | 방법 카드 (모드별 색상) | Guidance 모드 |
| `.img-r` | 이미지 (rounded, shadow) | 생성 이미지 |
| `.small` | 작은 회색 텍스트 | 캡션 |

## 다음 발표 슬라이드 구조

### Part 0: Title (1 slide)

```html
<section class="title-slide">
  <h1>Example-based Selective Guidance</h1>
  <p>Training-Free Safe Generation via<br>Image + Text Exemplar When + Where + How</p>
  <div>
    <span class="tag tag-orange">NEW: Image Exemplar</span>
    <span class="tag tag-blue">Training-free</span>
    <span class="tag tag-green">Multi-concept</span>
  </div>
  <div class="small">Weekly Meeting · April XX, 2026</div>
</section>
```

### Part 1: Recap + Problem Statement (3 slides)

**Slide 1.1**: v4 When+Where+How 1-slide 요약 (pipe diagram + formula)
**Slide 1.2**: 현재 문제점 — SR vs NN 트레이드오프 scatter plot
  - Ours dag_adaptive HOW (SR 90.5%, NN 0.95%) vs Ours hybrid HOW (SR 54.1%, NN 5.06%)
  - "Why v7 fails: mask over-coverage (0.88 avg area)"
**Slide 1.3**: 접근 방향 — Image exemplar로 WHERE 개선

### Part 2: New Method (5-6 slides)

**Slide 2.1**: Architecture Overview
```
WHEN: Global CAS → 이 step 개입 여부
WHERE: CLIP Exemplar + Text CrossAttn Probe → spatial mask
HOW: Hybrid Guidance (online target + exemplar anchor)
```
- pipe diagram으로 WHEN→WHERE→HOW 표현
- 각 단계에 Image/Text 어디서 어떻게 사용되는지 표시

**Slide 2.2**: WHERE Detail — CLIP Patch Token Probe (v15)
```
formula box:
  K_target = to_k(CLIP_patches(exemplar_images))  # 256 spatial tokens
  A_probe = softmax(Q · K_target^T / √d)          # per-pixel nudity attention
```
- v13 (CLS repeat) vs v15 (256 patch tokens) 비교 다이어그램
- 왜 spatial 정보 보존이 중요한지

**Slide 2.3**: WHERE Detail — Hybrid Fusion (v14)
```
mask_attn = cross_attention_probe()   # focused, ~0.31
mask_cas = noise_spatial_cas()        # broad, ~0.88
mask = mask_attn × mask_cas           # intersection, ~0.20
```
- mask 시각화: v6 mask vs v7 mask vs hybrid mask

**Slide 2.4**: HOW Detail — Exemplar Anchor Direction
- Pre-computed concept directions (16 exemplar pairs)
- 3 UNet calls (25% speedup)
- proj_anchor formula with clamped positive projection

**Slide 2.5**: Timestep Adaptive (v18)
- Early steps: strong broad guidance (prevent nudity layout)
- Late steps: gentle focused guidance (preserve details)

**Slide 2.6** (optional): Image + Text 조합 방식
- Text probe for clear concepts (explicit keywords)
- Image probe for ambiguous concepts (indirect nudity)
- Union/intersection 선택 기준

### Part 3: Experimental Results (5-7 slides)

**Slide 3.1**: Ring-A-Bell Main Results Table
- `.best` for our best, `.hl` for baselines
- Columns: Method, Type (TF/Opt), NN%, SR%, Full%, NotRel%, VQA

**Slide 3.2**: Multi-Dataset Results
- Ring-A-Bell + MMA + P4DN + UnlearnDiff 종합 테이블
- MMA 강조: "Image exemplar는 adversarial prompts에도 작동"

**Slide 3.3**: ⭐ Image Comparison Grid (실제 이미지!)
```html
<div class="three-col">
  <div><img src="assets/baseline_00.png" class="img-r"><br><span class="small">Baseline</span></div>
  <div><img src="assets/safree_00.png" class="img-r"><br><span class="small">SAFREE</span></div>
  <div><img src="assets/ours_00.png" class="img-r"><br><span class="small">Ours</span></div>
</div>
```
- 4-5 prompts × 3-5 methods comparison grid
- **이번에는 반드시 `<img>` 태그 사용**

**Slide 3.4**: ⭐ Attention Map / Mask Visualization
- Text probe mask vs Image probe mask vs Hybrid mask
- 기존 `meeting_pack/outputs/mask_compare_*.png` 참고
- GradCAM-style overlay 시각화

**Slide 3.5**: COCO FP + FID (Benign Preservation)
- COCO FP%, FID, CLIP Score 비교

**Slide 3.6**: VQAScore Results
- Original prompt alignment + Anchor alignment
- Country nude body 결과 포함

**Slide 3.7** (optional): I2P 7-Category Results
- Training Safe Denoiser Table 3 형식

### Part 4: Ablation Study (2-3 slides)

**Slide 4.1**: WHERE Ablation
- Noise CAS vs CrossAttn Text vs CLIP Patch vs Hybrid Fusion

**Slide 4.2**: Key Hyperparameter Sensitivity
- CAS threshold sweep
- Safety scale sweep
- Number of exemplar images trend

### Part 5: Discussion + Next Steps (2-3 slides)

**Slide 5.1**: CVPR Reviewer Feedback 대응
- VLM reliability: Human eval 계획
- Generalization: I2P multi-concept 결과 (pilot)
- Overhead: < 3% wall-clock overhead 측정

**Slide 5.2**: NeurIPS Paper Plan
- Paper structure outline
- Remaining experiments
- Timeline

**Slide 5.3**: Summary
- Key contribution: Training-free + Example-based (image+text) + Selective (when+where)
- Best results 한줄 요약

---

## 생성 필요 시각 자료

### 반드시 필요 (blocking)

| Asset | 생성 방법 | 용도 |
|-------|---------|------|
| Method comparison grid (5 prompts × 5 methods) | `generate_baseline.py` + `generate_v{N}.py` + SAFREE | Slide 3.3 |
| Attention mask comparison (text vs image vs hybrid) | `make_meeting_panels.py` 수정 | Slide 3.4 |
| SR vs NN Pareto scatter plot | matplotlib script | Slide 1.2 |
| Pipeline architecture diagram | HTML/CSS pipe diagram | Slide 2.1 |

### 있으면 좋음 (nice-to-have)

| Asset | 생성 방법 | 용도 |
|-------|---------|------|
| GradCAM overlay per version | `make_gradcam_panels.py` | Slide 3.4 |
| CLIP patch token spatial map | 새 스크립트 필요 | Slide 2.2 |
| Timestep adaptive schedule chart | matplotlib | Slide 2.5 |
| I2P radar chart (7 categories) | matplotlib | Slide 3.7 |

---

## meeting_pack 기존 자료 활용

`CAS_SpatialCFG/meeting_pack/outputs/` 에 있는 기존 이미지들:

| 파일 | 내용 | 재사용 가능 |
|------|------|------------|
| `compare_0000.png` ~ `compare_0015.png` | Method comparison grids | ✅ 형식 참고 |
| `gradcam_v4_v13_00/01.png` | v4 vs v13 GradCAM | ✅ 직접 사용 |
| `mask_compare_00/01/02.png` | Mask comparison | ✅ 직접 사용 |
| `mask_v4_overlay_*.png` | v4 mask overlay | ✅ 직접 사용 |
| `mask_v13_overlay_*.png` | v13 mask overlay | ✅ 직접 사용 |
| `v4_mask_pipeline.png` | v4 mask pipeline 시각화 | ✅ 직접 사용 |
| `mask_naked_beach_*` | Specific prompt mask vis | ✅ 참고 |

### meeting_pack HTML 구조

- `meeting_20260320.html`: v4 중심, base64 embedded images (매우 큰 파일)
- `meeting_20260327.html`: v13 Exemplar Probe, KaTeX, 가장 최신 CSS → **base template로 사용**

---

## 이미지 포함 방식 결정

### Option A: 상대 경로 참조 (권장)
```html
<img src="CAS_SpatialCFG/meeting_pack/outputs/compare_0000.png" class="img-r">
```
- 장점: 파일 크기 작음, 관리 용이
- 단점: 파일 이동 시 깨짐

### Option B: base64 inline (meeting_20260320.html 방식)
```html
<img src="data:image/png;base64,iVBOR..." class="img-r">
```
- 장점: 단일 파일로 완결
- 단점: HTML 파일 수 MB로 커짐

### Option C: 전용 assets 폴더 (최적)
```bash
mkdir -p presentation_assets/
# 필요한 이미지만 복사/생성
cp meeting_pack/outputs/compare_0000.png presentation_assets/
# HTML에서 참조
```
```html
<img src="presentation_assets/compare_0000.png" class="img-r" style="max-height:300px;">
```

**권장: Option C** — 깔끔하고 관리 용이

---

## 체크리스트: 발표 전 준비

- [ ] Best config 확정 (v14/v15/v16 중)
- [ ] Ring-A-Bell 전체 결과 확보 (NN%, SR%, VQA)
- [ ] Baseline 비교 결과 (SAFREE, SLD, SD baseline, DAG)
- [ ] Image comparison grid 생성 (5+ prompts)
- [ ] Attention mask visualization 생성
- [ ] COCO FP 결과
- [ ] Pipeline diagram (HTML/CSS)
- [ ] 결과 테이블 채우기
- [ ] presentation_assets/ 폴더 구성
- [ ] HTML 파일 로컬 테스트 (브라우저에서 열기)
