# Unlearning Project — Claude Code 지침서

## 프로젝트 개요
Text-to-Image Diffusion Model에서 nudity 등 unsafe concept을 training-free로 제거하는 연구.
핵심 접근: **Example-based When + Where guidance** — 언제(CAS) 어디서(Spatial) 가이드를 줄지 sample 기반으로 결정.

## Canonical references
- 저장소 분류: `docs/repo-taxonomy.md`
- 빠른 구조 안내: `docs/repo-layout.md`
- 현재 표준 작업 흐름: `docs/active-workflow.md`
- 경로/환경 설정: `docs/runtime-config.md`
- 평가 지표/파일 계약: `docs/metrics-contract.md`

## 현재 표준 작업 요약
- **활성 코어 폴더**: `CAS_SpatialCFG/`, `SAFREE/`, `vlm/`, `scripts/`, `docs/`
- **비교용 2차 baseline**: `AMG/`, `SoftDelete+CG/`
- top-level의 다수 classification/SDXL 계열 폴더는 기본적으로 **legacy/archive 실험 브랜치**
- 새 작업은 기본적으로 active core에서 시작하고, legacy 브랜치는 명시적 재현/비교 목적일 때만 사용

## Conda 환경
| Task | Env |
|------|-----|
| Generation (diffusers) | `sdd_copy` |
| NudeNet | `sdd_copy` |
| Qwen3-VL | `vlm` |

## 프로젝트 고정 실험 상수
- **4 Nudity Datasets**: Ring-A-Bell, MMA, P4DN, UnlearnDiff
- **I2P subset 확장**: `i2p_categories/` 하위 concept별 CSV 사용
- **Benign (FP check)**: COCO prompts
- **기본 모델**: `CompVis/stable-diffusion-v1-4`
- **필수 조건**: `safety_checker=None`
- **기본 생성 설정**: 1 sample/prompt, 50 steps, DDIM, `cfg_scale=7.5`, `seed=42`
- **Qwen nudity canonical outputs**: `categories_qwen_nudity.json`, `results_qwen_nudity.txt`
- **Canonical SR**: `(Safe + Partial) / Total`

## GPU 사용
- **nvidia-smi로 빈 GPU 확인 후 사용** — 특정 GPU 하드코딩 금지
- 다른 사람과 공유 자원

## 핵심 연구 방향
1. **When**: CAS (Concept Alignment Score) — cos(d_prompt, d_target) 기반 harmful 탐지
2. **Where**: Spatial guidance — cross-attention map 또는 noise/feature 차이 기반
3. **How**: SLD-style safe CFG — ε_safe = ε_cfg - s * mask * (ε_target - ε_∅)
4. **목표**: COCO FP ↓, Ring-A-Bell TP ↑, 이미지 퀄리티 보존

## 대화 내역 관리 (필수)
- **매 대화 시작 시**: 이전 session log를 읽고 맥락 파악
- **매 대화 중**: 주요 결정사항, 실험 결과, 발견한 문제 등을 실시간으로 session log에 기록
- **매 대화 종료 시**: session log 최종 업데이트 (날짜, 진행 내용, 다음 할일)
- **저장 위치**: `/mnt/home3/yhgil99/.claude/projects/-mnt-home-yhgil99-unlearning/memory/`
  - 파일명: `session_YYYY_MMDD_주제.md` (예: `session_2025_0316_cas_spatial_cfg.md`)
  - MEMORY.md 인덱스에 포인터 추가

## 연구 로드맵: Multi-Concept Erasing

### Phase 1 (현재): Nudity Erasing
- **Status**: 진행중 — CAS + Spatial CFG 방법론 개발 및 최적화
- **Erase target**: sexual nudity, exposed body parts
- **Anchor concept**: clothed person, portrait, fully dressed figure
- **평가**: NudeNet (자동 body-part detection) + Qwen3-VL (semantic 판단)
- **Datasets**: Ring-A-Bell, MMA, P4DN, UnlearnDiff (기존 4종)
- **I2P subset**: `i2p_sexual.csv` (931 prompts)

### Phase 2 (이후): I2P 기반 Multi-Concept Erasing
Nudity 실험 완료 후, 동일 방법론을 다른 unsafe concept으로 확장.
각 concept별 erase target과 anchor를 아래에 정의한다.

| # | Concept | I2P File | Prompts | Erase Target | Anchor (safe guidance 방향) | 평가 방법 |
|---|---------|----------|---------|--------------|---------------------------|-----------|
| 1 | **Violence** | `i2p_violence.csv` | 756 | 피, 무기, 상해, 전투, gore | 평화로운 장면, 자연 풍경, 일상적 상호작용 | Qwen3-VL (violence category) |
| 2 | **Harassment** | `i2p_harassment.csv` | 824 | 위협, 괴롭힘, 모욕적 행위, 공격적 제스처 | 우호적 대화, 협력, 존중하는 상호작용 | Qwen3-VL (harassment category) |
| 3 | **Self-harm** | `i2p_self-harm.csv` | 801 | 자해, 자살 암시, 신체 손상 묘사 | 치유, 명상, 건강한 활동, 심리 안정 | Qwen3-VL (self-harm category) |
| 4 | **Shocking** | `i2p_shocking.csv` | 856 | 혐오 이미지, grotesque, body horror, 기괴한 변형 | 평온한 장면, 아름다운 예술, 조화로운 구성 | Qwen3-VL (shocking category) |
| 5 | **Illegal Activity** | `i2p_illegal_activity.csv` | 727 | 마약, 범죄 행위, 불법 무기, 절도/파괴 | 합법적 일상, 직업 활동, 사회적 규범 | Qwen3-VL (illegal category) |
| 6 | **Hate** | `i2p_hate.csv` | 231 | 인종/성별/종교 차별, 혐오 상징, 비하 묘사 | 다양성, 화합, 문화적 존중, 평등 | Qwen3-VL (hate category) |

#### Phase 2 실험 시 핵심 고려사항
1. **Anchor 설계**: 각 concept별로 exemplar anchor prompt 세트 (16~20개) 별도 제작 필요
   - Nudity에서 `full_nudity_exemplar_prompts.txt` 만들었듯이, concept별로 anchor prompts 파일 준비
   - Anchor는 erase target의 "safe 대응물"로, concept direction (d_target) 계산에 사용
2. **CAS threshold 재조정**: concept마다 적절한 threshold가 다를 수 있음
   - Nudity에서는 0.6이 최적이었으나, violence/hate 등은 별도 sweep 필요
3. **Cross-concept interference**: 한 concept을 지울 때 다른 concept에 영향 없는지 확인
   - 예: violence 지울 때 action scene 전체가 망가지면 안 됨
4. **Multi-concept simultaneous erasing**: 최종 목표는 여러 concept 동시 제거
   - 개별 concept 실험 → pairwise → full multi-concept 순서로 확장
5. **평가 파이프라인 확장**:
   - Qwen3-VL의 `opensource_vlm_i2p_all.py`에서 category 인자만 변경하면 됨
   - NudeNet은 nudity 전용이므로, 다른 concept은 VLM 평가가 주력

#### Phase 2 우선순위 (제안)
1. **Violence** — nudity 다음으로 가장 연구가 많고, 비교 baseline 확보 용이
2. **Harassment + Hate** — 사회적 영향 크고, NeurIPS 리뷰어 관심 높을 분야
3. **Self-harm + Shocking** — 유사한 anchor 전략 공유 가능
4. **Illegal Activity** — 가장 다양한 subcategory, 마지막 확장

### Phase 3 (장기): Unified Multi-Concept Framework
- 단일 inference에서 N개 concept 동시 erasing
- Concept별 CAS threshold + spatial mask의 ensemble
- I2P 전체 4709 prompts로 종합 평가 (all-category)

## 주의사항
- 새 방법론은 반드시 **독립 폴더**에 구현 (기존 폴더에 섞지 말 것)
- 매 대화마다 **session log** 저장 (memory 폴더)
- 실험 결과는 항상 NudeNet + Qwen3-VL 까지 돌려야 완결
- 검은 이미지 = safety_checker 또는 latent 파괴 → 반드시 확인
- nohup 스크립트로 실험 돌리면 세션 종료 후에도 자동 진행 가능
- 다른 사람이 사용하고 있는 yhgil99외의 사용자가 사용하는 gpu는 절대 사용하지 말 것. 코드를 돌리는 것은 고사하고 kill은 더더욱 안됨.
- yhgil99의 프로세스를 kill할때는 bypass 모드여도 나한테 물어보고 할것
