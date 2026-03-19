# Unlearning Project — Claude Code 지침서

## 프로젝트 개요
Text-to-Image Diffusion Model에서 nudity 등 unsafe concept을 training-free로 제거하는 연구.
핵심 접근: **Example-based When + Where guidance** — 언제(CAS) 어디서(Spatial) 가이드를 줄지 sample 기반으로 결정.

## 실험 프로세스 (매번 반복)

### 1. 이미지 생성
- **4 Nudity Datasets**: Ring-A-Bell, MMA, P4DN, UnlearnDiff
  - CSV 원본: `/mnt/home/yhgil99/unlearning/SAFREE/datasets/`
  - Ring-A-Bell: `nudity-ring-a-bell.csv` (79 prompts, col: "sensitive prompt")
  - MMA: `mma-diffusion-nsfw-adv-prompts.csv` (1000+, 실험시 200개 제한)
  - P4DN: `p4dn_16_prompt.csv`
  - UnlearnDiff: `unlearn_diff_nudity.csv`
- **Benign (FP check)**: COCO prompts
- **Model**: `CompVis/stable-diffusion-v1-4`, **safety_checker=None 필수**
- **Settings**: 4 samples/prompt, 50 steps, DDIM, cfg_scale=7.5, seed=42

### 2. NudeNet 평가
```bash
# Env: sdd_copy
CUDA_VISIBLE_DEVICES=<gpu> /mnt/home/yhgil99/.conda/envs/sdd_copy/bin/python \
    /mnt/home/yhgil99/unlearning/vlm/eval_nudenet.py <img_dir>
```
→ Output: `results_nudenet.txt` (Unsafe %)

### 3. Qwen3-VL 평가 (NOT Qwen2-VL!)
```bash
# Env: vlm
CUDA_VISIBLE_DEVICES=<gpu> /mnt/home/yhgil99/.conda/envs/vlm/bin/python \
    /mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py <img_dir> nudity qwen
```
→ Output: `categories_qwen_nudity.json`, `results_qwen_nudity.txt`
→ Categories: NotRel, Safe, Partial, Full

### 4. 결과 집계
- **SR = (Safe + Partial) / (Safe + Partial + Full) × 100** — NotRel 제외!
- NudeNet Unsafe%: 낮을수록 좋음
- COCO FP: trigger rate 낮을수록 좋음

## Conda 환경
| Task | Env |
|------|-----|
| Generation (diffusers) | `sdd_copy` |
| NudeNet | `sdd_copy` |
| Qwen3-VL | `vlm` |

## GPU 사용
- **nvidia-smi로 빈 GPU 확인 후 사용** — 특정 GPU 하드코딩 금지
- 다른 사람과 공유 자원

## 코드 구조
```
/mnt/home/yhgil99/unlearning/
├── CAS_SpatialCFG/           # 방법 1: CAS + Spatial CFG (SLD/DAG style)
│   ├── generate.py           # v3 메인 (SLD/DAG style, noise-based spatial)
│   ├── generate_baseline.py  # Baseline (no guidance)
│   ├── scripts/run_v3.sh     # 전체 파이프라인
│   ├── prompts/              # 데이터셋
│   └── outputs/v3/           # 결과
├── AMG/                      # 방법 2: Activation Matching Guidance
│   ├── generate.py           # AMG 메인 (h-space feature matching)
│   ├── generate_baseline.py  # Baseline
│   ├── scripts/run_amg.sh    # 전체 파이프라인
│   ├── prompts/              # 데이터셋
│   └── outputs/              # 결과
├── vlm/                      # 평가 스크립트 (공통)
│   ├── eval_nudenet.py       # NudeNet
│   └── opensource_vlm_i2p_all.py  # Qwen3-VL (nudity qwen)
├── SAFREE/datasets/          # 원본 CSV 데이터
└── CLAUDE.md                 # 이 파일
```

## 핵심 연구 방향
1. **When**: CAS (Concept Alignment Score) — cos(d_prompt, d_target) 기반 harmful 탐지
2. **Where**: Spatial guidance — cross-attention map 또는 noise/feature 차이 기반
3. **How**: SLD-style safe CFG — ε_safe = ε_cfg - s * mask * (ε_target - ε_∅)
4. **목표**: COCO FP ↓, Ring-A-Bell TP ↑, 이미지 퀄리티 보존

## 대화 내역 관리 (필수)
- **매 대화 시작 시**: 이전 session log를 읽고 맥락 파악
- **매 대화 중**: 주요 결정사항, 실험 결과, 발견한 문제 등을 실시간으로 session log에 기록
- **매 대화 종료 시**: session log 최종 업데이트 (날짜, 진행 내용, 다음 할일)
- **저장 위치**: `/mnt/home/yhgil99/.claude/projects/-mnt-home-yhgil99-unlearning/memory/`
  - 파일명: `session_YYYY_MMDD_주제.md` (예: `session_2025_0316_cas_spatial_cfg.md`)
  - MEMORY.md 인덱스에 포인터 추가

## 주의사항
- 새 방법론은 반드시 **독립 폴더**에 구현 (기존 폴더에 섞지 말 것)
- 매 대화마다 **session log** 저장 (memory 폴더)
- 실험 결과는 항상 NudeNet + Qwen3-VL 까지 돌려야 완결
- 검은 이미지 = safety_checker 또는 latent 파괴 → 반드시 확인
- nohup 스크립트로 실험 돌리면 세션 종료 후에도 자동 진행 가능
