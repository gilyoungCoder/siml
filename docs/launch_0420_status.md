# Launch 2026-04-20 Status Document

작성: 2026-04-20 07:35 KST

## 실험 목표
SD1.4, SD3, FLUX1 3 backbones × MJA(4 datasets) + RAB × {baseline, SAFREE, ours sweep, multi-concept(SD1.4 only)}.
Eval: Qwen3-VL with v2 + v3 prompt rubrics.

## GPU Pool (strict isolation)
- siml-01 GPU 0-7 (24GB × 8) — yhgil99 단독
- siml-06 GPU 4-7 (49GB × 4) — 0-3은 다른 사용자 (HARD FORBIDDEN)
- siml-06 GPU 0-3 (49GB × 4) — 추가 허용됨 (verified empty 후 사용)
- siml-08 GPU 4,5,6,7 (49GB × 4) — 0-3은 giung2 (HARD FORBIDDEN)
- siml-09 GPU 0 (97GB H100)
- 총 17 GPU 활용

## Sweep config
- probe_mode = both (fixed)
- cas_threshold = 0.6 (fixed)
- safety_scale: SD1.4/SD3 ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}, FLUX1 ∈ {1.5, 2.0, 2.5, 3.0}
- attn_threshold: RAB ∈ {0.05, 0.1, 0.2}, MJA ∈ {0.1, 0.2}, FLUX1 MJA ∈ {0.1}
- how_mode ∈ {anchor_inpaint, hybrid}
- family_guidance enabled (3-block hooks for SD3)

## 완료된 코드 작업
1. FLUX1 family probe 구현 (5 patches in `generate_flux1_v1.py`, helper in `attention_probe_flux.py`)
2. SD3 family text-probe 패치 (`generate_sd3_safegen.py`)
3. FLUX1 SAFREE 신규 구현 (`generate_flux1_safree.py`, 814 LOC, token filter + re-attention; latent filter no-op due to dim mismatch)
4. SafeGen `generate_family.py` 패치: v2 clip_grouped.pt 포맷 fallback (family_token_map 없어도 family_names로 로드)
5. SD3 attention_probe 3-block hook 축소 (메모리 60GB → 33-38GB, A6000 49GB에 fit)
6. Multi-concept pack builder (sexual+violent → 8 families merged, `CAS_SpatialCFG/exemplars/multiconcept_v1/sexual+violent/clip_grouped.pt`)

## Worker 스크립트 (모두 `scripts/launch_0420/`)
- `worker.sh` (master agent, manifest 기반, eval inline)
- `sd3_ours_worker_v2.sh` (siml-06 4-7, 3-block patch, wait-for-free)
- `flux1_ours_worker.sh` (siml-06 0-3 + siml-08 6,7 + siml-09 0, N_SLOTS=7)
- `multiconcept_worker.sh` (siml-01 g4, g6)
- `flux1_safree_worker.sh` (siml-09 g0, 5 datasets sequential)
- `sd3_safree_siml09.sh` (siml-09 g0)
- `sd14_safree_worker.sh` (siml-01 g0, g2)
- `eval_batch_worker.sh` (siml-01 g2, infinite loop with idempotent skip via `categories_qwen3_vl_*.json` + `.eval_v3_qwen3_vl_*.done` sentinel)

## 진행 현황 (07:30 KST 기준)
| 작업 | 진행 | ETA |
|---|---|---|
| Wave 1 baselines (15) | ✅ 15/15 완료 | done |
| FLUX1 SAFREE | ✅ 5/5 완료 | done (07:25) |
| SD3 SAFREE | 8 outdirs / 279 imgs (~58%) | ~1h |
| SD1.4 SAFREE | 8 outdirs / 200 imgs (~42%) | ~1h |
| SD1.4 ours sweep | 6786/13200 imgs (51%) | ~5h |
| SD3 ours sweep (3-block patch) | 3456/13200 imgs (26%) | ~14h ← 율속 |
| FLUX1 ours sweep (7 GPU) | 745/4800 imgs (15%) | ~4-5h (after 7 GPU 분산) |
| Multi-concept (sexual+violent) | 1742/4800 imgs (36%) | ~6h |
| Qwen3-VL eval (master inline) | 224 outdirs evaluated | ongoing |
| Eval batch (siml-01 g2) | 방금 재launch (filename 패치 후) | infinite loop |

## 알려진 이슈
1. **master worker_siml-09 죽음** (06:17 KST): `--no_cpu_offload` flag 미지원으로 19/22 FLUX1 ours job 실패 → 별도 worker로 대체
2. **eval_batch filename pattern** 불일치: 처음 `categories_qwen_*` 패턴 사용 → 수정 후 `categories_qwen3_vl_*` (master agent와 일치). 재launch됨

## 출력 위치
- Generation: `CAS_SpatialCFG/outputs/launch_0420/{baseline|safree|ours}_{sd14|sd3|flux1}/{dataset}/{config}/*.png`
- Eval: 각 outdir에 `categories_qwen3_vl_<concept>.json`, `results_qwen3_vl_<concept>.txt`
- Logs: `logs/launch_0420/`

## 모니터링 명령어
```bash
# 전체 GPU 상태
for h in siml-01 siml-06 siml-08 siml-09; do echo "=== $h ==="; ssh $h "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"; done

# 결과 카운트
ssh siml-09 "for cat in baseline_sd14 baseline_sd3 baseline_flux1 safree_sd14 safree_sd3 safree_flux1 ours_sd14 ours_sd3 ours_flux1 ours_sd14_multiconcept; do c=\$(find /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0420/\$cat -name '*.png' 2>/dev/null | wc -l); echo \"\$cat: \$c imgs\"; done"

# 평가 진행
ssh siml-09 "find /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0420 -name 'categories_qwen3_vl_*.json' | wc -l"
```

## 다음 단계 (전부 끝나면)
1. eval batch이 모든 outdir 평가 완료 대기 (~24h)
2. Wave 5: aggregation 표 생성
   - per-backbone × per-method × per-dataset SR (Safe+Partial)/Total
   - v2 vs v3 비교
   - best config per dataset 추출
