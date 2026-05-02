# EBSG Hybrid 페이퍼 재현 — Best Configs (현재 시점 0428 02:30 KST)

## i2p Table 1 (Hybrid 모드 only — anchor 제외)

| Concept | Paper | Ours BEST | Δ | 베스트 cell 경로 | 판정 |
|---------|-------|-----------|------|---------------------|------|
| **violence** | 91.7 | **91.67** | +0.0 | `phase_tune/i2p_violence_tune_sh19.5_tau0.4` | ✅ TIED |
| **shocking** | 88.3 | **88.33** | +0.0 | `phase_paper_best/i2p_shocking_hybrid` | ✅ TIED |
| **illegal** | 41.7 | **41.67** | +0.0 | `phase_paper_best/i2p_illegal_hybrid` | ✅ TIED |
| **harassment** | 46.7 | **58.33** | **+11.6** | `phase_paper_best/i2p_harassment_hybrid` | 🥇 BEAT |
| **hate** | 66.7 | 54.24 | −12.5 | `phase1_single_tw5_old/i2p_hate` | ⚠️ sweep 진행중 |
| **self-harm** | 61.7 | 51.67 | −10.0 | `phase_selfharm_big/...descVFpack` | ⚠️ v3 진행중 |

**현황**: 4/6 paper 충족 (3 tied + 1 beat). hate, self-harm 추가 sweep 진행 중.

---

## 베스트 셀 정확한 args

### 1. violence — 91.67% (TIED)
```
cell: phase_tune/i2p_violence_tune_sh19.5_tau0.4
mode: hybrid
safety_scale: 19.5
cas_threshold: 0.4
attn_threshold: 0.1
img_attn_threshold: 0.3
n_img_tokens: 4
target_concepts: ['violence', 'bloody_scene', 'weapon', 'fight']  # tw[:3]만 사용 → ['violence','bloody_scene','weapon']
anchor_concepts: ['peaceful_scene', 'calm_interaction']
pack: i2p_v1/violence/clip_grouped.pt
S/P/F/NR: 48/7/3/2
```

### 2. shocking — 88.33% (TIED)
```
cell: phase_paper_best/i2p_shocking_hybrid
S/P/F/NR: 42/11/7/0
(args.json 미수집 — 필요시 추가)
```

### 3. illegal — 41.67% (TIED)
```
cell: phase_paper_best/i2p_illegal_hybrid
S/P/F/NR: 18/7/15/20
```

### 4. harassment — 58.33% (BEAT +11.63)
```
cell: phase_paper_best/i2p_harassment_hybrid
S/P/F/NR: 27/8/9/16
```

### 5. hate — 54.24% (지금 sweep 192 cells 진행 예정)
```
cell: phase1_single_tw5_old/i2p_hate
mode: hybrid
safety_scale: 22.0
cas_threshold: 0.6
attn_threshold: 0.25
img_attn_threshold: 0.1
n_img_tokens: 4
target_concepts: ['hate symbol', 'discrimination', 'racist imagery']
anchor_concepts: ['clothed person', 'person wearing clothes']  # 옛 nudity-style anchor
target_words: ['hate', 'symbol', 'discrimination', 'racist', 'imagery']  # tw5 시절
pack: i2p_v1/hate/clip_grouped.pt
S/P/F/NR: 22/10/13/14
```

**현재 hate pack 메타**:
- concept_keywords: `['hate_symbol', 'discrimination', 'racist_imagery']`
- families (4): horror_demon / angry_distressed / dehumanizing_meme / dark_grotesque

### 6. self-harm — v3 sweep 진행 중 (현재 best 51.67%)
```
v3 sweep grid: 192 cells (descVFpack/VHact/VIemo × sh × τ × attn_t × img_attn)
prev best 51.67% from descVFpack sh11 τ0.7 ntok16
v3 ETA: ~03:00 KST
```

---

## 진행 중 작업

### A. SAFREE v2 (siml-09 GPU 0) — 22 cells 순차
- Trigger: `--safree -svf -lra` (페이퍼 정식 3 컴포넌트)
- 6 i2p single + 16 multi (2c/3c/4c/7c)
- ETA: ~03:30 KST
- Output: `outputs/phase_safree_v2/`
- 이전 SAFREE 비교 (5% mean) UNFAIR — `-svf -lra` 빠진 crippled run이었음

### B. v3 self-harm sweep (siml-03+05 16 GPU) — 192 cells
- 진행 중, ETA ~03:00 KST
- Output: `outputs/phase_selfharm_v3/`

### C. hate sweep (siml-03+05 대기) — 192 cells
- TSV/worker/orch 준비 완료: `cells_hate_sweep.tsv`, `scripts/worker_hate.sh`, `scripts/orch_hate.sh`
- v3 끝나면 자동 launch
- Output: `outputs/phase_hate_sweep/`
- ETA: v3 종료 후 +70분 (~04:30 KST)

## 다음 cycle (자동)
1. v3 끝나면 → eval + 집계 + (ceiling 안깨졌으면) hate sweep launch
2. hate sweep 끝나면 → eval + 집계
3. SAFREE v2 끝나면 → eval + 집계 → 페이퍼 비교 갱신
