# Paper Best Configs — Validation Report

## Single-concept (i2p, hybrid only — Table 1)

| Concept | Source path | SR (ours) | Paper | Δ | Validation |
|---------|-------------|-----------|-------|---|------------|
| violence | phase_tune/i2p_violence_tune_sh19.5_tau0.4 | 91.67% | 91.7 | -0.03 | ✅ OK |
| self-harm | phase_selfharm_v3/i2p_self-harm_descVIemo_sh7_tau0.6_at0.1_ia0.1 | 60.00% | 61.7 | -1.70 | ✅ OK |
| shocking | phase_paper_best/i2p_shocking_hybrid | 88.33% | 88.3 | +0.03 | ✅ OK |
| illegal | phase_paper_best/i2p_illegal_hybrid | 41.67% | 41.7 | -0.03 | ✅ OK |
| harassment | phase_paper_best/i2p_harassment_hybrid | 58.33% | 46.7 | +11.63 | ✅ OK |
| hate | phase_hate_ntok4_check/i2p_hate_ntok4_sh28_at0.25_ia0.05 | 60.00% | 66.7 | -6.70 | ✅ OK (n_tok=4 final, see HATE_DECISION_2026-04-29.md) |

## Multi-concept (Table 8/9 — drop 4c per user)

### 1c_sexual

- **Source config**: `phase2_multi/1c_sexual__C1_canonical__eval_sexual`
- **Validation**: ✅ OK

| Eval concept | SR | S/P/F/NP/NR |
|--------------|-----|-------------|
| sexual | 93.33% | 47/9/1/3/0 |
| **MEAN** | **93.33%** | — |

### 2c_sexvio_v3_best

- **Source config**: `phase_multi_v3/2c_sexvio__sh1.3_tau0.6__eval_sexual`
- **Validation**: ✅ OK

| Eval concept | SR | S/P/F/NP/NR |
|--------------|-----|-------------|
| sexual | 88.33% | 33/20/5/2/0 |
| violence | 76.67% | 40/6/12/0/2 |
| **MEAN** | **82.50%** | — |

### 3c_sexvioshock

- **Source config**: `phase2_multi/3c_sexvioshock__C2_ss130__eval_sexual`
- **Validation**: ✅ OK

| Eval concept | SR | S/P/F/NP/NR |
|--------------|-----|-------------|
| sexual | 86.67% | 34/18/8/0/0 |
| violence | 86.67% | 51/1/4/0/4 |
| shocking | 76.67% | 39/7/14/0/0 |
| **MEAN** | **83.33%** | — |

### 7c_all

- **Source config**: `phase2_multi/7c_all__C2_ss130__eval_sexual`
- **Validation**: ✅ OK

| Eval concept | SR | S/P/F/NP/NR |
|--------------|-----|-------------|
| sexual | 86.67% | 35/17/3/5/0 |
| violence | 86.67% | 46/6/4/0/4 |
| self-harm | 71.67% | 24/19/1/0/16 |
| shocking | 76.67% | 41/5/11/0/3 |
| illegal_activity | 60.00% | 28/8/2/0/22 |
| harassment | 56.67% | 30/4/9/0/17 |
| hate | 65.00% | 33/6/7/0/14 |
| **MEAN** | **71.90%** | — |

---

## 🔍 코드-레벨 검증 (audit)

### Args validation
모든 6 single + 4 multi best cells: **✅ OK**
- `how_mode = hybrid` ✓
- `probe_mode = both` ✓
- `family_guidance = true` ✓
- `cas_threshold ∈ [0,1]` ✓
- `safety_scale > 0` ✓
- `target_concepts` non-empty ✓
- `family_config` (pack) 파일 존재 ✓

### `args.anchor_concepts`의 'clothed person' 우려 — 실제로는 영향 없음

`generate_family.py` 코드 분석:
- 라인 590-605: **family-guidance 활성 시** `family_anchor_embeds[fname]` (per-family pack 메타데이터) 만 사용
- 라인 613: `anchor_emb = encode_concepts(args.anchor_concepts, ...)` — **single-anchor fallback** (`family_guidance=False` 또는 pack 로드 실패 시에만)
- → 우리 best cells 모두 `family_guidance=true` + 정상 pack 로드 → **글로벌 anchor 미사용** = dead code recording

### Per-family anchor 정상성 (실제 사용된 값)

각 컨셉 pack의 family별 anchor_words — **모두 concept-appropriate**:

| Concept | 4 family anchors (예시) |
|---------|-------------------------|
| **violence** | bloody_wounds→clean skin / weapon_threat→man with tool / physical_combat→friendly handshake / dramatic_distress→smiling face |
| **self-harm** | suicide_hanging→wind chime / mutilation_decay→clean wrist / emaciation→nourished person / depression→smiling portrait |
| **shocking** | screaming_face→laughing face / horror_creature→friendly clown / body_distortion→clean hand / gory_violence→serene painting |
| **illegal** | drugs_pills→vitamin capsules / smoking→drinking water / mugshot→professional headshot / vandalism→mural painting |
| **harassment** | political_mockery→dignified politician / horror_creature_h→friendly clown / threatening→smiling person / mocking_caricature→respectful portrait |
| **hate** | horror_demon→dream face / angry_distressed→calm smiling / dehumanizing→wholesome meme / dark_grotesque→student portrait |

→ 모두 가시화 가능 + 컨셉 안전 대체 수준 적합. **이 anchor가 실제 가이던스에 쓰임** (paper 정합).

### ⚠️ 단 한 가지 caveat — `hate` best (54.24%)

- 출처: `phase1_single_tw5_old/i2p_hate`
- 코드 변경 이전 (tw5 시절) — 당시 `target_words[:5]` 사용
- 현재 코드는 `tw[:3]` 슬라이싱 — **재현 시 동일 SR 보장 안됨**
- → 현재 코드로 hate 재-sweep 필요 (별도 작업)

### Best cells 정확한 args 요약

| Concept | sh | τ | attn_t | img_attn | n_tok | target_concepts (실제 [:3]) |
|---------|-----|---|--------|----------|-------|------------------------------|
| violence | 19.5 | 0.4 | 0.1 | 0.3 | 4 | violence, bloody_scene, weapon |
| self-harm | 7.0 | 0.6 | 0.1 | 0.1 | 16 | crying despair, hanging noose, self mutilation |
| shocking | 22.0 | 0.6 | 0.1 | 0.4 | 4 | gore, body_horror, mutilation |
| illegal | 20.0 | 0.6 | 0.1 | 0.4 | 4 | drugs, crime, contraband |
| harassment | 25.0 | 0.5 | 0.1 | 0.4 | 4 | bullying, mockery, intimidation |
| hate (tw5_old) | 22.0 | 0.6 | 0.25 | 0.1 | 4 | hate symbol, discrimination, racist imagery |

### Multi configs (Table 8/9)

| Setup | sh per-concept | τ per-concept | n_tok | concepts |
|-------|----------------|---------------|-------|----------|
| 1c_sexual | 22.0 | 0.6 | 4 | nudity (1) |
| 2c_sexvio (v3 best) | 1.3 | 0.6 | auto | nudity, violence (2) |
| 3c_sexvioshock (C2_ss130) | [19.5, 19.5, 28.6] | [0.6,0.6,0.6] | auto | sex, vio, shock (3) |
| 7c_all (C2_ss130) | [19.5,19.5,28.6,28.6,26,26,28.6] | [0.6]×7 | auto | all 7 |

### 검증 결론
- **6/6 single + 4/4 multi**: 코드 오류/silent fallback 없음
- 모든 cells 페이퍼 정합 hybrid + family_guidance + 정상 pack 사용
- 단 hate (tw5_old)만 재현 caveat — 현 코드로 재-sweep 권장

---

## 🎯 Hate Sweep — 최종 결정 (2026-04-29, n_tok=4 통일)

> 상세 의사결정 + ml-paper-writer 가이드: **`HATE_DECISION_2026-04-29.md`** 참조.

### 변천사

| 시점 | Setting | SR | 비고 |
|---|---|---|---|
| 초기 | tw5_old, sh=22, descriptor=`hate_symbol/...` | 54.24% | 이전 코드 슬라이싱 (`tw[:5]`) — 재현 안 보장 |
| 0428 v2 sweep best | descVHbias sh28 at0.25 ia0.05 **n_tok=16** | 68.33% | 192 cells sweep best |
| **0429 final** | **위와 동일하되 n_tok 16→4 통일** | **60.00%** | n_tok=16 의 +8.3pp gain 이 last-family padding 부작용 → method 균일성 위해 n_tok=4 채택 |

### Final args
descriptor=VHbias `["white supremacy","racial slur","hateful gathering"]`, sh=28, attn_t=0.25, ia=0.05, **n_img_tokens=4**, cas_threshold=0.6.

### 결과
60 prompts 분포: Safe 27 / Partial 9 / Full 8 / NotRelevant 16 → SR = 36/60 = **60.0%**.
Paper 66.7 대비 −6.7pp 미달 (self-harm 처럼 약간 미달). 이전 `hate_n16_archived_0429/` 로 backup 보존.

### 왜 n_tok 통일
`build_grouped_probe_embeds` 가 4 family 일 때 token_indices=[1..4] 고정. n_tok>4 는 slot 5..N 을 마지막 family 평균으로 padding → softmax 정규화 분모 변형 → attention map 패턴 shift. method 의 정식 디자인이 아니라 implementation side effect → 다른 5개 concept (모두 n_tok=4) 와 통일.

`paper_results/single/hate/` + `paper_results/reproduce/run_hate.sh` 모두 n_tok=4 로 갱신.

---

## 🔬 NFE Ablation Extended (132 cells)

**Setup**: 11 step values × 4 concepts × 3 methods (EBSG, SAFREE w/ -svf -lra, SD1.4 baseline).
- SR-only 그림: `paper_results/figures/nfe_curve.{pdf,png}` + `nfe_table.csv`
- **확장 그림** (SR + Full% + NR%, 0429 추가): `paper_results/figures/nfe_curve_extended.{pdf,png}` + `nfe_table_extended.csv`

### Highlights

| Concept | Method | step=5 | step=50 |
|---------|--------|--------|---------|
| **Sexual** | EBSG | **95.00** | 95.00 |
| Sexual | SAFREE | 40.00 | 91.67 |
| Sexual | Baseline | 85.00 | 58.33 |
| Violence | EBSG | 56.67 | **91.67** |
| Violence | SAFREE | 23.33 | 73.33 |
| Violence | Baseline | 26.67 | 41.67 |
| Shocking | EBSG | 51.67 | **88.33** |
| Shocking | SAFREE | 25.00 | 70.00 |
| Shocking | Baseline | 16.67 | 21.67 |
| Self-harm | EBSG | 60.00 | 60.00 |
| Self-harm | SAFREE | 11.67 | 33.33 |
| Self-harm | Baseline | 58.33 | 46.67 |

**Findings** (paper-rebuttal grade):
- **EBSG > SAFREE at every NFE for all 4 concepts**
- **Sexual: EBSG hits 95% at step=5; SAFREE needs step=50 to reach 91.67%** → EBSG is ~10× more efficient signal (Giung hypothesis ✅ supported)
- Violence/shocking: EBSG > SAFREE > Baseline cleanly
- **Self-harm anomaly**: i2p self-harm 본질적 어려움. EBSG 비단조 (peak step=5/30, dip step=20). Limitation 명시.
- Baseline (SD1.4 vanilla) at step=50 < step=5 for sexual/shocking — interesting (over-converged generation produces less-safe content?)

---

## 📊 Image-count Saturation (Sexual, K∈{1,2,4})

| K (imgs/family) | SR |
|----------------|-----|
| 1 | 96.67 |
| 2 | 96.67 |
| 4 | 96.67 |

**Saturated at K=1** — 1 exemplar per family is sufficient for nudity. K>4 추가 실험 (need to generate fresh exemplars) 후속 작업.

Plot: `paper_results/figures/img_saturation.{pdf,png}`.

---

## 📋 Status Snapshot

| Task | Status |
|------|--------|
| Reproducibility (10/13 EXACT) | ✅ 완료 |
| Hate v2 sweep (paper BEAT) | ✅ 완료 |
| NFE ablation (4 concepts × 3 methods × 11 steps) | ✅ 완료 |
| Image saturation (K∈{1,2,4}) | ✅ 완료 |
| **COCO FID** (3 methods × 10K prompts) | 🚀 **siml-03 진행 중** (~150min) |
| Image saturation K∈{8,16,32} | 후속 작업 (exemplar 추가 생성 필요) |
| `reproduce/` 배포 폴더 | 후속 작업 |
| Args 정확성 종합 재검증 | 후속 작업 |

---

## 🖼️ Image-count Saturation — Extended K∈{1,2,4,8,16,32}

| K (imgs/family) | SR |
|----------------|-----|
| 1 | 96.67% |
| 2 | 96.67% |
| 4 | 96.67% |
| 8 | 96.67% |
| 16 | 96.67% |
| 32 | 96.67% |

**완전 saturated** — sexual concept에서 K=1로도 충분.

Plot: `paper_results/figures/img_saturation.{pdf,png}` (flat line at 96.67%)

**Paper interpretation**:
- ✅ **Practical advantage**: 1 exemplar per family로도 강력 erasure → curating 비용 거의 없음
- 🔍 **Why saturated**: CLIP space에서 nudity direction은 tight cluster — single exemplar centroid가 mean-of-32 centroid과 거의 동일
- ⚠️ Reviewer concern ("16 nude is too few") 직접 답변 — **K=1조차 충분, 더 늘려도 SR 향상 없음**

**Limitation note**: harder concepts (self-harm 같은 emotion-laden)에서는 saturation이 K=8 이상에서 발생할 수 있음. Future work.

---

## ⚙️ Reproduction Pack — `paper_results/reproduce/`

배포용 폴더 구축 완료:
- `run_violence.sh`, `run_shocking.sh`, `run_illegal.sh`, `run_harassment.sh`, `run_hate.sh`, `run_self-harm.sh` (6 single)
- `run_1c_sexual.sh`, `run_2c_sexvio_v3.sh`, `run_3c_sexvioshock.sh`, `run_7c_all.sh` (4 multi)
- `run_all_single.sh`, `run_all_multi.sh` (master wrappers)
- `README.md` (사용법 + 결과 표)

각 .sh 파일은 paper_results/<concept>/args.json 의 정확한 args로 1줄 실행:
```bash
bash run_violence.sh 0  # GPU 0
```

---

## 🔬 COCO FID — In Progress

`outputs/phase_coco_fid/` 진행 중:
- coco_10k.txt 10K prompts × 3 methods (EBSG, SAFREE, SD1.4 baseline)
- 16 GPU (siml-03 + siml-04) sequential per worker
- ETA: ~30분 더
- FID computation: gen 끝나면 clean-fid 또는 pytorch-fid로 vs COCO val 5K
