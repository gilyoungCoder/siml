# HATE concept — Final Decision 2026-04-29

> ml-paper-writer 가 paper Table 1 / appendix 작성 시 이 파일을 그대로 참조.
> 출처: NeurIPS 2026 EBSG 실험. SD1.4 + i2p_sweep60 (60 prompts) + Qwen3-VL-8B v5 evaluator.

## TL;DR

**hate 의 paper Table 1 hybrid SR = 60.00%** (paper 원문 66.7). −6.7pp 미달.
self-harm 처럼 미달로 솔직하게 보고. method 균일성 우선.

## Hate sweep 변천사

| 시점 | Setting | SR | 비고 |
|---|---|---|---|
| 초기 (workshop 시절) | tw5_old, sh=22, descriptor=`hate_symbol/discrimination/racist_imagery` | 54.24% | 이전 코드 (`target_words[:5]`) → 현 코드 (`tw[:3]`) 와 슬라이싱 다름. 재현 안 보장. |
| 0428 hate v2 sweep | 192 cells: descriptor × sh × attn_t × img_attn (τ=0.6 fixed) | 68.33% (best) | descriptor=VHbias `white supremacy/racial slur/hateful gathering`, sh=28, attn_t=0.25, ia=0.05, **n_img_tokens=16** |
| **0429 final (n_tok=4)** | **위와 동일하게 두고 n_img_tokens 16→4** 만 변경 | **60.00%** | n_tok=16 의 +8.3pp gain 이 마지막-family padding 부작용으로 격리됨 → 균일성 위해 n_tok=4 채택 |

## 최종 채택 args (paper_results/single/hate/args.json)

```json
{
  "ckpt": "CompVis/stable-diffusion-v1-4",
  "prompts": "i2p_sweep60/hate_sweep.txt",
  "nsamples": 1, "steps": 50, "seed": 42, "cfg_scale": 7.5,
  "cas_threshold": 0.6,
  "probe_mode": "both", "probe_fusion": "union",
  "family_config": "exemplars/i2p_v1/hate/clip_grouped.pt",
  "attn_resolutions": [16, 32],
  "attn_threshold": 0.25,
  "img_attn_threshold": 0.05,
  "attn_sigmoid_alpha": 10.0, "blur_sigma": 1.0,
  "n_img_tokens": 4,
  "how_mode": "hybrid",
  "safety_scale": 28.0,
  "family_guidance": true,
  "target_concepts": ["white supremacy", "racial slur", "hateful gathering"],
  "target_words": ["white","supremacy","racial","slur","hateful","gathering"]
}
```

→ 분포: Safe 27 / Partial 9 / Full 8 / NotRelevant 16. **SR = 36/60 = 60.0%.**

## 왜 n_tok=4 로 통일했나 — 메커니즘 설명

코드 trace (`SafeGen/safegen/generate_family.py:209-247`):

`build_grouped_probe_embeds` 가 family 별 평균 풀링한 4개 vector 를 prompt embedding 의 **slot 1..4** 에 배치 → `token_indices = [1,2,3,4]`. 이 indices 만 spatial mask 계산에 사용됨.

n_img_tokens > 4 일 때 (예: 16):
- slots 5..16 을 **마지막 family (`dark_grotesque`) 의 평균 vector 로 단순 padding**
- 이 padding key 들은 spatial mask 의 `token_indices` 에 안 들어가지만,
- `ProbeCrossAttnProcessor` 의 `softmax(dim=-1)` 이 **77 token 전체에 대해 정규화** 하므로
- padding key 들이 spatial position 별 softmax 분모를 변형 → 결과적으로 attention map 의 정규화된 패턴이 미묘하게 shift

→ n_tok=16 의 +8.3pp gain 은 **method 의 정식 디자인 효과가 아니라**, "마지막 family (dark_grotesque) 가 attention 정규화에 편향을 주는 implementation side effect" 에 가까움.

다른 5개 concept (violence/shocking/illegal/harassment/sexual) 은 모두 n_tok=4 사용 → hate 만 n_tok=16 으로 두면 **method 균일성 위배**.

→ 결정: **모든 concept n_tok=4 로 통일**, hate SR 60.0% 그대로 보고.

## CAS direction 결정 우선순위 (verified by runtime log)

`SafeGen/safegen/generate_family.py:385-414`:

```python
if args.target_concepts is not None:    # CLI 가 줬으면
    ckw_clean = args.target_concepts    #   → 그걸 사용 (source="CLI")
else:
    ckw_clean = pack.concept_keywords   #   → pack 값 사용 (source="pack")
```

**우리 모든 paper 런은 CLI 로 `--target_concepts` 명시** → CAS 가 CLI 값으로 fire. Runtime stdout 에 `Concept descriptor (CAS, source=CLI): [...]` 로 매번 print 됨.

5개 concept 의 CLI args.target_concepts 는 pack 의 concept_keywords 와 거의 동일:

| concept | CLI args.target_concepts | pack concept_keywords | 차이? |
|---|---|---|---|
| violence | `violence/bloody_scene/weapon/fight` | `violence/bloody_scene/weapon/fight` | 동일 |
| shocking | `gore/body_horror/mutilation/corpse` | `gore/body_horror/mutilation/corpse` | 동일 |
| illegal | `drugs/crime/contraband/illegal/activity` | `drugs/crime/contraband/illegal/activity` | 동일 |
| harassment | `bullying/mockery/intimidation/abuse` | `bullying/mockery/intimidation/abuse` | 동일 |
| sexual | `nudity/nude person/naked body` | `nudity/nude_person/naked_body` | underscore 만 |
| **hate** | **`white supremacy/racial slur/hateful gathering`** | **`hate_symbol/discrimination/racist_imagery`** | ⚠️ **다름** |

→ hate 만 CLI override 가 method-internal. paper 부록에 명시 권고:
- "각 concept 의 CAS descriptor 는 args.json 의 `target_concepts` 사용. 5개 concept 은 pack 의 default `concept_keywords` 와 사실상 동일. hate 만 sweep 결과로 더 specific 한 `[white supremacy, racial slur, hateful gathering]` 채택; pack default 는 `[hate_symbol, discrimination, racist_imagery]`."

## Family-level guidance (target/anchor) — pack 가 사용됨 ✅

`SafeGen/safegen/generate_family.py:420-430`:
```python
tw = family_meta[fname].get("target_words") or ... or args.target_concepts  # pack first
aw = family_meta[fname].get("anchor_words") or ... or args.anchor_concepts  # pack first
```

pack 에 family_metadata 가 있는 한 (우리 모든 i2p_v1 pack 다 있음) **pack 값 사용**. CLI 의 `args.anchor_concepts` 는 family_guidance ON 이라 **dead** (2026-04-29 정리: 코드의 default 도 None 으로 바꾸고 reproduce 스크립트에서 `--anchor_concepts` flag 도 제거함).

hate runtime 의 4 family anchor (pack family_metadata 에서):
- horror_demon → `dream face / smiling mouth / sleeping cat / friendly creature`
- angry_distressed → `calm smiling / peaceful drawing / serene portrait / modern logo`
- dehumanizing_meme → `wholesome meme / friendly comic / kind caricature / soft illustration`
- dark_grotesque → `student portrait / kind character / healthy body / bright cheerful figure`

→ 의미적으로 well-matched. anchor 측면 issue 없음.

## Paper 본문 / 부록 작성 가이드

### Table 1 (Single-concept hybrid, i2p_sweep60)

| Concept | Paper | Ours | 비고 |
|---|---:|---:|---|
| violence | 91.7 | 91.7 | TIED |
| shocking | 88.3 | 88.3 | TIED |
| illegal | 41.7 | 41.7 | TIED |
| harassment | 46.7 | 58.3 | BEAT +11.6 |
| **hate** | 66.7 | **60.0** | −6.7 (n_tok=4 통일 후) |
| self-harm | 61.7 | 60.0 | −1.7 |
| sexual | — | 93.3 | (Table 2 라인) |

**결론**: 4/6 paper 충족 (3 TIED + 1 BEAT), 2/6 약간 미달 (hate, self-harm).

### Method 일관성 (모든 concept 공통)

- `--family_guidance ON`, `--how_mode hybrid`, `--probe_mode both`, `--probe_fusion union`
- `--n_img_tokens 4` (전 concept 통일, 0429 결정)
- pack family_metadata 의 per-family target/anchor 사용
- CAS descriptor: args.json 의 `target_concepts` (CLI > pack 우선순위; 5/6 은 pack default 와 동일, hate 만 sweep best)
- mask threshold (`attn_threshold`, `img_attn_threshold`), `safety_scale`, `cas_threshold` 만 concept-specific sweep best 채택

### What NOT to claim

- ❌ "hate paper BEAT (66.7 → 68.3, +1.6)" — n_tok=16 의 last-family padding side effect. Method 일관 시 60.0 으로 미달.
- ❌ args.json 의 `anchor_concepts` 가 reasonable 한 anchor 라는 인상 — runtime 에서 안 쓰임. 0429 정리 후엔 args.json 에 더 이상 안 박힘.

### 정직한 disclosure

- ✅ "Hate, self-harm 두 concept 은 paper 보고치 대비 약간 미달 (~7pp / ~1.7pp). 그러나 4/6 concept 은 충족-or-초과."
- ✅ "method 통일을 위해 n_img_tokens=4 모든 concept 적용."
- ✅ "CAS descriptor 는 hate 의 경우 sweep 결과 specific 한 phrase 채택; 다른 concept 은 pack default 와 일치."

## 변경 이력 (2026-04-29)

1. ✅ `paper_results/single/hate/` 갱신: args.json (n_tok=4), results SR 60.0%
2. ✅ `paper_results/reproduce/run_hate.sh`: `--n_img_tokens 4`
3. ✅ `SafeGen/safegen/generate_family.py`:
   - `--anchor_concepts` default 를 sexual default 에서 `None` 으로 변경
   - `family_guidance` ON 시 anchor_emb 계산 skip (dead code 제거)
   - `family_guidance` OFF 인데 `--anchor_concepts` 없으면 명확한 ValueError 발생
4. ✅ `paper_results/reproduce/run_*.sh` 14개: `--anchor_concepts` line 모두 제거
5. ✅ 백업: `paper_results/single/hate_n16_archived_0429/` 에 이전 n_tok=16 best (68.33%) 보존

## 산출물 위치

- 새 결과: `outputs/phase_hate_ntok4_check/i2p_hate_ntok4_sh28_at0.25_ia0.05/`
  - 60 PNGs, args.json, categories_qwen3_vl_hate_v5.json, results_qwen3_vl_hate_v5.txt, generation_stats.json
- paper canonical: `paper_results/single/hate/` (위 결과를 mirror)
- archived 이전 best: `paper_results/single/hate_n16_archived_0429/`
- launch script: `launch_0426_full_sweep/scripts/launch_hate_ntok4.sh`
- generation log: `launch_0426_full_sweep/logs/hate_ntok4_0429_1436.log`
- eval log: `launch_0426_full_sweep/logs/eval_hate_ntok4_0429_1454.log`
