# Multi-Concept Integration Spec

## 목적

이 문서는 각 concept별 target-anchor mapping spec을
기존 `v3`, `v4`, `v14~v19` 구조에 어떻게 연결할지 정리한 내부 설계 문서다.

핵심은 `nude -> clothed`처럼 가능한 한 **concept family 단위 1:1 target-anchor pair**를 만들고,
그 pair를 묶은 `concept pack`을 각 버전이 읽도록 하는 것이다.

---

## 1. Core Representation

각 concept는 하나의 넓은 label이 아니라 여러 `family`의 묶음으로 본다.

예:

- violence
  - bodily injury
  - weapon threat
  - detention/coercion
- sexual
  - exposed body
  - revealing clothing
  - explicit act

각 family는 아래를 가진다.

- `target_words_primary`
- `target_words_secondary`
- `anchor_words`
- `target_exemplar_prompts`
- `anchor_exemplar_prompts`
- `mapping_strength`

이걸 concept pack으로 저장한다.

---

## 2. Concept Pack Format

```text
concept_packs/<concept>/
  metadata.json
  families.json
  target_prompts.txt
  anchor_prompts.txt
  target_keywords_primary.txt
  target_keywords_secondary.txt
  anchor_keywords.txt
  concept_directions.pt
  clip_patch_tokens.pt
```

`families.json`은 family별 1:1 mapping을 담는다.

```json
{
  "concept": "violence",
  "families": [
    {
      "name": "weapon_threat",
      "mapping_strength": "strong",
      "target_words_primary": ["gun", "knife", "rifle"],
      "anchor_words": ["flashlight", "camera", "tool"]
    }
  ]
}
```

---

## 3. Target-Anchor 1:1 Mapping Rule

### 3.1 Concept-level이 아니라 family-level로 1:1을 만든다

`violence -> peace`처럼 너무 넓게 잡으면 drift가 커진다.

대신:

- `knife -> flashlight`
- `prison bars -> hallway railing`
- `bloody wound -> clean skin`

처럼 **family-level local rewrite**를 만든다.

### 3.2 Anchor-push는 family별로만 쓴다

anchor 쪽으로 민다고 해도
전체 concept를 한 번에 anchor 하나로 밀면 안 된다.

좋은 방식:

- violence/weapon family -> neutral object-holding anchor
- violence/detention family -> free-standing anchor
- sexual/exposed-body family -> clothed-body anchor

즉 anchor는 concept 전체가 아니라 **family마다 따로** 있다.

---

## 4. Existing Version Mapping

## 4.1 v3

`v3`는 `CAS + cross-attn + dag_adaptive` 구조다.

멀티컨셉에서는 concept별로 아래를 반복하면 된다.

1. concept family별 `d_target`
2. concept family별 `d_anchor`
3. concept family별 `CAS`
4. concept family별 `mask`
5. correction 합산

즉 `v3`는 **multi-family additive correction**으로 확장하기 가장 쉽다.

## 4.2 v4

`v4`는 noise spatial CAS 기반 WHERE가 강하다.

멀티컨셉에서는

- concept family별 global CAS
- family별 spatial CAS
- family별 soft mask

를 만들고 correction을 합산하면 된다.

단점:

- text keyword 의존성이 큼
- harassment/hate처럼 implicit한 카테고리엔 약함

## 4.3 v14

`v14`는 `cross-attn probe × noise CAS fusion`이라
가장 자연스러운 multi-concept base 중 하나다.

family별로

- `mask_attn_family`
- `mask_noise_family`
- `mask_fused_family`

를 만든 뒤,
`dag_adaptive` correction을 합산하면 된다.

즉 `v14`는 **family-level fused WHERE**의 기본형으로 적합하다.

## 4.4 v15

`v15`는 CLIP patch token probe가 핵심이다.

이건 text가 약한 category에서 가장 중요하다.

적합한 카테고리:

- violence
- shocking
- self-harm 일부

약한 카테고리:

- harassment/hate full dataset

`v15`는 image-heavy family에 특히 강하다.

## 4.5 v16

`v16`의 contrastive direction은
target-anchor가 비교적 깔끔한 family에서 가장 잘 맞는다.

예:

- exposed body -> clothed body
- weapon threat -> neutral object holding
- mugshot -> studio portrait

즉 `mapping_strength=strong` family 우선 적용이 맞다.

## 4.6 v17

`v17`의 projection 기반 접근은 concept pack interface는 공유할 수 있지만,
projection 품질에 더 민감하다.

따라서:

- primary family만 먼저
- weak/noisy family는 제외

전략이 좋다.

## 4.7 v18

`v18`은 timestep-adaptive schedule이다.

멀티컨셉에선 매우 유리하다.

- early step: broad but safe family 먼저
- late step: local family correction

특히

- sexual / violence처럼 object-local한 family
- shocking처럼 face-local한 family

에 유리하다.

## 4.8 v19

`v19`는 multi-exemplar 구조라
멀티컨셉과 가장 직접적으로 이어진다.

다만 여기서도 concept-level exemplar를 한꺼번에 합치기보다,
**family-level exemplar bank**를 두는 것이 좋다.

---

## 5. Multi-Concept Correction Rule

매 denoising step에서:

1. concept별로 family loop
2. family별 CAS gate
3. family별 mask 계산
4. family별 `target-anchor` correction 계산
5. 전체 correction 합산

```python
total_correction = 0
for concept in concepts:
    for family in concept.families:
        if cas_family > threshold_family:
            corr = correction_family(mask_family, d_target_family, d_anchor_family)
            total_correction += corr
eps_safe = eps_cfg - total_correction
```

핵심은 concept 하나당 correction 하나가 아니라
**family 하나당 correction 하나**라는 점이다.

---

## 6. Prompt Generality Rule

exemplar prompt는 object-centric하고 짧아야 한다.

좋은 예:

- `a person holding a gun in an alley`
- `a person holding a flashlight in an alley`

나쁜 예:

- 화가 이름, 렌즈, 색감, 장식 수식어가 과도하게 붙은 prompt

이유:

- concept pack의 일반화가 깨진다
- image probe가 style에 과적합된다
- family-level mapping이 흐려진다

---

## 7. When High-IP Subset First

아래 카테고리는 full dataset 전체보다 high-IP subset first가 맞다.

- harassment
- hate
- illegal_activity 일부

이유:

- label noise
- benign-looking prompts
- weak CAS trigger

반대로 아래는 full dataset 또는 넓은 subset이 비교적 가능하다.

- sexual
- violence
- shocking

---

## 8. Immediate Recommendation

실제 구현 순서는 다음이 가장 좋다.

1. `violence`
2. `sexual`
3. `shocking`
4. `self-harm`
5. `illegal_activity`
6. `harassment`
7. `hate`

그리고 각 concept는 다음 순서로 들어가야 한다.

1. `strong 1:1 family`
2. `medium 1:1 family`
3. `weak/noisy family`

즉 multi-concept 확장은
`concept-by-concept`가 아니라
**`family-by-family`, strong-to-weak 순서**로 가는 것이 맞다.
