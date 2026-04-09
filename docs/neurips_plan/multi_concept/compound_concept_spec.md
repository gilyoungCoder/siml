# Compound (Compositional) Concept Erasing Spec

> 미팅 2026-04-04 논의 기반 (조규상 × 길영환)

## 용어 정의

| 용어 | 의미 | 예시 | 처리 방식 |
|------|------|------|----------|
| **Single concept** | 하나의 unsafe concept 제거 | "nude" | 현재 v14-v19 |
| **Multi-concept** | 서로 다른 unsafe concept 동시 제거 | "nude" + "violence" | concept_packs 복수 로딩 |
| **Compound concept** | harmful attribute + 추가 속성 결합 | "young + nude", "child + violence" | **이 문서의 대상** |

## 핵심 인사이트 (미팅 기반)

1. **Compound concept은 multi-concept보다 쉬울 수 있다**
   - Multi-concept: 독립적인 guidance 방향 간 conflict 발생 → semantic 붕괴
   - Compound: 하나의 combined negative concept → 단일 guidance 방향 → conflict 적음
   
2. **Compound를 하나의 negative concept으로 처리하면 단일 concept과 크게 다르지 않다**
   - "young + nude" → `d_target = eps("young nude person") - eps("")`
   - 기존 CAS + spatial mask 파이프라인 그대로 사용 가능

3. **하지만 generic 확장에는 한계**
   - Compound를 계속 추가하면 guidance 누적 → semantic 붕괴
   - 파라미터 세팅에 민감 (inference-time 방법론의 고유 한계)

4. **Jailbreak 관점에서**: harmful concept이 누적될수록 suppression 효과 강해짐
   - "nude" alone → 어느정도 생성됨
   - "young + nude" → extremely harmful → 모델 자체가 강하게 억제
   - 우리 방법론이 추가로 개입할 여지가 오히려 적을 수 있음

## 구현 접근 방식

### Approach A: Compound를 단일 concept으로 처리 (권장)

```python
# "young + nude"를 하나의 concept direction으로
d_target = eps("young nude person") - eps("")
d_anchor = eps("young clothed person") - eps("")

# CAS, spatial mask, guidance 모두 기존과 동일
# concept_directions.pt를 compound용으로 별도 생성하면 끝
```

**장점**: 기존 코드 수정 0. concept_directions.pt만 새로 만들면 됨.
**단점**: Compound별로 별도 exemplar 이미지 + concept direction 필요

### Approach B: Attribute-Augmented Concept Direction

```python
# Base concept direction + attribute modifier
d_base = eps("nude person") - eps("")           # 기존 nudity direction
d_attr = eps("young person") - eps("person")    # age attribute direction
d_compound = normalize(d_base + alpha * d_attr)  # compound = base + weighted attribute

# alpha 조절로 attribute 강도 제어
# alpha=0 → 기존 nudity erasing
# alpha=1 → young+nude compound erasing
```

**장점**: 새 compound마다 exemplar 이미지 안 만들어도 됨. Attribute direction만 추가.
**단점**: direction 합산이 실제로 compound를 잘 표현하는지 보장 없음.

### Approach C: Compound Exemplar Pack

```python
# 기존 concept_pack과 동일한 구조, 단 exemplar prompt가 compound
concept_packs/compound_young_nude/
  target_prompts.txt:  "young nude person standing...", "teenage body exposed..."
  anchor_prompts.txt:  "young clothed person standing...", "teenage person in school uniform..."
  metadata.json:       {"compound_of": ["sexual", "young"], "cas_threshold": 0.6}
```

**장점**: 가장 정확. Compound의 시각적 패턴을 직접 학습.
**단점**: Compound 조합마다 새 pack 필요 (조합 폭발).

### 권장: Approach A (단기) + Approach B (중기)

**단기 (NeurIPS)**: Approach A로 주요 compound 몇 개만 테스트
- young + nude
- weapon + violence  
- drug + self-harm

**중기**: Approach B로 attribute direction을 분리하여 임의 조합 가능하게

## Compound Concept 목록 (우선순위)

### Tier 1: 가장 중요 (법적/윤리적으로 extremely harmful)

| Compound | Base Concept | Attribute | 위험도 | 비고 |
|----------|-------------|-----------|--------|------|
| **young + nude** | nudity | age (young) | Extreme | CSAM 관련, 모델이 가장 강하게 억제 |
| **child + violence** | violence | age (child) | Extreme | 아동 대상 폭력 |
| **real person + nude** | nudity | identity (celebrity) | High | deepfake 관련 |

### Tier 2: 중요 (사회적으로 sensitive)

| Compound | Base Concept | Attribute | 위험도 |
|----------|-------------|-----------|--------|
| **ethnic + violence** | violence | ethnicity | High |
| **religious + hate** | hate | religion | High |
| **gender + harassment** | harassment | gender | High |
| **elderly + abuse** | violence | age (old) | Medium |

### Tier 3: 일반적 (맥락에 따라 unsafe)

| Compound | Base Concept | Attribute |
|----------|-------------|-----------|
| weapon + school | violence | location |
| drug + teenager | illegal | age |
| gore + realistic | shocking | style |

## 구현 계획

### Phase 1: Compound Concept Pack 생성 (Approach A)

Tier 1의 3개 compound에 대해 concept pack 생성:

#### 1. young_nude
```
concept_packs/compound_young_nude/
  target_prompts.txt:
    "young person standing nude in a room, realistic"
    "teenage body without clothes, photograph"
    "underage looking person in state of undress"
    ... (16 prompts)
  anchor_prompts.txt:
    "young person standing in school uniform in a room, realistic"
    "teenage person in casual clothes, photograph"
    "young looking person in proper attire"
    ... (16 matching prompts)
  target_keywords_primary.txt:
    young nude
    child naked
    teen undress
    underage bare
  metadata.json:
    {"name": "young_nude", "compound_of": ["sexual","young"],
     "cas_threshold": 0.55, "probe_source": "both"}
```

#### 2. weapon_violence
```
target: "person holding gun threatening another person"
anchor: "person holding camera photographing another person"
```

#### 3. real_person_nude (deepfake prevention)
```
target: "celebrity name nude photorealistic"
anchor: "celebrity name portrait formal clothes photorealistic"
```

### Phase 2: Attribute Direction Library (Approach B)

```python
# attribute_directions.py
ATTRIBUTES = {
    "young": {
        "positive": "young person, child, teenager, underage",
        "negative": "adult person, mature, elderly",
    },
    "realistic": {
        "positive": "photorealistic, photograph, real, hd photo",
        "negative": "painting, cartoon, drawing, illustration",
    },
    "celebrity": {
        "positive": "famous person, celebrity, well-known figure",
        "negative": "anonymous person, unknown individual",
    },
    "ethnic": {
        "positive": "specific ethnicity, racial features",
        "negative": "generic person, neutral appearance",
    },
}

def compute_attribute_direction(text_encoder, tokenizer, attr_name, device):
    """Compute d_attr = encode(positive) - encode(negative)"""
    attr = ATTRIBUTES[attr_name]
    pos_embed = encode_concepts(text_encoder, tokenizer, attr["positive"].split(", "), device)
    neg_embed = encode_concepts(text_encoder, tokenizer, attr["negative"].split(", "), device)
    return pos_embed - neg_embed  # attribute direction

def compound_direction(d_base, d_attr, alpha=0.5):
    """Combine base concept direction with attribute direction."""
    d_compound = d_base + alpha * d_attr
    return d_compound / d_compound.norm(dim=-1, keepdim=True)
```

### Phase 3: Compound + Multi-Concept 통합

최종 목표: 여러 compound concept을 동시에 제거

```python
# 예: "young nude" + "weapon violence" 동시 제거
compounds = [
    {"base": "sexual", "attr": "young", "alpha": 0.7},
    {"base": "violence", "attr": "weapon", "alpha": 0.5},
]

# Denoising loop:
total_correction = 0
for comp in compounds:
    d_compound = compound_direction(d_base[comp["base"]], d_attr[comp["attr"]], comp["alpha"])
    cas_val = cos_sim(d_prompt, d_compound)
    if cas_val > threshold:
        mask = compute_mask(...)
        correction = dag_adaptive(mask, d_compound)
        total_correction += correction

eps_final = eps_cfg - total_correction
```

## Guidance Conflict 문제와 해결

### 문제 (미팅에서 논의)
- 여러 concept/compound를 동시에 지우면 guidance 방향 간 conflict
- Semantic 붕괴: 이미지가 깨지거나 원래 의도와 완전히 다른 결과

### 해결 방안

1. **Correction Capping**: 총 correction magnitude에 상한선
   ```python
   total_correction = clamp(total_correction, max=max_correction_norm)
   ```

2. **Conflict Detection**: 두 correction 방향의 cosine similarity가 음수면 conflict
   ```python
   if cos_sim(corr_1, corr_2) < -0.3:
       # conflict detected — use the stronger one only
       total_correction = corr_1 if norm(corr_1) > norm(corr_2) else corr_2
   ```

3. **Priority-Based Erasing**: concept별 우선순위 → 충돌 시 high-priority만 적용
   ```python
   priorities = {"young_nude": 1, "violence": 2, "harassment": 3}
   # Sort by priority, apply sequentially, skip if conflict with higher priority
   ```

4. **Adaptive Scale Reduction**: concept 수에 따라 개별 guidance scale 감소
   ```python
   scale_per_concept = base_scale / sqrt(num_active_concepts)
   ```

## 평가 계획

### Compound Concept 평가 데이터셋

기존 I2P에는 compound-specific 데이터가 없음 → 직접 구축 필요:

1. **young+nude**: uncensored LLM으로 compound prompt 생성 (조규상님 방식 참고)
2. **weapon+violence**: I2P violence에서 weapon 키워드 있는 subset
3. **Metric**: Q16 IP + Qwen3-VL + CLIP Score (기존과 동일)

### Cross-Concept Interference 테스트

Compound를 지울 때 관련 없는 concept에 영향이 없는지:
- young+nude를 지우면서 "adult nude" erasing 성능이 유지되는지
- weapon+violence를 지우면서 "tool holding" 이미지가 망가지지 않는지

## 논문에서의 위치

### Related Work에서 차별화
- 기존 unlearning: single concept only (ESD, SalUn, etc.)
- DAG: single concept with spatial precision
- SAFREE: single concept, text-space only
- **Ours**: single + multi + compound concept, training-free, image+text exemplar

### Table 구조

```
Table X: Compound Concept Erasing
| Method | young+nude | weapon+violence | celebrity+nude | Avg IP↓ | CLIP↑ |
|--------|-----------|----------------|---------------|---------|-------|
| SD v1.4 | ... | ... | ... | ... | ... |
| SLD | ... | ... | ... | ... | ... |
| Ours (single) | ... | ... | ... | ... | ... |
| Ours (compound) | ... | ... | ... | ... | ... |
```

## 참고 논문 (미팅에서 공유)

- RPG-RT (arxiv 2505.21074): Rule-based Preference Modeling으로 T2I red-teaming
- JPA (arxiv 2404.02928): Jailbreaking Prompt Attack, text embedding space에서 NSFW concept 탐색
- IVO (arxiv 2602.00175): Initial latent variable optimization으로 unlearning 우회

→ 이 논문들은 **공격(jailbreak) 관점**이지만, 우리 **방어(defense)** 방법의 robustness를 평가하는 데 활용 가능. 특히 IVO는 unlearning 기반 방어를 우회하는 방법이므로, "우리 inference-time 방법은 모델 weight을 건드리지 않아 IVO 공격에 면역"이라는 argument 가능.
