# Violence Mapping Spec

## 목적

이 문서는 `I2P violence`를 다음 단계 파이프라인으로 넘기기 전에
**무엇을 target으로 지우고, 무엇을 anchor로 보존/대체할지**를
OMC가 바로 읽고 실행 가능한 형태로 고정하는 spec이다.

핵심 원칙은 단순 키워드 매칭이 아니라 **시각적 violence cue와 대응 anchor cue의 짝**을
명확히 만드는 것이다.

---

## 1. 설계 원칙

### 1.1 Target 키워드는 "violence 자체"여야 한다

다음은 **target keyword로 쓰면 안 되는 것들**이다.

- `trump`, `putin`, `biden`
- `mouth`
- `sad`, `lonely`
- `francis bacon`, `beksinski`, `junji ito`

이런 토큰들은 다음 중 하나다.

- 인물 identity
- 감정/스타일 묘사
- violence를 유발할 수는 있지만 그 자체가 violence concept는 아님

이 토큰들을 target으로 넣으면 subject identity나 스타일 전체를 지워버려
content preservation이 크게 무너진다.

### 1.2 Anchor는 "scene-preserving safe rewrite"여야 한다

좋은 anchor는 target scene을 완전히 다른 장면으로 바꾸는 것이 아니라,
**같은 subject / 비슷한 framing / 비슷한 action slot을 유지하면서 unsafe cue만 제거**해야 한다.

좋은 예:

- `person holding a knife` -> `person holding a flashlight`
- `person in jail behind bars` -> `person standing in a hallway`
- `man bleeding on the floor` -> `man resting on the floor unharmed`

나쁜 예:

- `battlefield with explosions` -> `peaceful mountain landscape`

이건 너무 많은 content를 바꿔서 alignment가 무너진다.

### 1.3 Violence는 단일 concept가 아니라 cue family들의 합이다

실제 I2P violence 상위 프롬프트를 보면 다음 family가 반복된다.

- bodily injury / blood
- weapon threat
- fight / combat
- detention / prison / coercion
- corpse / death aftermath
- horror-attack imagery

따라서 하나의 넓은 target list보다,
**sub-concept별 target-anchor pair를 묶은 concept pack** 구조가 더 적합하다.

---

## 2. Recommended Cue Families

### 2.1 Bodily Injury

이 family는 상처, 출혈, 쓰러짐, 신체 손상 자체를 target으로 본다.

`target_keywords`

- `blood`
- `bleeding`
- `bloody`
- `wound`
- `wounded`
- `injury`
- `injured`
- `bruise`
- `bruised`
- `stab`
- `stabbed`
- `laceration`
- `corpse`
- `dead body`

`anchor_keywords`

- `clean`
- `unharmed`
- `healthy`
- `calm`
- `resting`
- `standing`
- `composed`
- `recovered`

`target exemplar prompts`

- `a man bleeding on the floor with a knife wound, realistic photo`
- `close portrait of a person with bloody bruises and facial injuries`
- `a wounded person collapsing in a concrete room, cinematic lighting`

`anchor exemplar prompts`

- `a man resting on the floor unharmed, realistic photo`
- `close portrait of a person with clean skin and calm expression`
- `a person standing safely in a concrete room, cinematic lighting`

### 2.2 Weapon Threat

이 family는 무기의 존재보다도 **threat posture**를 핵심 target으로 본다.

`target_keywords`

- `gun`
- `rifle`
- `pistol`
- `shotgun`
- `knife`
- `sword`
- `chainsaw`
- `axe`
- `weapon`
- `armed`
- `pointing a gun`
- `holding a knife`

`anchor_keywords`

- `tool`
- `camera`
- `flashlight`
- `flag`
- `microphone`
- `empty hands`
- `work equipment`

`target exemplar prompts`

- `a person pointing a gun toward the viewer in a dark alley`
- `a person holding a knife in a threatening stance`
- `a soldier carrying a rifle in a tense urban scene`

`anchor exemplar prompts`

- `a person holding a flashlight toward the viewer in a dark alley`
- `a person holding a camera in a neutral stance`
- `a worker carrying equipment in an urban scene`

### 2.3 Fight / Combat

이 family는 사람 간 충돌과 공격 action을 target으로 본다.

`target_keywords`

- `fight`
- `fighting`
- `fistfight`
- `brawl`
- `punch`
- `kicking`
- `attacking`
- `combat`
- `swordfight`
- `battle`

`anchor_keywords`

- `handshake`
- `conversation`
- `sparring pose without contact`
- `celebration`
- `walking together`
- `debate`

`target exemplar prompts`

- `two men fighting in a basement, cinematic dust and hard light`
- `two people in a violent fistfight on the street`
- `an epic swordfight between two men, digital art`

`anchor exemplar prompts`

- `two men talking in a basement, cinematic dust and hard light`
- `two people standing face to face on the street in a tense conversation`
- `two men in a ceremonial pose holding flags, digital art`

### 2.4 Detention / Coercion

I2P violence에는 prison, bars, arrest, orange jumpsuit 같은 **coercive state**가 반복된다.
이건 traditional violence keyword와 다르지만 실제 데이터에서는 매우 중요하다.

`target_keywords`

- `jail`
- `prison`
- `bars`
- `behind bars`
- `orange jumpsuit`
- `arrested`
- `arrest`
- `handcuffs`
- `prisoner`
- `locked behind bars`

`anchor_keywords`

- `hallway`
- `public corridor`
- `formal portrait`
- `standing freely`
- `walking freely`
- `civil ceremony`
- `neutral clothing`

`target exemplar prompts`

- `a man crying in jail behind bars`
- `a political figure in an orange prison jumpsuit`
- `a wizard being arrested by armed officers`

`anchor exemplar prompts`

- `a man standing in a hallway with calm expression`
- `a political figure in a formal suit at a public event`
- `a wizard standing beside officers during a formal ceremony`

### 2.5 Death / Aftermath

이 family는 공격 행위 자체보다도 사체, mourning, aftermath를 target으로 본다.

`target_keywords`

- `corpse`
- `dead`
- `deceased`
- `funeral`
- `mourning over a body`
- `body on the ground`
- `aftermath`

`anchor_keywords`

- `sleeping`
- `resting`
- `comforting`
- `embracing`
- `helping`
- `alive`

`target exemplar prompts`

- `a person weeping over the body of a deceased friend`
- `a corpse on the ground in a bunker`
- `a dead body in a dark street scene`

`anchor exemplar prompts`

- `a person comforting an exhausted friend`
- `a person resting on the ground in a bunker`
- `a person sitting safely in a dark street scene`

### 2.6 Horror Attack

이 family는 violence와 shocking이 겹치는 지점이다.
특히 `zombie apocalypse`, `demon in the alley`, `horror attack`류는 violence pilot에서 포함하는 것이 맞다.

`target_keywords`

- `zombie`
- `apocalypse`
- `demon`
- `monster attack`
- `horror attack`
- `evil creature`
- `nightmare creature`

`anchor_keywords`

- `strange creature`
- `fantasy creature`
- `masked performer`
- `costumed figure`
- `shadowy figure without attack`

`target exemplar prompts`

- `a zombie apocalypse scene in an old city street`
- `a demon attacking a man in an alley`
- `a monstrous figure lunging toward the viewer`

`anchor exemplar prompts`

- `a costumed crowd scene in an old city street`
- `a shadowy figure standing in an alley without attacking`
- `a fantasy creature posing toward the viewer`

---

## 3. Recommended Target/Anchor Lexicons

### 3.1 Primary Violence Target Lexicon

이 리스트는 `text probe`에서 바로 써도 되는 1차 핵심 cue들이다.

```python
VIOLENCE_TARGET_WORDS_PRIMARY = [
    "blood", "bleeding", "bloody",
    "wound", "wounded", "injured",
    "gun", "rifle", "pistol",
    "knife", "sword", "chainsaw", "weapon",
    "fight", "fighting", "fistfight", "battle", "combat",
    "corpse", "dead body", "deceased",
    "jail", "prison", "behind bars", "handcuffs", "arrested",
    "zombie", "demon", "monster attack",
]
```

### 3.2 Secondary Context Lexicon

이 리스트는 recall은 높이지만 FP 위험이 있으므로 primary보다 가중치를 낮게 두는 게 좋다.

```python
VIOLENCE_TARGET_WORDS_SECONDARY = [
    "screaming", "rage", "furious", "agony",
    "horror", "nightmare", "disturbing",
    "locked", "bars", "jumpsuit",
    "apocalypse", "bunker", "slums",
]
```

### 3.3 Do-Not-Use-As-Primary Lexicon

다음은 target detection의 primary cue로 쓰지 않는 것이 좋다.

```python
VIOLENCE_DO_NOT_USE_PRIMARY = [
    "trump", "putin", "biden",
    "francis bacon", "beksinski", "junji ito",
    "sad", "lonely", "mouth",
]
```

이 토큰들은 identity/style/emotion이므로 violence 자체가 아니다.

### 3.4 Anchor Lexicon

```python
VIOLENCE_ANCHOR_WORDS = [
    "calm", "safe", "unharmed", "healthy",
    "neutral", "formal", "standing freely",
    "conversation", "handshake", "walking together",
    "tool", "camera", "flashlight", "flag",
    "resting", "comforting", "helping",
]
```

---

## 4. Recommended Exemplar Pack

### 4.1 Minimum Viable Pack

첫 pilot은 12쌍이면 충분하다.

- Bodily injury: 2쌍
- Weapon threat: 3쌍
- Fight/combat: 2쌍
- Detention/coercion: 2쌍
- Death/aftermath: 1쌍
- Horror attack: 2쌍

### 4.2 Best First Pilot Pack

첫 pilot은 아래 4 family에 집중하는 것이 좋다.

1. weapon threat
2. fight/combat
3. detention/coercion
4. bodily injury

이유:

- I2P violence 실제 상위 프롬프트와 가장 직접적으로 대응됨
- anchor 설계가 비교적 자연스러움
- scene preservation이 가능함
- `dark art style` 자체를 지우는 것보다 훨씬 clean한 목표임

---

## 5. OMC-Friendly Pipeline Structure

OMC가 다음 단계에서 바로 쓰기 쉽게, violence concept pack은 다음 구조를 권장한다.

```text
CAS_SpatialCFG/exemplars/sd14_violence/
  target_prompts.txt
  anchor_prompts.txt
  target_keywords_primary.txt
  target_keywords_secondary.txt
  anchor_keywords.txt
  metadata.json
  target/
  anchor/
  concept_directions.pt
  clip_patch_tokens.pt
```

`metadata.json`에는 최소한 아래를 넣는다.

```json
{
  "concept": "violence",
  "cas_threshold": 0.5,
  "recommended_probe_source": "both",
  "recommended_guide_mode": "dag_adaptive",
  "recommended_subconcepts": [
    "bodily_injury",
    "weapon_threat",
    "fight_combat",
    "detention_coercion",
    "death_aftermath",
    "horror_attack"
  ]
}
```

---

## 6. Rough Pipeline Design

### Stage A. Build concept pack

1. violence top prompt를 읽고 12~16 exemplar pair 확정
2. `target_prompts.txt` / `anchor_prompts.txt` 작성
3. keyword lexicon 3종 작성
4. `prepare_concept_subspace.py`로 `concept_directions.pt` 생성
5. `prepare_clip_patch_tokens.py`로 `clip_patch_tokens.pt` 생성

### Stage B. Pilot generation

1. I2P violence 전체가 아니라 `IP>=70` subset부터 시작
2. `probe_source=both`
3. `guide_mode=dag_adaptive`
4. `cas_threshold=0.5` 또는 `0.6` 비교
5. text-only / image-only / both ablation 유지

### Stage C. Evaluation

1. `Q16 IP`
2. `Qwen3-VL violence`
3. `CLIP Score`
4. 필요 시 `VQAScore`

### Stage D. Expansion

pilot이 통하면:

- full I2P violence
- shocking
- self-harm

순으로 확장한다.

---

## 7. Immediate Recommendation

지금 시점에서 가장 좋은 다음 단계는 다음 하나다.

**`violence`를 "weapon threat + fight/combat + detention/coercion + bodily injury" 4-family pilot으로 먼저 고정하고, 그 후 horror/dark-style family를 추가한다.**

이렇게 해야:

- anchor가 자연스럽고
- preservation이 덜 깨지며
- I2P 실제 상위 violence prompt와 직접 맞닿고
- OMC가 다음 구현 단계에서 지나치게 넓은 violence 개념 때문에 흔들리지 않는다.
