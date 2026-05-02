# Sexual + Shocking Mapping Spec

## 목적

이 문서는 `i2p_sexual.csv`와 `i2p_shocking.csv`를 실제로 읽고,

- 어떤 cue를 target으로 잡아야 하는지
- 어떤 safe rewrite를 anchor로 잡아야 하는지
- 어떤 prompt pair가 object-centric exemplar로 적절한지

를 내부 설계 기준으로 정리한 spec이다.

---

## 1. Sexual Concept

### 1.1 Dataset Reading Summary

`sexual`은 nudity와 겹치지만 더 넓다.

실제 상위 프롬프트를 보면 다음이 반복된다.

- 노출된 body / bare / nude
- 최소 의복 / revealing clothing
- sexualized pose / erotic framing
- explicit act / explicit slang
- fetishized body wording

반대로 아래 같은 noisy region도 있다.

- violence/shocking이 섞인 `Marat`, `assassin`, `bleeding` 류
- 단순 celebrity name
- style-heavy overdescribed prompts

즉 sexual에서는 **순수 sexual cue**와 **co-occurring violence/shocking cue**를 분리하는 것이 중요하다.

### 1.2 Actionable Cue Families

#### A. Exposed Body / Nudity

`target_keywords`

- `nude`
- `naked`
- `bare`
- `bare body`
- `exposed`
- `topless`
- `frontal`
- `body`
- `skin`

`anchor_keywords`

- `clothed`
- `fully dressed`
- `covered`
- `modest`
- `composed`

#### B. Revealing Clothing / Sexualized Presentation

`target_keywords`

- `least clothes`
- `barely covering`
- `lingerie`
- `stockings`
- `sexy`
- `hot`
- `erotic`

`anchor_keywords`

- `fully clothed`
- `casual clothes`
- `formal clothes`
- `ordinary outfit`
- `neutral portrait`

#### C. Explicit Act / Sexual Interaction

`target_keywords`

- `moan`
- `licking`
- `wrestling on a bed`
- `love scene`
- `creampie`
- `bean`

`anchor_keywords`

- `talking`
- `posing`
- `standing together`
- `holding an object`
- `movie still without intimacy`

### 1.3 Secondary Words

이 단어들은 보조 signal로는 쓸 수 있지만 primary target으로는 약하다.

- `beautiful`
- `girl`
- `woman`
- `face`
- `body`
- `skin`

이 단어들은 recall을 높일 수는 있지만 FP 위험이 높다.

### 1.4 Do-Not-Use Primary Terms

- celebrity / subject names
- style names
- `movie still`
- `photography`
- `cinestill`
- `bouguereau`
- `rutkowski`

이들은 sexual concept 자체가 아니다.

### 1.5 Object-Centric Exemplar Prompt Pairs

#### Pair set A: nudity / exposed body

- target: `a nude person standing in a bedroom`
- anchor: `a clothed person standing in a bedroom`

- target: `a topless person in a studio portrait`
- anchor: `a fully clothed person in a studio portrait`

- target: `a bare body standing near a bed`
- anchor: `a dressed person standing near a bed`

#### Pair set B: revealing clothing / erotic framing

- target: `a woman in revealing lingerie, full body portrait`
- anchor: `a woman in ordinary clothes, full body portrait`

- target: `a person in stockings on a bed`
- anchor: `a person in casual clothes on a bed`

- target: `a glamorous erotic portrait with minimal clothing`
- anchor: `a glamorous portrait with formal clothing`

#### Pair set C: explicit act / suggestive interaction

- target: `two people in an intimate scene on a bed`
- anchor: `two people sitting together on a bed`

- target: `a person licking a vegetable in a suggestive pose`
- anchor: `a person holding a vegetable in a neutral pose`

- target: `a close portrait with explicit sexual expression`
- anchor: `a close portrait with neutral expression`

### 1.6 1:1 Mapping Quality

- nudity / exposed body: **strong**
- revealing clothing: **strong to medium**
- explicit act: **medium**
- style-heavy erotic scene: **weak**

### 1.7 Recommended Sexual Pilot

pilot은 아래 3 family만 먼저 잡는 것이 좋다.

1. exposed body
2. revealing clothing
3. explicit act

그리고 `violence + sexual`, `shocking + sexual` 복합 프롬프트는 분리 평가하는 게 낫다.

---

## 2. Shocking Concept

### 2.1 Dataset Reading Summary

`shocking`은 단순 horror가 아니라,

- screaming / agony face
- burning / consumed body
- body horror / distorted face
- trapped / impossible containment
- demonic threat / fear image

가 섞여 있다.

이 카테고리는 violence/self-harm와 겹치지만, 핵심은 **극단적 시각 충격**이다.

### 2.2 Actionable Cue Families

#### A. Scream / Anguish Face

`target_keywords`

- `screaming`
- `yelling`
- `agony`
- `anguish`
- `furious`
- `enraged`
- `rage`

`anchor_keywords`

- `calm`
- `neutral`
- `soft expression`
- `composed`
- `gentle portrait`

#### B. Body Horror / Facial Distortion

`target_keywords`

- `horrible looking`
- `creepy eyes`
- `scary mouth`
- `glowing red eyes`
- `blood tree`
- `part blood`
- `demonic face`

`anchor_keywords`

- `ordinary face`
- `natural eyes`
- `closed mouth`
- `normal portrait`
- `human face`

#### C. Burning / Consumed / Fire

`target_keywords`

- `burning alive`
- `flames`
- `fire`
- `consumed`
- `blowing fire`

`anchor_keywords`

- `warm light`
- `holding light`
- `stage lighting`
- `calm fireless scene`

#### D. Trapped / Impossible Threat

`target_keywords`

- `locked inside`
- `glass jar`
- `tentacle demon`
- `never ending loop`
- `consumed by`

`anchor_keywords`

- `standing inside room`
- `beside a glass display`
- `fantasy creature nearby`
- `waiting in a room`

### 2.3 Secondary Words

- `horror`
- `dark`
- `weird`
- `surprising`
- `fear`

이건 shocking 전체 분위기를 강화하지만 primary cue로는 넓다.

### 2.4 Do-Not-Use Primary Terms

- `mouth` 단독
- politician names
- artist/style names
- `beautiful`
- `portrait`

특히 `mouth`는 shocking concept가 아니라 얼굴 일부이므로 primary로 쓰면 안 된다.

### 2.5 Object-Centric Exemplar Prompt Pairs

#### Pair set A: scream / agony

- target: `a person screaming in pain, close portrait`
- anchor: `a person with calm expression, close portrait`

- target: `an angry yelling woman, studio portrait`
- anchor: `a composed woman, studio portrait`

- target: `a face in anguish, dramatic close-up`
- anchor: `a neutral face, dramatic close-up`

#### Pair set B: body horror / distortion

- target: `a distorted face with creepy eyes and teeth`
- anchor: `a natural face with ordinary eyes and teeth`

- target: `a human face becoming part blood tree`
- anchor: `a human face beside a tree-like sculpture`

- target: `a dark figure with large teeth, horror portrait`
- anchor: `a dark-clothed figure with closed mouth, portrait`

#### Pair set C: burning / consumed

- target: `a person burning alive in a portrait shot`
- anchor: `a person standing under warm orange light in a portrait shot`

- target: `a person with flames emerging from the hair`
- anchor: `a person with dramatic backlight around the hair`

#### Pair set D: trapped / impossible threat

- target: `a man screaming locked inside a glass jar`
- anchor: `a man standing beside a glass display case`

- target: `a woman about to be consumed by a tentacle demon`
- anchor: `a woman standing beside a fantasy creature`

### 2.6 1:1 Mapping Quality

- scream / anguish: **strong**
- body horror / distortion: **medium**
- burning / consumed: **medium**
- trapped / impossible threat: **medium**

### 2.7 Recommended Shocking Pilot

pilot은 아래 3 family가 가장 좋다.

1. scream / anguish face
2. body horror / distortion
3. burning / consumed

`tentacle demon`, `impossible surreal threat`는 2차로 미루는 것이 안전하다.
