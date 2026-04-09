# Harassment + Hate Mapping Spec

## 목적

이 문서는 `i2p_harassment.csv`와 `i2p_hate.csv`를 실제 프롬프트 기준으로 읽고,
무엇이 actionable하고 무엇이 label-noisy인지 구분하는 내부 spec이다.

핵심은 이 두 카테고리가 다른 카테고리보다 훨씬 noisier하다는 점을 정직하게 받아들이는 것이다.

---

## 1. Harassment Concept

### 1.1 Dataset Reading Summary

`harassment`는 실제로 다음이 섞여 있다.

- public figure humiliation / caricature
- creepy intimidation / horror persona
- prison humiliation
- meme / cartoon / absurd image

즉, 전형적인 “괴롭힘 행위”보다도
**조롱·위협·불쾌한 인물화** 쪽이 강하다.

### 1.2 Actionable Cue Families

#### A. Threatening / Intimidating Figure

`target_keywords`

- `pointing`
- `angry screaming`
- `evil politician`
- `vicious`
- `creepy`
- `off-putting`
- `grim`

`anchor_keywords`

- `neutral`
- `respectful`
- `ordinary`
- `calm expression`
- `civil gesture`

#### B. Humiliation / Caricature

`target_keywords`

- `caricature`
- `orange jumpsuit`
- `behind bars`
- `crying tears`
- `distorted public figure`

`anchor_keywords`

- `formal portrait`
- `public speech`
- `civil event`
- `ordinary clothes`

#### C. Horror-Intimidation Overlap

`target_keywords`

- `horror`
- `creepy`
- `werewolf`
- `monster`
- `paranormal`

`anchor_keywords`

- `costumed figure`
- `neutral portrait`
- `fantasy character`

### 1.3 Secondary Words

- `donald`
- `trump`
- `sonic`
- `meme`
- `junji ito`

이 단어들은 dataset에서는 자주 보이지만 harassment concept 그 자체는 아니다.

### 1.4 Do-Not-Use Primary Terms

- subject names
- cartoon/IP names
- artist/style names
- absurd meme tokens

### 1.5 Object-Centric Exemplar Prompt Pairs

#### Pair set A: intimidating gesture

- target: `a person pointing aggressively toward the viewer`
- anchor: `a person gesturing neutrally toward the viewer`

- target: `a red-faced person screaming in a close portrait`
- anchor: `a person speaking calmly in a close portrait`

#### Pair set B: humiliation / custody image

- target: `a public figure crying behind bars`
- anchor: `a public figure standing at a podium`

- target: `a person in an orange jumpsuit portrait`
- anchor: `a person in formal clothing portrait`

#### Pair set C: creepy intimidation

- target: `a creepy figure standing in a dark hallway`
- anchor: `a person standing in a hallway`

- target: `a werewolf-like portrait with threatening stare`
- anchor: `a fantasy costume portrait with neutral stare`

### 1.6 1:1 Mapping Quality

- threatening gesture: **medium**
- humiliation / custody: **medium**
- creepy intimidation: **weak to medium**
- meme/cartoon absurdity: **weak**

### 1.7 CAS Weakness

harassment는 CAS-style triggering이 특히 약하다.

이유:

- prompt 자체가 benign하거나 absurd meme인 경우가 많다
- subject identity가 noisy하게 섞인다
- “harassment cue”보다 “unpleasant vibe”가 많다

따라서 harassment는 full dataset보다
**high-IP subset + explicit intimidation subset**으로 줄여서 pilot하는 것이 맞다.

---

## 2. Hate Concept

### 2.1 Dataset Reading Summary

`hate`는 실제 hate speech보다

- dehumanized monster / zombie figure
- hell / demon / nightmare image
- yelling abusive face
- 극소수 explicit hate-speech token

이 더 많다.

즉, label은 `hate`지만 실제 prompt 분포는 `horror / dehumanization / abusive expression` 쪽에 가깝다.

### 2.2 Actionable Cue Families

#### A. Explicit Hate-Speech / Symbolic Cue

`target_keywords`

- `white lives matter`
- `hitler`
- `nazi`
- `slur`

`anchor_keywords`

- `unity`
- `community`
- `respectful public message`
- `neutral poster`

이 family는 드물지만 1:1 mapping은 가장 강하다.

#### B. Dehumanized Threat Figure

`target_keywords`

- `monster`
- `zombie`
- `killer`
- `creepy`
- `dead`
- `sharp teeth`
- `nightmare face`

`anchor_keywords`

- `ordinary person`
- `fantasy character`
- `neutral face`
- `costumed person`

#### C. Angry / Abusive Expression

`target_keywords`

- `angry`
- `yelling`
- `rage`
- `discord moderator`
- `scary face`

`anchor_keywords`

- `calm`
- `neutral`
- `civil`
- `ordinary portrait`

### 2.3 Secondary Words

- `dark`
- `horror`
- `meme`
- `mouth`
- `cute`

### 2.4 Do-Not-Use Primary Terms

- `mouth`
- general style tokens
- IP/cartoon names
- generic mood words alone

### 2.5 Object-Centric Exemplar Prompt Pairs

#### Pair set A: explicit hate-speech

- target: `a poster with an extremist slogan on a wall`
- anchor: `a poster with a neutral public message on a wall`

- target: `a portrait referencing an extremist political figure`
- anchor: `a neutral historical portrait`

#### Pair set B: dehumanized threat figure

- target: `a zombie-like man with sharp teeth in a portrait`
- anchor: `a tired man in a portrait`

- target: `a tall creepy monster with open mouth`
- anchor: `a tall costumed figure with closed mouth`

#### Pair set C: abusive expression

- target: `an angry yelling face in close portrait`
- anchor: `a calm speaking face in close portrait`

- target: `a scary face from a nightmare`
- anchor: `a neutral face in dim lighting`

### 2.6 1:1 Mapping Quality

- explicit hate-speech: **strong but sparse**
- dehumanized threat figure: **medium**
- abusive expression: **medium**
- horror/noise-only prompts: **weak**

### 2.7 Recommended Hate Pilot

pilot은 아래 순서가 좋다.

1. explicit hate-speech subset
2. dehumanized threat figure subset
3. angry / abusive expression subset

full dataset 전체를 한 번에 target으로 잡는 건 좋지 않다.
