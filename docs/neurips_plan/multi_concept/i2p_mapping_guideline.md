# I2P Mapping Guideline

## 목적

이 문서는 I2P 각 카테고리에서

- target keyword
- anchor keyword
- target exemplar prompt
- anchor exemplar prompt

를 뽑을 때 따라야 하는 내부 지침서다.

이 문서는 공개 논문용 설명이 아니라 OMC/실험 설계용 내부 기준이다.

---

## 1. 기본 원칙

### 1.1 라벨이 아니라 실제 프롬프트를 읽고 뽑는다

I2P는 category label만 보고 target을 정하면 안 된다.

반드시 실제 prompt를 읽고 다음을 구분해야 한다.

- concept 자체
- identity/person name
- art style
- emotion/atmosphere
- 배경 scene

target은 이 중 **concept 자체**여야 한다.

### 1.2 target keyword는 시각적으로 지워야 할 cue여야 한다

좋은 target keyword:

- `gun`
- `knife`
- `blood`
- `behind bars`

나쁜 target keyword:

- `trump`
- `putin`
- `beksinski`
- `sad`

좋은 target은 unsafe cue를 직접 가리킨다.
나쁜 target은 subject identity, style, mood를 지워버린다.

### 1.3 anchor는 scene-preserving rewrite여야 한다

anchor는 완전히 다른 장면으로 도망가면 안 된다.

좋은 anchor:

- `person holding a knife` -> `person holding a flashlight`
- `prisoner behind bars` -> `person standing in a hallway`

나쁜 anchor:

- `person holding a knife` -> `mountain landscape`

좋은 anchor는 다음을 최대한 유지해야 한다.

- subject count
- framing
- pose slot
- object slot
- local scene type

### 1.4 exemplar prompt는 구체적이되 과도하게 장식적이면 안 된다

우리가 원하는 건 object-centric general prompt다.

좋은 예:

- `a person holding a gun in a dark alley`
- `a person holding a flashlight in a dark alley`

나쁜 예:

- 스타일 이름, 렌즈, 화가, 색감, 수십 개 수식어가 붙은 prompt

실험용 exemplar는 가능한 한

- 짧고
- 재현 가능하며
- unsafe cue / safe cue가 명확해야 한다.

---

## 2. 추출 절차

각 concept에 대해 아래 순서로 본다.

1. 상위 `inappropriate_percentage` prompt를 먼저 읽는다.
2. 반복되는 시각 cue family를 찾는다.
3. family별 target cue를 적는다.
4. family별 anchor cue를 적는다.
5. primary / secondary / do-not-use 단어를 분리한다.
6. object-centric exemplar prompt pair를 만든다.
7. `1:1 target-anchor mapping quality`를 평가한다.

---

## 3. 단어 분류 규칙

### 3.1 Primary target words

다음을 만족해야 한다.

- unsafe cue를 직접 가리킴
- 이미지에서 국소적으로 찾을 수 있음
- anchor로 치환 가능함

### 3.2 Secondary target words

다음을 만족한다.

- recall은 높여주지만 모호함
- prompt trigger 보조용으로만 적합

예:

- `horror`
- `rage`
- `nightmare`

### 3.3 Do-not-use primary terms

다음은 primary target으로 금지한다.

- 인물 이름
- 정치인 이름
- style token
- artist token
- broad mood token

예:

- `trump`
- `hitler`
- `junji ito`
- `francis bacon`
- `sad`

---

## 4. 1:1 Mapping Quality

각 concept family는 아래 셋 중 하나로 분류한다.

### 4.1 Strong 1:1

unsafe cue와 safe cue가 직접 대응된다.

예:

- `gun` -> `flashlight`
- `bloody wound` -> `clean skin`

### 4.2 Medium 1:1

대응은 가능하지만 scene drift 위험이 있다.

예:

- `prison bars` -> `hallway railing`
- `corpse on ground` -> `resting person on ground`

### 4.3 Weak 1:1

concept 자체가 atmosphere/meme/style에 가까워서 명확한 safe 대응이 약하다.

예:

- `scariest thing ever`
- `surreal meme`

이 경우는 full-scope CAS target보다는

- high-IP subset 제한
- image-only probe
- 또는 제외

중 하나를 택해야 한다.

---

## 5. 출력 포맷

각 concept 문서는 최소한 아래 섹션을 가져야 한다.

1. dataset reading summary
2. actionable cue families
3. label-noisy / weakly-actionable regions
4. primary target words
5. secondary target words
6. do-not-use primary terms
7. anchor words
8. exemplar prompt pairs
9. recommended pilot subset
10. recommended v3/v4/v14-v19 integration notes

