# Self-harm + Illegal Activity Mapping Spec

## 목적

이 문서는 `i2p_self-harm.csv`와 `i2p_illegal_activity.csv`를 기반으로,
실제로 multi-concept erasing에 쓸 수 있는 target-anchor mapping을 정리한 내부 spec이다.

---

## 1. Self-harm Concept

### 1.1 Dataset Reading Summary

`self-harm`은 단일 개념이 아니다.

실제 상위 프롬프트에서는 다음이 반복된다.

- hanging / noose / suicide-like cue
- emaciation / anorexia / self-destruction
- depression / collapse / isolation room
- mutilated body / shredded face / blood
- horror possession / ghost / demon

이 중 앞의 세 가지는 self-harm에 가깝고,
뒤의 두 가지는 violence/shocking으로 새는 경우가 많다.

### 1.2 Actionable Cue Families

#### A. Hanging / Suicide Cue

`target_keywords`

- `hanging`
- `noose`
- `hanged`
- `hanging from`
- `limp puppet`

`anchor_keywords`

- `standing`
- `seated`
- `resting`
- `supported`
- `safe rope-free scene`

#### B. Emaciation / Self-Deterioration

`target_keywords`

- `anorexic`
- `eating herself to death`
- `withered`
- `fetal pose`
- `starving`

`anchor_keywords`

- `healthy`
- `resting`
- `recovering`
- `supported`
- `ordinary body`

#### C. Depression / Isolation / Collapse

`target_keywords`

- `depression`
- `mental health`
- `head leaned low`
- `dirty room`
- `hopeless`
- `crying`
- `alone`

`anchor_keywords`

- `tidy room`
- `calm room`
- `supported posture`
- `meditation`
- `recovery`
- `therapy-like scene`

### 1.3 Secondary Words

- `horror`
- `dark`
- `ghost`
- `surreal`
- `scary`

보조 signal로는 유용하지만 self-harm 자체는 아니다.

### 1.4 Do-Not-Use Primary Terms

- artist/style names
- `woman`
- `face`
- `beautiful`
- `beksinski`
- `junji ito`

### 1.5 Object-Centric Exemplar Prompt Pairs

#### Pair set A: hanging / noose

- target: `a person hanging from a noose in a dim room`
- anchor: `a person standing in a dim room`

- target: `a limp puppet hanging by strings`
- anchor: `a puppet standing upright on strings`

#### Pair set B: emaciation / self-deterioration

- target: `an emaciated person in a realistic portrait`
- anchor: `a healthy person in a realistic portrait`

- target: `a withered woman in a fetal pose`
- anchor: `a resting woman curled on a sofa`

#### Pair set C: depression / isolation

- target: `a person sitting alone in a dirty room with head down`
- anchor: `a person sitting in a tidy room with calm posture`

- target: `a person leaning against a wall in despair`
- anchor: `a person standing against a wall in a calm pose`

### 1.6 1:1 Mapping Quality

- hanging / noose: **strong**
- emaciation / self-deterioration: **medium**
- depression / isolation: **medium**
- gore-heavy self-harm: **weak to medium**

### 1.7 Recommended Self-harm Pilot

pilot은 아래만 먼저 잡는 것이 좋다.

1. hanging / noose
2. emaciation / self-deterioration
3. depression / isolation room

고어/귀신/악마 중심 프롬프트는 self-harm pilot의 primary target에서 빼는 것이 낫다.

---

## 2. Illegal Activity Concept

### 2.1 Dataset Reading Summary

`illegal_activity`는 실제로 여러 하위 개념이 섞여 있다.

- drugs / pills / substance use
- smoking / alcohol / intoxication
- mugshot / arrest evidence
- riot / vandalism / graffiti
- noisy horror/meme contamination

이 카테고리는 label noise가 심하지만,
`pills / drugs / smoking / mugshot`은 비교적 직접적인 cue를 준다.

### 2.2 Actionable Cue Families

#### A. Drugs / Pills / Substance Table

`target_keywords`

- `pills`
- `drugs`
- `powder`
- `addiction`
- `skull on the table`
- `substance`

`anchor_keywords`

- `medicine`
- `vitamins`
- `supplements`
- `bottles`
- `objects on a table`

#### B. Smoking / Alcohol / Consumption

`target_keywords`

- `smoking`
- `weed`
- `cigar`
- `cigarette`
- `vodka`
- `drinking`

`anchor_keywords`

- `holding a cup`
- `tea`
- `coffee`
- `glass bottle`
- `breathing`

#### C. Mugshot / Custody Evidence

`target_keywords`

- `mugshot`
- `prison cell`
- `prisoner`
- `holding prison bars`

`anchor_keywords`

- `front portrait`
- `ID-style portrait`
- `headshot`
- `standing in a hallway`

#### D. Riot / Graffiti / Vandalism

`target_keywords`

- `riot`
- `graffiti`
- `covered with graffiti`
- `vandalized`

`anchor_keywords`

- `crowd scene`
- `mural`
- `painted train`
- `clean wall`

### 2.3 Secondary Words

- `psychedelic`
- `surreal`
- `creepy`
- `dark`

이건 illegal 자체보다 atmosphere를 강화한다.

### 2.4 Do-Not-Use Primary Terms

- `trump`
- `nigel farage`
- `moistcritikal`
- general horror entity terms
- style names

### 2.5 Object-Centric Exemplar Prompt Pairs

#### Pair set A: drugs / pills

- target: `pills and drugs on a wooden table`
- anchor: `medicine bottles on a wooden table`

- target: `a pile of pills beside a smoking skull`
- anchor: `a row of bottles beside a ceramic sculpture`

#### Pair set B: smoking / alcohol

- target: `a person smoking a cigar in a portrait`
- anchor: `a person holding a cup in a portrait`

- target: `a bottle of vodka beside cigarettes`
- anchor: `a glass bottle beside tea boxes`

- target: `a band member smoking weed in a studio shot`
- anchor: `a band member holding a microphone in a studio shot`

#### Pair set C: mugshot / custody evidence

- target: `a front-facing mugshot portrait`
- anchor: `a front-facing studio portrait`

- target: `a prisoner holding bars in a cell`
- anchor: `a person holding a railing in a hallway`

#### Pair set D: graffiti / riot

- target: `a train covered with graffiti`
- anchor: `a train painted with a public mural`

- target: `a crowd scene during a riot`
- anchor: `a crowd scene during a public event`

### 2.6 1:1 Mapping Quality

- drugs / pills: **strong**
- smoking / alcohol: **strong**
- mugshot / custody evidence: **medium**
- riot / graffiti: **medium**
- horror/meme contamination: **weak**

### 2.7 Recommended Illegal Activity Pilot

pilot은 아래 3 family부터 가는 게 좋다.

1. drugs / pills
2. smoking / alcohol
3. mugshot / custody evidence

riot/graffiti는 2차 확장으로 미루는 것이 안전하다.
