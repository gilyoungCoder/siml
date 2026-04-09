# Mapping Docs Guide

## 목적

이 문서는 `docs/neurips_plan/multi_concept/` 아래에 만든 I2P 매핑 관련 문서들을

- 어떤 순서로 읽어야 하는지
- 각 문서가 무슨 역할을 하는지
- 실제 실험/구현 단계에서 어떤 문서를 기준으로 움직여야 하는지

를 설명하는 **사용 지침서**다.

이 문서는 OMC/OMX가 다음 세션에서도 빠르게 문맥을 복구할 수 있게 하기 위한
운영용 설명서다.

---

## 1. 문서 전체 구조

현재 매핑 관련 문서는 크게 3층으로 나뉜다.

### A. 공통 규칙 문서

- `i2p_mapping_guideline.md`

역할:

- 모든 concept에 공통으로 적용되는 추출 규칙
- target / anchor를 뽑는 기준
- primary / secondary / do-not-use 구분 원칙
- 1:1 mapping quality를 평가하는 기준

즉, **“어떻게 뽑아야 하느냐”**를 설명하는 최상위 규칙 문서다.

### B. concept별 spec 문서

- `violence_mapping_spec.md`
- `sexual_shocking_mapping_spec.md`
- `selfharm_illegal_mapping_spec.md`
- `harassment_hate_mapping_spec.md`

역할:

- 실제 dataset prompt를 읽고 정리한 결과
- concept family 분해
- target keywords
- anchor keywords
- exemplar prompt pairs
- high-IP subset / noisy subset 판단

즉, **“각 컨셉에서 실제로 뭘 쓸 거냐”**를 적은 문서다.

### C. 구조 통합 문서

- `multi_concept_pipeline.md`
- `multiconcept_integration_spec.md`

역할:

- 기존 v3 / v4 / v14~v19 구조에 concept pack을 어떻게 넣을지
- concept-by-concept가 아니라 family-by-family로 확장하는 이유
- text / image / both probe를 어떻게 concept별로 다르게 써야 하는지

즉, **“이걸 시스템에 어떻게 꽂을 거냐”**를 적은 문서다.

---

## 2. 추천 읽기 순서

다음 순서가 가장 좋다.

### 1단계: 공통 규칙 이해

먼저 읽을 것:

- `i2p_mapping_guideline.md`

이걸 먼저 읽어야 이후 문서들을 같은 기준으로 해석할 수 있다.

핵심 질문:

- 왜 label만 보고 target을 잡으면 안 되는가?
- 왜 artist/style/name 토큰은 primary target으로 쓰면 안 되는가?
- 왜 anchor는 scene-preserving rewrite여야 하는가?

### 2단계: concept별 실제 매핑 읽기

그다음 읽을 것:

- `violence_mapping_spec.md`
- `sexual_shocking_mapping_spec.md`
- `selfharm_illegal_mapping_spec.md`
- `harassment_hate_mapping_spec.md`

읽는 순서는 추천상:

1. `violence`
2. `sexual/shocking`
3. `self-harm/illegal`
4. `harassment/hate`

이유:

- `violence`, `sexual`, `shocking`은 비교적 actionable family가 명확하다.
- `harassment`, `hate`는 label noise가 크므로 나중에 읽는 게 이해가 쉽다.

### 3단계: 구조 통합

마지막에 읽을 것:

- `multi_concept_pipeline.md`
- `multiconcept_integration_spec.md`

이 단계에서

- concept pack 구조
- family-level correction
- v3 / v4 / v14~v19 연결 방식

을 본다.

---

## 3. 문서별 사용 목적

## 3.1 `i2p_mapping_guideline.md`

사용 시점:

- 새 concept를 추가할 때
- 기존 spec을 수정할 때
- target/anchor 선정 기준이 흔들릴 때

이 문서를 먼저 기준으로 잡고,
그 다음 concept spec을 수정해야 한다.

### 3.2 `violence_mapping_spec.md`

사용 시점:

- violence pilot exemplar 제작
- violence target_words_primary 설계
- violence anchor prompt pair 구성

현재 가장 직접적으로 활용 가능한 spec이다.

### 3.3 `sexual_shocking_mapping_spec.md`

사용 시점:

- nudity beyond nude/clothed를 다룰 때
- shocking face/anguish/body-horror family를 pilot할 때

`sexual`은 nudity와 매우 가깝지만
“revealing clothing / explicit act”가 추가된 확장 버전으로 읽어야 한다.

### 3.4 `selfharm_illegal_mapping_spec.md`

사용 시점:

- self-harm pilot을 할 때
- illegal activity에서 `drugs / smoking / mugshot` family만 먼저 자를 때

특히 illegal은 noise가 심하므로
여기 적힌 actionable family만 우선적으로 써야 한다.

### 3.5 `harassment_hate_mapping_spec.md`

사용 시점:

- 왜 harassment/hate가 어려운지 설명할 때
- full dataset보다 subset first가 필요한 이유를 설명할 때
- weak CAS category를 다룰 때

이 문서는 “바로 실험용”이라기보다
**왜 보수적으로 가야 하는지**를 설명하는 방어 문서 역할도 한다.

### 3.6 `multi_concept_pipeline.md`

사용 시점:

- 전체 로드맵을 볼 때
- concept 확장 우선순위를 정할 때
- 논문 narrative와 연결할 때

### 3.7 `multiconcept_integration_spec.md`

사용 시점:

- 실제 구현 설계를 할 때
- v3 / v4 / v14~v19 중 어떤 베이스를 택할지 논의할 때
- concept pack을 코드에 연결할 때

---

## 4. OMC/OMX에서 이 문서들을 쓰는 방식

다음 세션에서 OMC/OMX가 이 작업을 재개할 때는 다음 흐름이 좋다.

### Step 1. guideline 먼저 읽기

항상:

- `i2p_mapping_guideline.md`

를 먼저 읽고 기준을 복구한다.

### Step 2. 대상 concept spec 읽기

예를 들어 violence 구현이면:

- `violence_mapping_spec.md`

만 우선 읽는다.

sexual/shocking 구현이면:

- `sexual_shocking_mapping_spec.md`

를 읽는다.

### Step 3. 통합 구조 문서 읽기

그다음:

- `multiconcept_integration_spec.md`

를 읽고 어떤 version family에 연결할지 결정한다.

### Step 4. 코드/실험으로 넘어간다

이때만

- `CAS_SpatialCFG/generate_v*`
- `prepare_concept_subspace.py`
- `prepare_clip_patch_tokens.py`

로 넘어간다.

즉 문서 → spec → integration → code 순서다.

---

## 5. 실험 우선순위

현재 문서 기준 추천 순서는 아래다.

1. `violence`
2. `sexual`
3. `shocking`
4. `self-harm`
5. `illegal_activity`
6. `harassment`
7. `hate`

그리고 각 concept 내부에서도

1. `strong 1:1 family`
2. `medium 1:1 family`
3. `weak / noisy family`

순으로 들어가야 한다.

이 규칙은 매우 중요하다.

왜냐하면 concept 전체를 한 번에 넣으면 drift와 실패 원인을 분리하기가 어려워지기 때문이다.

---

## 6. 가장 중요한 해석 포인트

### 6.1 concept 전체를 지우지 말고 family를 지워라

이 문서 세트의 가장 중요한 메시지는:

**concept-level erase가 아니라 family-level erase가 맞다**는 것이다.

예:

- `violence`
  - weapon threat
  - bodily injury
  - detention/coercion

- `sexual`
  - exposed body
  - revealing clothing
  - explicit act

이 family들이 진짜 설계 단위다.

### 6.2 anchor는 concept당 하나가 아니라 family당 하나다

`violence -> peace`
이렇게 가면 너무 넓다.

대신:

- `knife -> flashlight`
- `prison bars -> hallway`
- `bloody wound -> clean skin`

같이 family별 local anchor를 둬야 한다.

### 6.3 harassment/hate는 실패가 아니라 어려운 데이터다

이 문서 세트는 harassment/hate를
“우리가 못 한다”가 아니라
“dataset label과 prompt cue가 약하게 연결되어 있어서 pilot 범위를 줄여야 한다”
로 해석한다.

즉, 이건 방법론 실패가 아니라 데이터 특성 문제다.

---

## 7. 다음 세션에서 해야 할 일

이 문서들을 기반으로 다음 세션에서는 보통 아래 중 하나를 하면 된다.

### A. 구현 세션

- 특정 concept 하나 선택
- 해당 spec 읽기
- concept pack 파일 구조 만들기
- exemplar prompt txt / keyword txt 생성

### B. refinement 세션

- 각 spec의 keyword를 더 줄이거나 정리
- anchor-target pair를 더 object-centric하게 다듬기
- high-IP subset 정의를 더 명확히 하기

### C. integration 세션

- `multiconcept_integration_spec.md` 기준으로
- v14 / v15 / v18 중 어디를 first multi-concept base로 할지 확정

---

## 8. 한 줄 요약

이 문서 세트는:

- `i2p_mapping_guideline.md` = 공통 규칙
- `*_mapping_spec.md` = concept별 실제 mapping
- `multiconcept_integration_spec.md` = 기존 구조와의 연결

로 읽으면 된다.

그리고 핵심은 언제나 같다.

> **label을 지우는 게 아니라, family-level unsafe cue를 지우고, 그 cue에 1:1로 대응하는 safe anchor로 밀어야 한다.**

