# MJ-Bench 분석 메모 for OMC

> 대상 논문: **MJ-BENCH: Is Your Multimodal Reward Model Really a Good Judge for Text-to-Image Generation?**  
> arXiv: 2407.04842v1  
> repo: https://github.com/BillChan226/MJ-Bench

---

## 1. 이 논문이 실제로 하는 일

이 논문은 **이미지 생성 모델 자체를 평가하는 benchmark**가 아니라,  
**이미지 생성 결과를 평가하는 multimodal judge 자체를 평가하는 benchmark**다.

즉 질문은:

- “GPT-4o, Claude, Qwen-VL, CLIP-based scorer 같은 judge가 정말 좋은 judge인가?”
- “alignment / safety / quality / bias를 얼마나 정확하게 판정하는가?”

이다.

핵심 데이터 단위는 아래 triplet이다.

- `I`: instruction / prompt
- `M_p`: chosen image
- `M_n`: rejected image

즉 한 샘플은:

> “이 prompt에 대해 두 이미지 중 어느 쪽이 더 바람직한가?”

를 이미 정답(label)로 갖고 있는 **preference pair**다.

---

## 2. 평가 축 4개

MJ-Bench는 judge를 아래 4개 perspective에서 본다.

1. **Alignment**
2. **Safety**
3. **Quality / Artifact**
4. **Bias / Fairness**

각 perspective는 더 잘게 쪼개진다.

---

## 3. Alignment는 어떻게 평가하나

### 세부 subcategory
- **Object**
- **Attribute**
- **Action**
- **Spatial / Location**
- **Count**

즉 judge가 단순히 “대충 prompt 비슷함”이 아니라,

- 물체가 맞는지
- 속성이 맞는지
- 행동이 맞는지
- 위치 관계가 맞는지
- 개수가 맞는지

를 선호쌍(preference pair) 기준으로 얼마나 잘 가르는지 본다.

### 데이터 수집 방식
- Pick-a-pic
- HPDv2
- ImageRewardDB

같은 공개 preference dataset에서 후보를 뽑고,  
LLaVA-NeXT-34B를 surrogate로 써서 alignment 오류 유형을 분류한 뒤,  
최종적으로 **human filtering**으로 검증한다.

즉 완전 자동 데이터셋이 아니라,  
**자동 선별 + 사람 검수** 기반 curated benchmark다.

---

## 4. Safety는 어떻게 평가하나

Safety는 크게 두 상위 카테고리로 나뉜다.

### 4.1 Toxicity
세부 subcategory:
- **Crime**
- **Shocking**
- **Disgust**

### 4.2 NSFW
세부 subcategory:
- **Evident**
- **Subtle**
- **Evasive**

이 NSFW taxonomy가 우리에게 가장 중요하다.

### 우리 프로젝트와의 연결

MJ-Bench:
- Evident
- Subtle
- Evasive

우리:
- Full
- Partial
- Safe
- NotRelevant

가장 자연스러운 대응은:

- **Evident → Full**
- **Subtle → Partial**
- **Evasive → evasive / artistic / attack-like hard case**

즉 MJ-Bench는 unsafe를 binary가 아니라  
**강도/명확성에 따라 graded하게 보는 외부 근거**가 된다.

---

## 5. Quality / Artifact는 어떻게 보나

Quality는 두 갈래다.

### Distortion
- Human face distortion
- Human limb distortion
- Object distortion

### Blur
- Defocused blur
- Motion blur

즉 judge가 생성 이미지의 artifact를 얼마나 잘 잡는지 본다.

---

## 6. Bias / Fairness는 어떻게 보나

Bias는 pairwise accuracy만 보는 것이 아니라,  
demographic variation 전체에서 judge 점수가 얼마나 공정하고 일관적인지 본다.

상위 카테고리:
- **Occupation**
- **Education**

변수:
- age
- race
- gender
- nationality
- religion

즉 여기서는 “chosen/rejected 잘 고르나”보다,  
**judge 점수가 demographic에 따라 불필요하게 흔들리는가**가 핵심이다.

---

## 7. judge 입력 방식 2종

이 논문에서 매우 중요한 부분이다.

### 7.1 Single-input judge

대상:
- CLIP-based score models
- single-image VLMs

방식:
1. `(I, M_p)`를 넣어 점수 `s_p`
2. `(I, M_n)`를 넣어 점수 `s_n`
3. 두 점수 차이로 preference 생성

즉 pairwise를 직접 판단시키는 게 아니라,  
**각 이미지 점수를 따로 매기고 비교해서 승패를 만든다.**

---

### 7.2 Multi-input judge

대상:
- pairwise image input이 가능한 VLM

방식:
- prompt + 두 이미지 동시에 입력
- Analyze-then-Judge
- 출력:
  - image1 rating
  - image2 rating
  - preference in `{0, 1, 2}`

의미:
- `1`: image1이 더 좋음
- `2`: image2가 더 좋음
- `0`: tie / equal / cannot decide

즉 multi-input judge는  
**pairwise preference를 직접 출력**한다.

---

## 8. 가장 중요한 부분: tie를 어떻게 처리하나

이 논문은 tie를 대충 넘기지 않고,  
**judge의 reliability를 분해해서 보는 핵심 요소**로 다룬다.

---

### 8.1 Single-input judge에서 tie 생성

single-input judge는 `s_p`와 `s_n`의 차이를 보고 preference를 만든다.

threshold `ζ_t`를 둔다.

규칙:

- `s_p > s_n + ζ_t` → chosen 정답
- `s_n > s_p + ζ_t` → rejected 선택 = 오답
- `|s_p - s_n| < ζ_t` → **tie**

즉 점수 차가 충분히 크지 않으면  
“judge가 둘을 구분 못했다”고 보고 tie 처리한다.

---

### 8.2 Multi-input judge에서 tie 생성

multi-input VLM은 prompt 자체가 tie 옵션을 갖는다.

- `0` = tie / cannot decide

즉 여기서는 tie가 threshold 기반이 아니라  
**judge가 직접 선언한 tie**다.

---

## 9. tie를 두 가지 방식으로 평가한다

이 논문은 tie를 한 방식으로만 처리하지 않는다.  
**with tie / without tie**를 둘 다 보고한다.

---

### 9.1 Accuracy with tie

논문 표현:
- **with tie (considering tie as false predictions)**

즉 tie를 **오답**으로 친다.

개념적으로:

\[
Acc_{with\ tie} = \frac{\# correct}{\# total\ pairs}
\]

여기서 tie는 numerator에 안 들어가고 denominator에는 들어간다.

### 의미
이건 judge가 실전에서
- 얼마나 자주 제대로 결정을 내리는지
- 못 고르는 경우까지 포함해

보는 지표다.

즉 **실전성 / 결정성 포함 성능**이다.

---

### 9.2 Accuracy without tie

논문 표현:
- **without tie (filtering out all tie predictions)**

즉 tie를 평가 집합에서 제거한다.

개념적으로:

\[
Acc_{without\ tie} = \frac{\# correct}{\# non\text{-}tie\ predictions}
\]

### 의미
이건 judge가

> “결정했을 때는 얼마나 잘 맞히는가?”

를 보는 지표다.

즉 **확신 있는 판정만 놓고 본 순수 정확도**다.

---

### 9.3 왜 둘 다 보나

예를 들어 어떤 judge가:
- tie를 엄청 많이 내면
- without-tie accuracy는 높아질 수 있다
- 하지만 실제 judge로 쓰기엔 별로일 수 있다

반대로 어떤 judge가:
- 무조건 결정은 내리는데
- 자주 틀리면
- with-tie accuracy는 낮을 수 있다

즉 이 논문은 tie를 통해 아래 둘을 분리해서 본다.

1. **결정성(decisiveness)**
2. **결정했을 때의 정확성**

이게 매우 중요하다.

---

## 10. score model에서 tie-threshold sweep이 의미하는 것

Appendix C.2에서 score model에 대해 tie threshold를 바꿔가며 본다.

### 왜 하나
threshold를 올리면 tie가 더 많이 생긴다.  
즉 judge가 “정말 점수 차가 클 때만” 승패를 선언하게 된다.

그래서 threshold sweep은 사실상

- score margin이 큰지
- confidence가 있는지
- noise가 많은지

를 보는 실험이다.

### 논문 해석
- **PickScore-v1**는 threshold를 올려도 비교적 안정
  - score margin이 크고 confidence가 높음
- **HPS-v2.1**는 기본 성능은 좋지만 threshold를 올리면 성능 하락
  - 점수 분산/noise가 더 큼

즉 threshold 분석은
**“이 scorer 점수 차이를 얼마나 믿을 수 있나”**를 본다.

---

## 11. 표(Table)별로 무엇을 읽어야 하나

### Table 1
메인 결과표.

#### Alignment / Safety / Artifact
각각:
- `Avg w/ tie`
- `Avg w/o tie`

를 제공한다.

즉 세 축은 모두 preference classification으로 보고,
tie 포함 / tie 제거 정확도를 같이 본다.

#### Bias
bias는 아래 3개 metric으로 본다.
- **ACC**
- **NDS**
- **GES**

즉 bias는 완전히 다른 평가 프레임이다.

---

### Table 2
judge feedback으로 fine-tune한 이미지 생성 모델을  
사람이 직접 평가한 결과.

metrics:
- **FR**: fixed seed ranking
- **RR**: random seed ranking
- **AR**: average ranking
- **AV**: average voting

즉:

> “automatic judge benchmark에서 좋은 judge가, 실제로 그 judge feedback으로 모델을 학습시켰을 때도 좋은가?”

를 human eval로 본다.

---

### Table 3
fine-tuning algorithm 비교:
- **DPO**
- **DDPO**

즉 judge 자체가 아니라,  
judge feedback을 어떤 alignment 알고리즘으로 쓸 때 더 안정적인지 본다.

논문 결론:
- DPO가 더 안정적
- GPT-4o / GPT-4-vision feedback이 좋음

---

### Table 4
open-source multi-input VLM의 입력 순서 민감도 분석

모드:
- `single`
- `pair-f` (chosen 먼저)
- `pair-r` (rejected 먼저)

핵심 발견:
- **Qwen-VL-Chat**은 non-prioritized image 선호 경향
- **InternVL-chat-v1-5**는 prioritized image 선호 경향

즉 **이미지 입력 순서에 민감한 judge bias**가 있다.

우리 쪽에서 pairwise multi-image judge를 쓴다면 반드시 참고해야 한다.

---

### Table 5
rating scale별 성능 비교

비교:
- numeric [0,1], [0,5], [0,10], [0,100]
- Likert 5 / Likert 10

핵심 발견:
- open-source VLM은 **Likert scale에서 더 잘함**
- 숫자화는 상대적으로 약함
- closed-source VLM은 scale 변화에 더 stable

즉 우리 Qwen judge prompting에서  
무조건 숫자 하나를 강제하는 게 최선은 아닐 수 있다.

---

## 12. Bias metric 3개 의미

### 12.1 ACC
demographic variation pair 간 점수 차가 threshold 이하이면 unbiased라고 보고 계산하는 accuracy.

즉 pairwise fairness 관점.

---

### 12.2 GES
**Gini-based Equality Score**

- Gini coefficient 기반
- `GES = 1 - G`

높을수록 demographic 간 점수 분배가 평등함.

---

### 12.3 NDS
**Normalized Dispersion Score**

- 표준편차 / 평균 기반 dispersion score
- `NDS = 1 - NSD`

높을수록 점수 분산이 작고 일관적임.

---

## 13. 논문의 실질적 결론

### judge family별
- **closed-source VLM**: 전반적으로 최고
- **CLIP-based scoring models**: alignment / quality에선 강할 수 있음
- **VLMs**: safety / bias는 reasoning 때문에 더 강함

### ties 관점
이 논문은 tie를 단순 예외처리가 아니라,
**judge 신뢰도를 분해해서 보는 핵심 메커니즘**으로 사용한다.

핵심:
- `with tie` = 실전적 usefulness 포함
- `without tie` = 확신 있는 판정만의 정확도
- threshold sweep = confidence / score noise 분석

---

## 14. 우리 repo에 주는 직접적인 의미

### 우리가 MJ-Bench를 쓰는 가장 좋은 방식
1. **Qwen3-VL judge calibration benchmark**
2. **graded safety taxonomy reference**
3. 필요하면 prompt / rating scale / tie 처리 설계 참고

### 우리 쪽 대응
- Evident → Full
- Subtle → Partial
- Evasive → evasive hard case
- NotRelevant → MJ-Bench엔 직접 대응 없음, 우리 별도 축 유지 필요

---

## 15. OMC용 바로 쓸 결론

### 핵심만 짧게
- MJ-Bench는 **judge benchmark**다.
- 데이터는 `(instruction, chosen, rejected)` preference triplet.
- 평가 축은 alignment / safety / quality / bias.
- safety는 특히 **crime / shocking / disgust / evident / subtle / evasive**로 쪼갠다.
- tie는 두 방식으로 처리:
  - **with tie**: tie를 오답 처리
  - **without tie**: tie를 제거하고 정확도 계산
- score model은 threshold `ζ_t`로 tie를 만든다.
- threshold sweep은 confidence / score noise 분석이다.

### 우리 연구에 주는 직접 메시지
- Qwen3-VL judge를 그냥 쓰지 말고 **judge benchmark로 calibration**해야 한다.
- Full / Partial 같은 graded safety label은 MJ-Bench NSFW taxonomy로 정당화 가능하다.
- tie도 반드시 분리해서 봐야 한다.

