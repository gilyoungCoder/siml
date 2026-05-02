# 2026-04-08 VLM Evaluation Related Work — Detailed Korean Analysis

이 문서는 우리 diffusion safety 연구의 NeurIPS evaluation story를 위해, 아래 세 논문을 **짧은 요약이 아니라 쉽게 풀어서 자세히 설명한** 노션 업로드용 메모다.

- VQAScore (ECCV 2024)
- MJ-Bench (NeurIPS 2025)
- LLaVA-Reward (ICCV 2025)

## 왜 이 세 논문을 같이 읽어야 하는가

우리 연구는 method도 중요하지만 evaluation story가 매우 중요하다. Reviewer는 단순히 “unsafe가 줄었다”만으로는 잘 설득되지 않는다. 반드시 다음을 묻게 된다.

1. 정말 unsafe가 줄었는가?
2. 그 과정에서 원래 prompt 의미를 너무 망친 것은 아닌가?
3. 그걸 판정하는 judge는 믿을 만한가?
4. safety / alignment / fidelity를 따로따로 보는 대신 더 체계적으로 볼 수는 없는가?

이 세 논문은 각각 이 질문에 답을 준다.

- VQAScore는 **prompt-image alignment를 VLM으로 더 정교하게 보는 방법**을 제시한다.
- MJ-Bench는 **multimodal judge 자체를 benchmark해야 한다**는 점을 보여준다.
- LLaVA-Reward는 **alignment / fidelity / safety를 learned reward model로 함께 다룰 수 있다**는 최신 흐름을 보여준다.

즉 우리 평가 story는 다음처럼 정리될 수 있다.

- VQAScore = alignment / 의미 보존
- Qwen3-VL = primary safety judge
- MJ-Bench = judge validation + graded safety taxonomy 근거
- LLaVA-Reward = auxiliary reward-style cross-check

---

## 1. VQAScore 상세 설명

### 왜 필요한가
기존 CLIPScore류는 이미지와 텍스트 임베딩 similarity를 재는 방식이라 빠르고 간단하지만, compositional prompt에는 약하다. 예를 들어 object, relation, counting, 논리 구조가 들어가면 text encoder가 bag-of-words처럼 동작해 문장 구조 차이를 충분히 반영하지 못한다.

이 논문은 alignment를 retrieval similarity로 보지 말고, **질문응답 문제**로 바꾸자고 제안한다.

### 핵심 아이디어
prompt `t`와 image `i`가 있을 때,

> “Does this figure show '{t}'?”

라는 질문을 VQA 모델에 던지고, 그 모델이 **Yes라고 답할 확률**을 alignment score로 사용한다.

즉 점수는 단순 cosine similarity가 아니라,
- 모델이 질문을 이해하고
- 이미지가 그 질문을 만족하는지 판단한 결과

가 된다.

### 왜 강한가
이렇게 하면 object 관계, attribute 조합, composition 같은 구조를 더 잘 반영할 수 있다. 논문은 off-the-shelf VQA 모델로도 성능이 좋고, 자체 모델인 CLIP-FlanT5는 bidirectional image-question encoding을 통해 더 강한 결과를 보인다고 주장한다. 또한 GenAI-Bench라는 compositional benchmark도 함께 제안해 metric의 타당성을 뒷받침한다.

### 우리 연구에 주는 의미
이 논문은 우리에게 “왜 VLM-based alignment evaluation을 쓰는가”에 대한 가장 직접적인 근거다. 특히 우리 setting에서는 안전한 이미지가 **원래 prompt의 safe한 의미를 얼마나 보존하는가**가 중요하기 때문에, VQAScore를 다음처럼 쓰는 것이 적절하다.

- `VQA(erased_image, original_prompt)`
- `VQA(erased_image, anchor_prompt)`
- `VQA(erased_image, erased_prompt)`
- `Gap = VQA(anchor) - VQA(original)`

즉 VQAScore는 safety metric이 아니라, **semantic preservation / prompt faithfulness metric**으로 쓰는 것이 맞다.

### 주의점
이 논문을 과대해석하면 안 된다.
- VQAScore는 safety-specific metric이 아니다.
- unsafe 여부 자체를 최종 판정해 주는 모델로 쓰면 안 된다.
- 특히 subtle/evasive unsafe case에서는 alignment와 safety가 같은 개념이 아니다.

그래서 우리 논문에서는 VQAScore를 **Qwen judge를 대체하는 metric**이 아니라, **의미 보존을 보는 complementary metric**으로 두는 게 가장 적절하다.

---

## 2. MJ-Bench 상세 설명

### 왜 필요한가
요즘 T2I evaluation에서 multimodal judge를 점점 많이 쓰지만, 정작 그 judge 자체를 얼마나 믿을 수 있는지는 별도로 검증되지 않는 경우가 많다. 이 논문은 바로 그 문제를 다룬다.

핵심 메시지는 간단하다.

> judge도 benchmark를 통해 평가받아야 한다.

즉 MJ-Bench는 image generation model을 직접 평가하는 benchmark가 아니라, **multimodal judge가 얼마나 좋은 judge인지 평가하는 benchmark**다.

### 데이터셋 구조
MJ-Bench는 preference dataset 구조를 가지며, 같은 prompt에 대해 두 이미지를 놓고 어느 쪽이 더 alignment/safety/quality/bias 측면에서 나은지 label을 제공한다.

큰 축은:
- alignment
- safety
- image quality / artifact
- bias / fairness

이다.

### safety taxonomy가 왜 중요한가
우리 연구와 가장 직접적으로 연결되는 것은 safety 세분화다.

#### NSFW
- **Evident**: 대놓고 unsafe
- **Subtle**: 애매하지만 분명 unsafe 신호가 있음
- **Evasive**: 우회적/예술적/공격 회피적 unsafe

#### Toxicity
- crime
- shocking
- disgust

이 taxonomy는 우리 current safety labeling과 매우 잘 맞는다.

- Full ↔ Evident
- Partial ↔ Subtle
- artistic/evasive unsafe ↔ Evasive

완전히 동일한 taxonomy는 아니지만, 핵심은 **unsafe를 binary가 아니라 graded severity로 본다**는 점이다. 이건 reviewer에게 매우 강한 외부 reference가 된다.

### 논문이 보여주는 핵심 결과
프로젝트 페이지 기준으로 읽히는 메시지는 다음과 같다.

- closed-source VLM judge가 평균적으로 가장 강하다.
- alignment / image quality에서는 작은 scorer가 경쟁력 있을 수 있다.
- safety / bias는 reasoning capacity가 필요한 경우가 많아서 VLM judge가 더 강하다.
- numeric score보다 natural-language / Likert-style feedback이 더 안정적일 수 있다.

이건 우리에게 좋은데, 왜냐하면 우리는 Qwen3-VL 같은 multimodal judge를 safety evaluator로 쓰고 있기 때문이다.

### 우리 연구에 주는 의미
MJ-Bench는 우리 generated image를 직접 평가하는 메인 metric이라기보다,

1. **Qwen3-VL judge calibration benchmark**
2. **graded safety taxonomy justification**

라는 두 역할이 가장 중요하다.

즉 논문에서는 이런 식으로 쓰는 게 좋다.
- “우리는 Qwen3-VL safety judge를 사용하지만, 그 신뢰도는 MJ-Bench를 통해 보강한다.”
- “우리의 Full / Partial style safety labeling은 MJ-Bench의 Evident / Subtle / Evasive와 같은 graded safety framing과 맞닿아 있다.”

### 주의점
- MJ-Bench는 judge benchmark이지 method benchmark가 아니다.
- pairwise preference에서 좋은 judge가 single-image 4-way classification에서도 똑같이 좋은 것은 아니다.
- 우리의 `NotRelevant`는 MJ-Bench safety taxonomy와 직접 대응되지는 않는다.

즉 MJ-Bench는 main metric을 대체하지 않는다. 대신 **judge를 정당화하는 외부 기준**으로 써야 한다.

---

## 3. LLaVA-Reward 상세 설명

### 왜 흥미로운가
VQAScore는 alignment metric이고, MJ-Bench는 judge benchmark다. LLaVA-Reward는 조금 다른 방향으로 간다. 이 논문은 text-to-image evaluation 자체를 **reward modeling** 문제로 본다.

즉 질문은 이런 식이다.
- alignment를 점수화할 수 있는가?
- fidelity/artifact를 점수화할 수 있는가?
- safety를 점수화할 수 있는가?
- 그리고 그 점수를 generation steering에도 쓸 수 있는가?

### 핵심 아이디어
기존 MLLM-based judge는 긴 instruction prompt를 넣고 자연어 답변을 생성한 뒤, 그 답변을 다시 점수로 해석하는 경우가 많았다. LLaVA-Reward는 이걸 바꾼다.

- pretrained multimodal LLM hidden state를 이용하고
- reward head를 붙여
- `(image, prompt)` 쌍에 대한 reward를 바로 출력한다.

즉 text generation judge가 아니라 **reward model**이다.

### SkipCA란 무엇인가
논문은 decoder-only MLLM에서 visual token 정보가 깊은 층으로 갈수록 약해질 수 있다고 보고, 이를 보완하기 위해 **Skip-connection Cross Attention (SkipCA)**를 쓴다.

쉽게 말하면,
- 후반 textual hidden state와
- 초반 visual token representation을
- cross-attention으로 다시 연결해서
- reward 예측 시 image-text correlation을 더 강하게 반영하게 만드는 구조다.

이건 alignment뿐 아니라 safety 판단에도 중요할 수 있다. 왜냐하면 safety도 단순 이미지 분류가 아니라, prompt와 image의 관계를 어느 정도 읽어야 하기 때문이다.

### 논문이 다루는 평가 축
LLaVA-Reward는 다음 4가지 perspective를 다룬다.
- text-image alignment
- fidelity / artifact
- safety
- overall ranking

즉 alignment / quality / safety를 분리해서 학습 가능한 reward head 관점으로 본다.

### 학습 데이터 / objective
논문은 paired preference와 unpaired binary data를 모두 활용한다. alignment와 fidelity는 pairwise preference, safety는 binary label 기반 데이터(예: UnsafeBench)를 사용한다. objective는 Bradley-Terry ranking loss와 classification loss 계열이다.

즉 이 모델의 score는 zero-shot heuristic이 아니라, **사람 선호와 annotation을 학습한 reward**다.

### 실험에서 중요한 메시지
1. alignment / fidelity / safety 같은 여러 축에서 human-aligned score를 잘 준다.
2. FK steering 같은 inference-time scaling에도 연결될 수 있다.

우리 method는 지금 reward-guided method는 아니지만, training-free steering이라는 큰 맥락에서는 미래 연결점이 있다.

### 우리 연구에 주는 의미
LLaVA-Reward는 지금 당장 우리 main evaluation backbone이 되기보다는,
- alignment
- safety
- fidelity

를 continuous score로 보는 **secondary evaluator 후보**다.

박준형이 말한 “prompt에서 너무 멀어지지 않으면서 harmful concept과는 멀어지는지”를 더 체계적으로 보자는 문제의식과도 잘 맞는다. 예를 들어,
- original prompt fidelity
- anchor prompt affinity
- safety reward
- artifact/fidelity reward

를 하나의 multimodal reward framework로 볼 가능성을 보여준다.

### 주의점
- learned reward model은 training data에 의존한다.
- reward hacking 가능성이 있다.
- scalar reward는 해석성이 떨어질 수 있다.
- 따라서 단독 truth judge로 쓰면 안 된다.

즉 우리 논문에서는 **main metric이 아니라 auxiliary learned cross-check** 정도로 두는 것이 가장 적절하다.

---

## 세 논문을 합친 최종 해석

이 세 논문을 같이 보면, 우리 evaluation stack은 꽤 현대적으로 설명될 수 있다.

### 추천 역할 분담
- **VQAScore**: prompt-image alignment / safe meaning preservation
- **Qwen3-VL**: primary safety judge
- **MJ-Bench**: judge reliability validation + graded safety taxonomy 근거
- **LLaVA-Reward**: learned reward-style auxiliary metric

### reviewer 대응용 핵심 문장
- VQAScore는 compositional alignment를 더 잘 측정하는 open-source VLM metric이다.
- MJ-Bench는 multimodal judge의 reliability를 검증하는 benchmark이며, graded safety evaluation의 근거를 제공한다.
- LLaVA-Reward는 alignment/fidelity/safety를 통합적으로 다루는 reward-model based evaluation의 최신 방향을 보여준다.

즉 우리 VLM-based evaluation은 단순 보조 실험이 아니라,
**alignment / safety / judge calibration / reward-style future extension**을 모두 포괄하는 평가 프레임으로 정리될 수 있다.
