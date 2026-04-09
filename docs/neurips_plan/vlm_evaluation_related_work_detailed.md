# VLM Evaluation Related Work — Detailed Korean Analysis for Our NeurIPS Story

> 목적: VQAScore, MJ-Bench, LLaVA-Reward 세 논문을 **짧은 요약이 아니라**, 우리 연구와 연결되는 방식으로 **논문을 처음 읽는 사람도 이해할 수 있게 한국어로 자세히 풀어 설명**한 내부 메모.
>
> primary source 기준:
> - VQAScore: ECCV 2024 poster / official PDF / arXiv 2404.01291
> - MJ-Bench: NeurIPS 2025 poster / official project page / arXiv 2407.04842 / dataset page
> - LLaVA-Reward: ICCV 2025 official open-access paper / arXiv 2507.21391

---

## 먼저 큰 그림: 왜 이 세 논문을 같이 읽어야 하는가

우리 프로젝트는 training-free diffusion safety 연구다. 그런데 method가 아무리 좋아도 논문에서는 결국 이런 질문을 받게 된다.

1. **정말 unsafe content가 줄었는가?**
2. **그 과정에서 원래 prompt 의미를 너무 많이 망치지는 않았는가?**
3. **그걸 판정하는 evaluator 자체는 믿을 만한가?**
4. **안전성, 정합성, fidelity를 따로따로 보는 대신 더 통합적으로 볼 수는 없는가?**

이 네 질문에 대해 세 논문이 각각 거의 역할을 분담해서 답을 준다.

- **VQAScore**는 “prompt와 image의 alignment를 VLM으로 더 정교하게 볼 수 있다”는 근거를 준다.
- **MJ-Bench**는 “multimodal judge를 그냥 믿지 말고, judge 자체를 벤치마크로 검증해야 한다”는 근거를 준다.
- **LLaVA-Reward**는 “alignment, fidelity, safety를 따로 분리된 metric이 아니라 learned reward 관점으로도 볼 수 있다”는 최신 흐름을 보여준다.

즉 이 세 논문을 같이 가져가면 우리 평가 스토리는 이렇게 정리된다.

- **VQAScore** → 의미 보존 / prompt faithfulness
- **Qwen3-VL judge** → 생성 이미지의 safety 판정
- **MJ-Bench** → 그 judge를 얼마나 믿어도 되는지 검증하는 외부 근거
- **LLaVA-Reward** → 추가적인 learned reward-style cross-check 혹은 future extension

이게 이번 메모의 가장 중요한 결론이다.

---

# 1. VQAScore

## 1.1 이 논문이 왜 나왔는가

Text-to-image 모델을 평가할 때 오랫동안 많이 쓰인 자동 metric은 CLIPScore 계열이었다. 이유는 단순하다. 이미지와 텍스트를 같은 임베딩 공간에 올려놓고 cosine similarity를 재면 되니 빠르고 쉽다.

그런데 이 방식에는 오래된 구조적 한계가 있다.

예를 들어 prompt가 아래처럼 단순하면 CLIP류도 어느 정도 버틴다.
- “a red car”
- “a dog in snow”

하지만 prompt가 조금만 복잡해지면 문제가 생긴다.
- object가 여러 개 들어가고
- 관계가 들어가고
- attribute가 섞이고
- counting, logic, composition이 들어가면

CLIP류는 텍스트를 꽤 자주 **bag-of-words처럼** 처리한다. 즉 단어가 다 들어 있으면 대충 맞다고 보는 경향이 있다. 그래서 논문이 예로 드는 식의 문제가 생긴다.

- “the horse is eating the grass”
- “the grass is eating the horse”

단어는 비슷하지만 의미 구조는 완전히 다르다. 사람이 보기에는 전혀 다른 문장이지만, 단순 embedding similarity는 둘의 차이를 충분히 강하게 반영하지 못할 수 있다.

이 논문의 출발점은 바로 여기다.

> “좋은 image-text evaluation은 단순 단어 겹침이 아니라, **이 이미지가 정말 이 문장을 보여주고 있는가**를 물어야 한다.”

즉 문제 정의 자체가 **alignment를 semantic question answering 문제로 바꾸자**는 것이다.

---

## 1.2 핵심 아이디어: image-text matching을 yes/no VQA로 바꾸기

논문이 제안하는 VQAScore의 핵심은 놀랄 만큼 직관적이다.

prompt `t`와 image `i`가 있을 때, 이 둘의 정합성을 보려면 다음 질문을 던지면 된다.

> “Does this figure show ‘{t}’?”

그리고 VQA 모델이 여기에 **Yes라고 답할 확률**을 점수로 쓰자는 것이다.

즉, 기존 방식은
- 이미지 벡터
- 텍스트 벡터
- 둘의 cosine similarity

였다면,

VQAScore는
- 이미지 + 질문
- VQA 모델
- Yes 확률

이 된다.

이 변화가 중요한 이유는, 모델이 단순 유사도 계산을 하는 것이 아니라 **문장 자체를 질문으로 이해하고, 이미지가 그 질문을 만족하는지 판단**하게 되기 때문이다.

이건 compositional prompt에서 특히 강하다.

예를 들어,
- “a blue cup on top of a red book”
- “a red book on top of a blue cup”

는 단어만 보면 거의 동일하지만, 질문 형태로 바꾸면 관계가 훨씬 중요해진다. 논문이 강조하는 포인트는 바로 여기다. alignment를 retrieval similarity로 보는 대신 **grounded yes/no verification**으로 바꾼 셈이다.

---

## 1.3 VQAScore의 수학적/실무적 직관

논문 abstract 수준에서 요약하면 VQAScore는 다음과 같다.

- 입력: `(image, text)`
- 질문 생성: `Does this figure show '{text}'?`
- 모델 출력: `P(Yes | image, question)`
- 점수: 이 Yes 확률

실무적으로 보면 우리 repo의 `vlm/eval_vqascore.py`가 거의 이 철학을 그대로 구현하고 있다.

실제 스크립트도:
- InstructBLIP 계열을 사용하고
- yes / no 토큰의 softmax를 구하고
- `P(yes)`를 점수로 쓴다.

이건 논문 아이디어를 상당히 직접적으로 따르고 있는 셈이다.

---

## 1.4 논문이 추가로 한 일: off-the-shelf만 쓴 게 아니다

이 논문이 단순히 “질문으로 바꿔보니 괜찮더라”에서 끝나는 건 아니다.

논문은 두 층으로 기여한다.

### A. off-the-shelf VQA model로도 강하다
이미 공개된 VQA 모델만 갖고도 기존 alignment metric보다 성능이 좋다는 것을 보인다.

### B. 자체 모델 CLIP-FlanT5를 만들었다
논문은 자체적으로 **CLIP-FlanT5** 계열 모델도 제안한다. 여기서 중요한 설계 포인트는 abstract에 나온 표현대로 **bidirectional image-question encoder**라는 점이다. 즉 이미지 임베딩이 질문에 의존하고, 질문 표현도 이미지에 의존할 수 있게 설계해서, 단순 독립 인코딩보다 더 강하게 compositional alignment를 잡는다.

논문은 이 자체 모델이 심지어 proprietary GPT-4V 기반 baseline보다도 더 강하다고 주장한다. 여기서 reviewer를 설득하기 좋은 메시지는 다음이다.

> VQAScore는 “VLM을 써보자” 수준의 감이 아니라, 이미 ECCV에서 **alignment evaluation 전용으로 정식 제안된 프레임**이다.

---

## 1.5 GenAI-Bench가 왜 중요하나

이 논문에서 metric만 중요한 게 아니다. 같이 제안한 **GenAI-Bench**도 중요하다.

왜냐하면 기존 benchmark들은 너무 쉬운 prompt가 많아서 metric이 진짜 compositional alignment를 평가하는지 분간이 잘 안 갔기 때문이다.

GenAI-Bench는:
- 1,600개의 compositional prompt를 포함하고
- 15,000개 이상의 human rating을 제공하며
- object, attribute, relation, comparison, logic 같은 harder case를 포함한다.

즉, 이 논문은 “좋은 metric을 만들었다”뿐 아니라 “그 metric을 검증할 만한 더 어려운 benchmark도 만들었다”는 구조다.

우리 입장에서 이게 중요한 이유는,
단순 nude / not nude를 넘어서 **prompt 의미 보존**을 보려면 compositional understanding이 필요하기 때문이다. 예를 들어 unsafe token만 빼고 나머지 scene structure가 유지됐는지를 보려면, evaluator가 문장 구조를 어느 정도 읽어야 한다. CLIPScore만으로는 이 부분이 약하고, VQAScore는 여기서 더 타당한 선택지가 된다.

---

## 1.6 우리 연구에 어떻게 바로 연결되는가

이제 우리 연구로 가져와 보자.

우리 method는 harmful prompt를 그대로 넣되, generation 과정에서 unsafe concept만 줄이고 나머지 의미는 보존하려는 방향이다. 그러면 alignment 평가에서 핵심 질문은 단순히

- “이 이미지가 prompt와 비슷한가?”

가 아니라,

- “이 이미지가 **원래 prompt의 safe한 부분은 여전히 잘 담고 있는가?**”
- “unsafe 부분은 줄어들었는가?”

가 된다.

그래서 우리에게 맞는 VQAScore 사용법은 논문 원형을 약간 변형한 **삼중 비교**다.

### 1) `VQA(erased_image, original_prompt)`
이 값이 너무 높으면 harmful semantics가 여전히 많이 남아 있을 수 있다.

### 2) `VQA(erased_image, anchor_prompt)`
이 값이 높으면, safe rewrite된 의미는 잘 보존했다는 뜻이다.

### 3) `VQA(erased_image, erased_prompt)`
이건 harmful token이 제거된 중립 버전과의 정합성을 보는 보조축이 된다.

### 4) `Gap = VQA(anchor) - VQA(original)`
이 gap이 크면, harmful semantics에서는 멀어지고 safe semantics에는 가까워졌다는 뜻이다.

즉 VQAScore는 우리에게 **safety 판정기**라기보다,

> “우리 방법이 image를 그냥 망가뜨린 것이 아니라, harmful meaning만 골라서 덜어냈는가?”

를 보는 **semantic preservation metric**이다.

이 해석은 현재 repo의 `docs/evaluation_alignment_and_mjbench.md`와도 매우 잘 맞는다.

---

## 1.7 이 논문을 어디까지 믿고 어디서 멈춰야 하는가

이 부분이 중요하다. 사용자가 “요약하지 말고 제대로 분석”을 원했으니 caveat를 분명히 적어야 한다.

### VQAScore의 강점
- CLIPScore보다 compositional prompt에 강함
- open-source라 재현 가능
- 우리 코드에 이미 들어와 있어 integration cost가 낮음
- alignment preservation 논리에는 매우 잘 맞음

### 하지만 VQAScore는 safety metric이 아니다
이건 절대로 혼동하면 안 된다.

VQAScore가 높은 image는 prompt 의미를 잘 담고 있다는 뜻이지, 그 image가 safe하다는 뜻이 아니다. 오히려 harmful prompt를 그대로 잘 그린 unsafe image는 original prompt 기준 VQAScore가 높게 나올 수 있다.

즉,
- VQAScore = **정합성 / 의미 보존**
- safety judge = **안전성 판정**

으로 역할을 분리해야 한다.

### subtle / evasive case에서는 해석이 더 조심스러워야 함
예술적 nude, subtle NSFW, evasive unsafe image는 “prompt를 잘 따랐다”와 “safe하다”가 서로 다른 방향일 수 있다. 이건 우리 연구가 딱 마주치는 어려움이다. 그래서 VQAScore를 main safety metric으로 쓰면 안 되고, Qwen judge나 다른 safety-specific 평가와 반드시 함께 가야 한다.

---

## 1.8 우리 논문에서 어떻게 쓰는 게 가장 좋은가

가장 설득력 있는 문장은 이런 식이다.

- VQAScore는 prompt-image alignment를 평가하기 위한 ECCV 2024의 open-source reference다.
- 우리는 이를 그대로 safety metric으로 쓰는 것이 아니라, harmful prompt / anchor prompt / erased prompt의 상대 비교로 확장해 **semantic preservation**을 본다.
- 따라서 safety reduction과 content preservation을 동시에 분리해서 논할 수 있다.

즉 VQAScore는 우리 evaluation story에서 **alignment half**를 책임지는 논문이다.

---

# 2. MJ-Bench

## 2.1 왜 이 논문이 중요한가: judge도 평가 대상이다

요즘 image generation 논문들은 점점 더 VLM judge나 reward model을 evaluator로 사용한다. 그런데 여기엔 아주 위험한 함정이 있다.

모델을 평가하는 judge를 그냥 가져다 쓰면, 결국 reviewer는 이렇게 묻는다.

> “좋아, 그런데 그 judge는 얼마나 믿을 만한데?”

MJ-Bench의 가장 큰 가치는 바로 이 질문을 정면으로 다룬다는 점이다.

이 논문은 단순히 “더 좋은 image generation metric을 만들자”가 아니라,

> “Multimodal judge 자체를 벤치마크해야 한다.”

는 논문이다.

이건 우리한테 직접 중요하다. 왜냐하면 우리도 Qwen3-VL 같은 multimodal judge를 중심으로 Safe / Partial / Full / NotRelevant 판정을 하려 하기 때문이다. 그래서 MJ-Bench는 method benchmark보다 오히려 **evaluator benchmark**로서 가치가 크다.

---

## 2.2 MJ-Bench가 실제로 무엇을 만들었는가

MJ-Bench는 preference dataset 기반 benchmark다. 기본 형태는 같은 prompt에 대해 두 개의 candidate image를 놓고, 어느 쪽이 더 좋은지 label을 주는 식이다.

이 benchmark는 judge를 다음 4개 축에서 본다.

1. **alignment**
2. **safety**
3. **image quality / artifact**
4. **bias / fairness**

즉 judge가 단순히 “예쁘다/안 예쁘다”만 맞히는지 보는 게 아니라,
- prompt를 잘 따르는지
- 안전한지
- quality artifact를 구분하는지
- bias/fairness 쪽에서 이상하게 치우치지 않는지

를 분해해서 본다.

이 구조 자체가 이미 중요한 메시지다.

> 좋은 multimodal judge는 하나의 scalar로 모든 걸 덮는 존재가 아니라, 여러 evaluation perspective에서 따로 검증돼야 한다.

우리 연구에 바로 연결하면, 우리도 사실 비슷한 구조가 필요하다.
- safety는 safety대로 보고
- alignment는 alignment대로 보고
- content destruction / NotRelevant는 또 따로 보고
- 가능하면 judge reliability 자체도 보고

즉 MJ-Bench는 우리 evaluation 체계를 “과하다”가 아니라 “오히려 현대적이다”라고 정당화해 준다.

---

## 2.3 safety taxonomy가 왜 특히 강력한가

MJ-Bench safety는 특히 유용하다. 프로젝트 페이지 기준 safety는 크게 두 갈래다.

### A. Toxicity
- crime
- shocking
- disgust

### B. NSFW
- **Evident**
- **Subtle**
- **Evasive**

여기서 우리 연구와 가장 직접적으로 연결되는 건 NSFW 3분류다.

### Evident
대놓고 unsafe하고 명확한 경우다. reviewer 입장에서 설명하면 “누가 봐도 explicit한 case”다.

### Subtle
애매하지만 분명 unsafe signal이 남아 있는 경우다. 완전 explicit은 아니지만 그냥 safe라고 하기도 어렵다.

### Evasive
우회적이고 tricky한 경우다. 예술, 고전회화, 은유적 표현, 애매한 framing, 공격을 피하려는 형태 등으로 인해 일반적인 detector가 쉽게 놓칠 수 있는 경우다.

이 세 분류는 우리 쪽 taxonomy와 꽤 잘 대응된다.

- **Full** ↔ Evident
- **Partial** ↔ Subtle
- **adversarial / artistic / evasive unsafe** ↔ Evasive

물론 완전히 1:1은 아니다. 하지만 중요한 건 외부 benchmark가 이미 **unsafe를 단계적으로 나눠서 보는 관점**을 정식으로 채택했다는 사실이다.

이건 reviewer 대응에서 매우 강하다.

왜냐하면 우리도 unsafe를 단순 binary label이 아니라
- 완전히 unsafe
- 부분적으로 unsafe
- safe
- 아예 prompt와 관계없이 망가진 이미지

같이 세분화하고 싶기 때문이다.

MJ-Bench는 최소한 safety severity를 다층적으로 보자는 생각이 학계에서 충분히 자연스럽다는 것을 보여준다.

---

## 2.4 논문이 발견한 핵심 결과를 쉽게 풀면

MJ-Bench가 흥미로운 이유는 단순히 dataset taxonomy 때문만은 아니다. judge family별 강점/약점도 보여준다.

프로젝트 페이지 수준에서 읽히는 핵심 결론은 다음과 같다.

### 1) closed-source VLM judge가 평균적으로 가장 강하다
GPT-4o 같은 모델이 평균적으로 가장 안정적이라는 메시지가 있다.

### 2) 작은 scorer도 alignment/quality에서는 잘할 수 있다
흥미롭게도 alignment나 image quality에서는 CLIP 계열, 작은 scoring model이 꽤 강할 수 있다.

### 3) safety와 bias는 VLM이 강하다
반면 safety와 bias는 reasoning이 필요한 경우가 많아서, open/closed를 막론하고 **VLM judge가 더 강한 경향**이 있다.

이건 우리 연구에 매우 좋은 뉴스다. 우리는 Qwen3-VL 같은 multimodal judge를 safety 판정에 쓰고 있기 때문이다.

즉 reviewer가 “왜 하필 VLM judge냐?”고 하면, MJ-Bench를 근거로 이렇게 말할 수 있다.

> alignment 하나만 보면 작은 scorer가 강할 수 있지만, safety처럼 맥락적 판단이 필요한 영역에서는 VLM judge가 더 적합하다.

### 4) numerical score보다 자연어/Likert형이 더 안정적일 수 있다
이것도 중요한 포인트다. 논문은 judge feedback scale도 실험하는데, 단순 0~10 숫자보다 natural-language나 Likert scale이 더 안정적일 수 있다고 말한다.

우리에게 이게 중요한 이유는, Qwen judge prompting을 설계할 때 무조건 숫자 하나만 뱉게 하는 게 능사가 아닐 수 있다는 뜻이기 때문이다. 실제 논문용 evaluator를 만들 때는
- discrete label
- rationale
- Likert-like scale

사이의 trade-off를 고려할 근거가 된다.

---

## 2.5 우리 연구에 실제로 어떻게 적용할 것인가

여기서 가장 실용적인 포인트는 두 가지다.

### A. Qwen3-VL judge를 검증하는 외부 benchmark로 사용
즉 MJ-Bench는 우리 generated images를 직접 채점하는 메인 metric이 아니라,

> “우리가 메인 evaluator로 쓰는 Qwen3-VL judge를 얼마나 믿어도 되는가?”

를 검증하는 **calibration benchmark**다.

repo에도 이미 `vlm/eval_mjbench_safety.py`가 있다. 즉 이건 단순 related work citation이 아니라, 실제로 돌릴 수 있는 validation 축이다.

논문 구성상으로는 이런 식이 좋다.

1. 먼저 MJ-Bench safety subset에서 Qwen3-VL judge accuracy를 보고
2. 그 다음 main experiment에서는 Qwen3-VL을 안전성 evaluator로 사용
3. appendix에서 subcategory별 강약점까지 공개

이렇게 하면 evaluator 신뢰도에 대한 방어력이 훨씬 올라간다.

### B. 우리 Full / Partial 분류의 외부 reference로 사용
우리 judge taxonomy는 보통
- NotRelevant
- Safe
- Partial
- Full

형태인데, 이 중 safety severity 축인 Full / Partial은 MJ-Bench의 Evident / Subtle / Evasive와 연결해 설명할 수 있다.

중요한 건 reviewer에게

> “우리가 safety를 세 단계 이상으로 나눈 것이 임의적인 규칙이 아니라, external benchmark에서도 graded safety judgment가 일반적이다”

라고 말할 수 있다는 점이다.

---

## 2.6 이 논문을 과대해석하면 안 되는 이유

여기서도 caveat를 분명히 해야 한다.

### MJ-Bench는 judge benchmark이지 method benchmark가 아니다
즉 MJ-Bench에서 judge accuracy가 높다고 해서, 우리 method가 좋아졌다는 뜻은 아니다. 어디까지나 evaluator calibration이다.

### pairwise preference와 single-image classification은 다르다
MJ-Bench는 보통 두 이미지 중 어느 쪽이 더 좋은지 고르는 setting이다. 반면 우리 main evaluation은 single-image에 대해 Full/Partial/Safe/NotRelevant를 부여한다.

즉 pairwise에서 강한 judge가 single-image classification에서도 완전히 같은 수준으로 강하다고 단정하면 안 된다.

### NotRelevant는 MJ-Bench가 직접 다루는 축이 아니다
우리는 unsafe reduction만큼이나 **content destruction**도 중요하다. 즉 사람이 아예 사라지거나 scene이 엉망이 된 이미지에 대해 NotRelevant를 주고 싶다. MJ-Bench는 safety, alignment, quality, bias를 다루지만, 우리 taxonomy의 NotRelevant와 완전히 같은 축은 아니다.

그래서 MJ-Bench는 Full/Partial류 severity-aware safety reasoning에는 좋지만, NotRelevant의 직접 reference로 쓰긴 약하다.

---

## 2.7 우리 논문에서 어떻게 쓰는 게 가장 좋은가

논문 문장 차원에서는 이런 식으로 쓰는 게 가장 적절하다.

- MJ-Bench is a benchmark for multimodal judges, not for image-generation methods directly.
- We use it to calibrate the reliability of our Qwen-based safety judge.
- Its NSFW taxonomy (Evident, Subtle, Evasive) also provides an external precedent for graded safety evaluation beyond binary safe/unsafe labels.

즉 MJ-Bench는 우리 evaluation story에서 **judge-validation half**를 책임지는 논문이다.

---

# 3. LLaVA-Reward

## 3.1 이 논문이 왜 흥미로운가

VQAScore와 MJ-Bench는 각각 alignment metric, judge benchmark에 가깝다. 그런데 LLaVA-Reward는 결이 조금 다르다.

이 논문은 image generation evaluation을 **reward modeling 문제**로 본다.

즉 질문은 이런 식이다.

- alignment를 점수화할 수 있는가?
- fidelity / artifact를 점수화할 수 있는가?
- safety를 점수화할 수 있는가?
- 나아가 그 점수를 generation steering에도 쓸 수 있는가?

이건 우리에게 바로 main evaluation을 대체해주지는 않지만, 아주 중요한 신호를 준다.

> 최신 흐름에서는 multimodal evaluation이 단순 classifier나 heuristic metric을 넘어서, **learned reward model**로 가고 있다.

이건 reviewer에게 “왜 VLM evaluation이 중요한가?”를 설명할 때 매우 유용하다.

---

## 3.2 논문의 핵심 아이디어를 쉽게 풀면

기존 MLLM-based judge는 종종 길고 복잡한 instruction prompt를 넣고, 모델이 자연어 답변을 생성한 뒤, 그 텍스트에서 점수를 해석하는 방식이었다.

이 방식의 문제는:
- 느리고
- prompt engineering에 민감하고
- 점수가 불안정하며
- 아주 미세한 품질 차이를 안정적으로 반영하기 어렵다는 것이다.

LLaVA-Reward는 이걸 바꾼다.

핵심 아이디어는:
- pretrained multimodal LLM의 hidden state를 이용해서
- `(text, image)` 쌍에 대한 **reward**를 직접 출력하자

는 것이다.

즉 “자연어 판정문을 생성하고 그걸 해석하는 judge”가 아니라,
처음부터 **reward head를 가진 평가 모델**을 만들겠다는 방향이다.

---

## 3.3 SkipCA가 왜 필요한가

이 논문에서 구조적으로 가장 눈에 띄는 것은 **Skip-connection Cross Attention (SkipCA)**다.

논문이 지적하는 문제는 decoder-only MLLM에서 깊은 층으로 갈수록 visual token 정보가 약해질 수 있다는 점이다. 그러면 reward를 마지막 hidden state 하나에 linear layer만 붙여 읽어내는 방식은 text-image correlation을 충분히 보지 못할 수 있다.

그래서 LLaVA-Reward는:
- 후반 textual hidden state(EOS 쪽)
- 초반 visual representation

을 cross-attention으로 다시 연결하는 reward head를 둔다.

쉽게 말하면,

> “마지막에 reward를 뽑을 때, 모델이 text만 보지 말고 image도 다시 강하게 보게 하자.”

는 설계다.

우리 관점에서 이게 중요한 이유는, alignment와 safety 모두 사실 **text-image correlation**을 잘 읽어야 하기 때문이다. 예를 들어 safety score도 단순 이미지 분류가 아니라 “이 이미지가 이 prompt와 어떤 관계로 unsafe한가?”를 어느 정도 읽어야 할 수 있다. SkipCA는 이런 multimodal correlation을 강화하려는 장치다.

---

## 3.4 이 논문이 보는 evaluation axes

논문이 훈련하는 perspective는 네 가지다.

1. **text-image alignment**
2. **artifact / fidelity**
3. **safety**
4. **overall ranking / inference-time scaling**

이게 우리에게 주는 메시지는 매우 중요하다.

우리는 현재 평가를
- safety
- alignment
- quality

로 나눠서 보고 있는데, 이 논문은 아예 그것이 하나의 multimodal reward framework 안에서도 가능하다고 보여준다.

즉 reviewer가 “왜 metric이 이렇게 여러 개냐?”고 하면,
사실 최근 learned evaluator들도 alignment, quality, safety를 다 같이 다룬다. 다만 우리는 더 해석 가능하고 robust한 구조를 위해 분해해서 보고 있다고 설명할 수 있다.

---

## 3.5 학습 데이터와 objective를 왜 봐야 하는가

LLaVA-Reward는 paired preference와 unpaired binary data를 모두 쓴다.

논문 본문에서 보이는 학습 데이터 축은 대략 이런 구조다.
- alignment: paired preference
- fidelity/artifact: paired preference
- safety: binary labeled data (UnsafeBench 등)
- ranking: pairwise preference

그리고 objective는 Bradley–Terry ranking loss, classification loss 등 reward modeling에서 익숙한 방식이다.

이게 의미하는 것은:

> 이 모델의 score는 zero-shot heuristic이 아니라, **사람 선호와 annotation에 맞추어 학습된 점수**다.

그래서 장점도 분명하지만, 동시에 위험도 있다. 이건 아래 caveat에서 더 중요하다.

---

## 3.6 논문이 보여주는 실험 메시지

이 논문이 전달하는 실험 메시지는 크게 두 갈래다.

### A. evaluator로서 강하다
alignment, artifact/fidelity, safety 등 여러 축에서 기존 conventional metric이나 일부 MLLM-based baseline보다 human-aligned score를 더 잘 준다고 주장한다.

즉 단순 benchmark용 점수기가 아니라, 진짜 사람 평가와 비슷한 방향의 learned evaluator가 될 수 있다는 것이다.

### B. FK steering 같은 inference-time scaling에도 쓸 수 있다
이 논문은 reward model을 evaluation으로만 쓰지 않고, FK steering 같은 inference-time scaling에 연결한다. 즉 reward가 높은 경로/샘플을 sampling 중에 더 선호하도록 만들 수 있다는 것이다.

우리 연구는 지금 당장 reward-steering 논문은 아니지만, training-free guidance라는 큰 맥락에서는 매우 흥미로운 연결점이다. 미래에는 safety-aware reward model을 이용해 generation trajectory를 steer하는 연구로 확장할 수 있다는 뜻이기 때문이다.

다만 지금 논문에는 safety reward로 FK steering을 본격 수행한 것은 아니라는 점은 선을 그어야 한다.

---

## 3.7 우리 연구에 어떻게 바로 연결되는가

### A. main metric이 아니라 auxiliary metric 후보
우리 현재 stack에서 제일 자연스러운 위치는 이거다.

- main safety decision: Qwen3-VL judge
- main alignment preservation: VQAScore
- auxiliary continuous cross-check: LLaVA-Reward

즉 “이 image가 safe한가 아닌가”를 LLaVA-Reward 하나로 결정하기보다,
- alignment reward
- safety reward
- fidelity/artifact reward

를 보조 score로 읽는 게 맞다.

### B. 박준형이 말한 질문에 잘 맞는다
사용자 메시지의 핵심 중 하나는:

> “기존 prompt에서 얼마나 벗어나지 않으면서 harmful concept과는 얼마나 멀어지는지 더 체계적으로 평가하고 싶다.”

LLaVA-Reward는 바로 이 질문에 강한 시사점을 준다. 왜냐하면 reward model은 본질적으로 여러 목적을 동시에 score로 만들기 때문이다.

우리 식으로 바꾸면,
- original prompt fidelity는 얼마나 남았는가?
- anchor/safe rewrite와는 얼마나 가까운가?
- safety reward는 얼마나 올라갔는가?
- artifact/fidelity는 망가지지 않았는가?

를 연속형 점수로 볼 수 있다.

이건 discrete label만으로는 잘 안 보이는 trade-off를 더 미세하게 보게 해준다.

---

## 3.8 하지만 왜 이걸 main truth로 쓰면 안 되는가

여기서도 caveat가 굉장히 중요하다.

### 1) learned reward model은 결국 학습 데이터 의존적이다
논문도 supplement 차원에서 high-quality training data 부족, source model 의존, reward hacking 가능성을 분명히 인정한다.

즉 이 모델은 보편적 진실판정기가 아니라,
**특정 preference distribution을 학습한 evaluator**다.

### 2) reward hacking 문제
reward model은 generation model이 그 reward를 직접 최적화하기 시작하면 쉽게 exploitation 대상이 될 수 있다. 이건 RLHF, reward modeling 전반에서 늘 있는 문제다.

우리 쪽에서 이를 평가용으로만 쓰더라도, 특정 score가 높다고 해서 반드시 사람이 보기에도 안전하고 좋은 이미지라는 보장은 없다.

### 3) 해석력은 discrete taxonomy보다 떨어질 수 있다
Qwen judge의 Full / Partial / Safe / NotRelevant는 사람이 이해하기 쉽다. 반면 LLaVA-Reward의 scalar reward는 연속형이라는 장점이 있지만, “왜 0.31이 나왔는가?”는 해석이 훨씬 어렵다.

즉 논문 본문에서는 main text보다 appendix/secondary result 쪽이 더 자연스럽다.

---

## 3.9 우리 논문에서 어떻게 쓰는 게 가장 좋은가

가장 좋은 포지셔닝은 다음과 같다.

- LLaVA-Reward is a strong recent example of a multimodal reward model that jointly scores alignment, fidelity, and safety.
- We do not rely on it as a sole evaluator.
- Instead, it motivates a future or auxiliary evaluation direction in which safety and prompt preservation can be scored continuously in addition to our discrete judge outputs.

즉 이 논문은 우리 evaluation story에서 **future-extension / auxiliary-evaluator half**를 책임진다.

---

# 4. 세 논문을 합쳐서 우리 NeurIPS evaluation story를 어떻게 만들 것인가

이제 세 논문을 합쳐서 우리 논문용 narrative를 만들어 보자.

## 4.1 가장 깔끔한 역할 분담

### VQAScore
- 우리가 원래 prompt의 safe 의미를 얼마나 보존했는지 보는 metric
- harmful token 제거 후 semantic preservation 증거

### Qwen3-VL judge
- 생성 이미지 safety를 실제로 분류하는 main evaluator
- Full / Partial / Safe / NotRelevant taxonomy 운용

### MJ-Bench
- Qwen3-VL judge reliability를 검증하는 benchmark
- graded safety taxonomy의 외부 근거

### LLaVA-Reward
- alignment + safety + fidelity를 continuous score로 cross-check하는 보조 evaluator 혹은 future extension

이렇게 두면 서로가 겹치지 않는다.

---

## 4.2 reviewer 입장에서 가장 설득력 있는 포인트

### 질문 1. “왜 NudeNet만으로 안 되나?”
답:
- NudeNet은 body-part detector라서 semantic subtlety를 잘 못 본다.
- 우리는 multimodal safety judge를 추가로 사용한다.
- 그리고 그 judge를 아무 근거 없이 쓰는 게 아니라 MJ-Bench 같은 judge benchmark로 정당화한다.

### 질문 2. “왜 VQAScore를 safety에 쓰나?”
답:
- safety 자체를 VQAScore로 보지 않는다.
- VQAScore는 의미 보존과 prompt faithfulness를 보는 보조 alignment metric이다.
- safety는 Qwen judge/NudeNet 등 safety-specific axis로 따로 본다.

### 질문 3. “왜 label이 binary가 아니고 Full/Partial 같은 중간 단계가 있나?”
답:
- MJ-Bench의 Evident/Subtle/Evasive처럼 graded safety judgment는 이미 외부 benchmark에서 채택된 framing이다.
- subtle / evasive unsafe case를 binary로만 보는 것은 오히려 정보를 버리는 것이다.

### 질문 4. “평가를 더 통합적으로 볼 방법은 없나?”
답:
- 최근엔 LLaVA-Reward 같은 multimodal reward model이 alignment, fidelity, safety를 한 framework 안에서 다루기 시작했다.
- 우리는 현재 해석 가능성을 위해 metric을 분리했지만, future work로 reward-style evaluator 통합이 가능하다.

---

## 4.3 우리 repo 기준 즉시 할 일

1. **VQAScore alignment 결과를 main/ablation 표로 정리**
   - original / anchor / erased prompt
   - gap 포함

2. **MJ-Bench로 Qwen judge calibration**
   - NSFW Evident / Subtle / Evasive breakdown
   - 가능하면 toxicity subset도 같이

3. **LLaVA-Reward는 우선 citation + future/auxiliary note로 반영**
   - 시간이 되면 실제 inference 붙여 score correlation 확인

4. **논문 evaluation section 서술 정리**
   - safety reduction
   - semantic preservation
   - judge validation
   - optional reward-style cross-check

---

## 4.4 최종 결론

세 논문을 다 읽고 나면, 우리 evaluation은 이렇게 정리하는 것이 가장 강하다.

> 우리는 unsafe reduction만 본 것이 아니다.  
> VQAScore로 prompt 의미 보존을 보고,  
> Qwen3-VL로 safety severity를 판정하고,  
> MJ-Bench로 judge reliability를 보강하며,  
> LLaVA-Reward를 통해 future continuous multimodal evaluation의 방향까지 연결한다.

즉 우리 VLM-based evaluation은 단순한 보조 실험이 아니라,
**alignment / safety / evaluator calibration / future reward modeling**을 잇는 꽤 현대적인 평가 프레임으로 정리될 수 있다.

---

## source links
- VQAScore ECCV 2024 poster: https://eccv.ecva.net/virtual/2024/poster/2239
- VQAScore arXiv: https://arxiv.org/abs/2404.01291
- VQAScore repo: https://github.com/linzhiqiu/t2v_metrics
- MJ-Bench project: https://mj-bench.github.io/
- MJ-Bench arXiv: https://arxiv.org/abs/2407.04842
- MJ-Bench dataset: https://huggingface.co/datasets/MJ-Bench/MJ-Bench
- LLaVA-Reward arXiv: https://arxiv.org/abs/2507.21391
- LLaVA-Reward ICCV open access: https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_Multimodal_LLMs_as_Customized_Reward_Models_for_Text-to-Image_Generation_ICCV_2025_paper.html
