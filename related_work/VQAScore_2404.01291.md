# VQAScore: Evaluating Text-to-Visual Generation with Image-to-Text Generation

**논문**: *Evaluating Text-to-Visual Generation with Image-to-Text Generation*  
**학회**: ECCV 2024 poster  
**arXiv**: https://arxiv.org/abs/2404.01291  
**공식 PDF**: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01435.pdf  
**프로젝트/코드**: https://github.com/linzhiqiu/t2v_metrics

---

## 1. 이 논문이 풀려는 문제를 쉽게 말하면

Text-to-image 모델을 평가할 때 가장 흔하게 드는 질문은 이것이다.

> “이 이미지가 정말 prompt가 말한 내용을 잘 보여주고 있는가?”

기존에는 CLIPScore처럼 image embedding과 text embedding의 유사도를 보는 방식이 많이 쓰였다. 그런데 저자들은 이런 embedding similarity 계열 metric이 생각보다 중요한 부분을 자주 놓친다고 본다. 특히 다음 같은 경우다.

- 객체는 맞는데 **관계**가 틀린 경우  
  - 예: “소년이 개를 안고 있다”와 “개가 소년을 안고 있다”는 단어는 비슷하지만 의미는 완전히 다르다.
- 개수나 비교가 중요한 경우  
  - 예: “사과 세 개”, “파란 컵이 빨간 컵보다 크다” 같은 prompt
- 문장이 길고 compositional한 경우  
  - 예: 대상, 속성, 관계, 수량이 한꺼번에 들어간 prompt

저자들의 핵심 문제의식은 간단하다.

> CLIP류 점수는 텍스트를 너무 “단어 묶음”처럼 다루는 경향이 있고, 그래서 고차원적인 compositional meaning을 제대로 평가하지 못한다.

이 논문은 이 문제를 해결하기 위해, text-image alignment를 단순 유사도가 아니라 **VQA(question answering) 문제**로 바꿔서 본다.

---

## 2. 핵심 아이디어: “이 이미지가 이 문장을 정말 보여주나요?”를 직접 묻자

이 논문이 제안하는 VQAScore는 직관적으로 매우 이해하기 쉽다.

prompt가 `t`, 이미지가 `i`일 때, 모델에게 이런 질문을 던진다.

> “Does this figure show '{t}'?”

그리고 VQA 모델이 **Yes**라고 대답할 확률을 alignment score로 쓴다.

즉,

- 입력: `(image, text)`
- 질문: yes/no 질문
- 출력: `P(Yes | image, question)`
- 이 값을 곧 **VQAScore**로 사용

이게 왜 좋으냐면, CLIP처럼 한 번에 전역 벡터 두 개를 비교하는 방식보다, 모델이 훨씬 직접적으로 “이 문장이 이 이미지에 맞는가?”를 판단하도록 만들기 때문이다.

저자들의 관점에서는, alignment는 본질적으로 “문장을 image가 충족하느냐”의 문제이지, “embedding 두 개가 비슷하냐”의 문제가 아니라는 것이다.

---

## 3. 논문이 말하는 기존 metric의 한계

논문은 특히 CLIP 계열 metric의 약점을 세게 짚는다.

### 3.1 bag-of-words 문제

CLIP text encoder는 매우 강력하지만, 평가 상황에서는 관계와 논리를 충분히 반영하지 못할 수 있다. 예를 들어,

- 단어는 다 맞는데
- 관계가 틀리거나
- 수량이 틀리거나
- 비교 방향이 반대인 경우

에도 점수가 너무 높게 나올 수 있다.

즉, “무엇이 등장했는가”는 잡아도, “어떻게 등장했는가”는 약하다는 것이다.

### 3.2 compositional prompt에서 더 심각함

요즘 T2I 모델 평가는 단순한 단어 수준 prompt보다,

- 두 객체의 상호작용
- 방향성
- 논리적 제약
- 개수
- 비교/구별

이 들어간 prompt가 훨씬 중요하다.

논문은 바로 이 영역에서 VQAScore가 강하다고 주장한다. 왜냐하면 question answering 형식이 compositional semantics를 더 직접적으로 건드리기 때문이다.

---

## 4. 실제 방법론을 조금 더 자세히

논문은 “질문을 잘 만들고, 그 질문에 대한 yes/no 판단을 안정적으로 읽어내는 것”을 핵심으로 둔다.

### 4.1 질문 템플릿

기본 템플릿은 매우 단순하다.

- `Does this figure show '{text}'?`

이 단순함이 중요하다. prompt를 다시 복잡하게 설명하거나 chain-of-thought를 요구하지 않고, 최소한의 yes/no 질문으로 정리한다.

### 4.2 score 계산

모델이 생성한 첫 토큰에서

- “yes” 확률
- “no” 확률

을 읽어내고, 그중 yes 확률을 점수로 사용한다.

이 점수는 0~1로 해석할 수 있다.

- 1에 가까울수록: 이 image가 해당 prompt를 잘 만족한다
- 0에 가까울수록: 이 image가 prompt를 거의 만족하지 않는다

### 4.3 open-source 모델과 자체 학습 모델

논문은 두 층위를 보여준다.

1. **off-the-shelf VQA 모델**로도 VQAScore를 만들 수 있다  
   - 즉 이 아이디어 자체가 강하다는 뜻
2. 여기에 더해 저자들은 **CLIP-FlanT5** 기반의 자체 VQA 모델도 제안한다  
   - 이 버전이 여러 benchmark에서 가장 강한 성능을 낸다

이 점이 우리에게 중요하다. 즉, VQAScore는 “특정 proprietary model 없이는 못 쓰는 metric”이 아니라, 철학 자체를 우리 evaluation에 가져올 수 있다는 뜻이다.

---

## 5. 논문이 같이 제안한 GenAI-Bench는 왜 중요한가

이 논문은 metric만 내놓은 것이 아니라, 그 metric을 검증할 benchmark도 함께 만든다. 그것이 **GenAI-Bench**다.

### 5.1 왜 새 benchmark가 필요했는가

저자들은 기존 benchmark가 다음 문제가 있다고 본다.

- prompt가 너무 쉬움
- compositional reasoning이 부족함
- prompt가 애매하거나 노이즈가 있음
- skill tag가 충분히 정리되어 있지 않음

### 5.2 GenAI-Bench의 특징

논문 기준으로 GenAI-Bench는

- **1,600개의 compositional prompt**
- **15,000+ human ratings**
- 5,000개가 넘는 human-verified skill tag

를 포함한다.

그리고 이 benchmark는 기존 SOTA text-to-image 모델도 어려워하는 고차원 prompt를 포함한다.

예를 들어,

- counting
- comparison
- differentiation
- logic

같은 higher-order skill이 들어간다.

즉 이 benchmark는 “alignment를 정말 제대로 보려면 얼마나 어려운 문제를 던져야 하는가”를 보여준다.

---

## 6. 실험 결과를 어떻게 읽어야 하는가

논문 표에서 핵심 메시지는 아주 분명하다.

- CLIPScore, BLIP류보다 VQAScore가 여러 benchmark에서 더 강함
- open-source VQA 모델로도 강한 성능
- 저자들의 **CLIP-FlanT5 기반 VQAScore**가 가장 강한 경우가 많음

중요한 건 단순히 “점수가 높다”가 아니다. 논문의 핵심은,

> compositional prompt일수록 VQA 기반 방식이 더 잘 맞는다

는 점이다.

이 메시지는 우리 연구와 잘 연결된다. 우리도 단순히 “nude”라는 단어가 있는지 없는지를 넘어서,

- 원래 prompt 의미는 얼마나 유지되는지
- harmful concept만 선택적으로 멀어졌는지

를 봐야 하기 때문이다.

---

## 7. 우리 연구에 적용하면 정확히 무엇이 좋은가

이 부분이 가장 중요하다.

### 7.1 우리 repo에서 이미 하고 있는 일과 맞닿아 있음

이 repo에는 이미

- `vlm/eval_vqascore.py`
- `vlm/eval_vqascore_alignment.py`

가 있다.

즉 우리는 이미 이 논문의 철학과 매우 가까운 evaluation을 하고 있다.

특히 우리 세팅에서 유용한 건 **삼중 비교**다.

1. `VQA(generated_image, original harmful prompt)`  
   - 낮아질수록 harmful meaning이 줄었다는 뜻
2. `VQA(generated_image, anchor prompt)`  
   - 높을수록 safe rewrite 의미를 잘 보존했다는 뜻
3. `VQA(generated_image, erased prompt)`  
   - harmful token만 지운 의미를 어느 정도 보존했는지 볼 수 있음

이렇게 하면 단순히 “safe해졌냐”가 아니라,

> “원래 prompt에서 harmful 부분만 빠지고 나머지 의미는 유지됐는가?”

를 더 정교하게 볼 수 있다.

### 7.2 reviewer 대응에 좋음

NeurIPS 리뷰어가 물을 수 있는 질문은 보통 이렇다.

- “안전해졌는데 prompt fidelity는 깨진 것 아닌가?”
- “그냥 사람을 지워서 safe해진 것 아닌가?”
- “배경이나 포즈까지 다 바꿔놓고 safe라고 하는 것 아닌가?”

여기에 VQAScore는 매우 좋은 답을 준다.

- original prompt score는 내려가고
- anchor prompt score는 올라가야 하며
- erased prompt score는 적절히 유지되어야 한다

이 구조는 **selective erasing**을 증명하는 보조 지표로 매우 좋다.

---

## 8. 하지만 이 논문을 safety 평가의 근거로 과하게 쓰면 안 되는 이유

여기서 선을 잘 그어야 한다.

### 8.1 VQAScore는 safety metric이 아니다

이 논문은 어디까지나 **alignment metric** 논문이다.

즉 VQAScore가 높다고 해서
- 이미지가 safe하다는 뜻은 아니고,
- VQAScore가 낮다고 해서 harmful concept이 잘 제거되었다는 것도 자동으로 성립하지 않는다.

예를 들어,
- 이미지가 완전히 망가져도 original prompt score는 낮아질 수 있다.
- subtle NSFW나 evasive case에서 alignment와 safety는 다른 축일 수 있다.

### 8.2 그래서 우리 논문에서의 적절한 역할은

정확히는 이렇게 두는 게 맞다.

- **Safety의 주 판정자**: Qwen3-VL judge, NudeNet, SigLIP2 등
- **Semantic preservation의 보조 핵심 metric**: VQAScore

즉 VQAScore는 “이 이미지가 얼마나 안전한가”를 직접 재는 도구가 아니라,

> “이 이미지가 원래 prompt/anchor rewrite와 어떤 관계를 맺고 있는가”

를 보는 도구다.

---

## 9. 우리 논문에서 어떻게 쓰는 게 가장 설득력 있는가

가장 좋은 framing은 이렇다.

> VQAScore는 prompt-image alignment를 평가하기 위한 강한 open-source 기준선이다. 우리는 이를 harmful/original, anchor/safe, erased prompt에 대해 각각 계산함으로써, 안전성 개입 이후에도 benign semantics가 얼마나 유지되는지를 정량화한다.

즉 이 논문을 통해 우리는

- VLM 기반 alignment 평가를 쓰는 근거
- CLIPScore보다 더 compositional한 평가를 쓴다는 근거
- safe/anchor alignment를 별도로 재는 정당성

을 확보할 수 있다.

---

## 10. 한 줄 결론

VQAScore는 우리 연구에서 **“안전성 그 자체”의 판정기**가 아니라,

> **harmful concept을 줄이면서도 원래 prompt 의미를 얼마나 보존했는지 보여주는 가장 설득력 있는 alignment 보조 지표**

로 쓰는 것이 맞다.
