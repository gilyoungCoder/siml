# MJ-Bench: Is Your Multimodal Reward Model Really a Good Judge for Text-to-Image Generation?

**논문**: *MJ-Bench: Is Your Multimodal Reward Model Really a Good Judge for Text-to-Image Generation?*  
**학회**: NeurIPS 2025 poster  
**arXiv**: https://arxiv.org/abs/2407.04842  
**프로젝트 페이지**: https://mj-bench.github.io/  
**데이터셋**: https://huggingface.co/datasets/MJ-Bench/MJ-Bench

---

## 1. 이 논문이 왜 중요한가: “judge를 평가해야 한다”는 문제 제기

우리처럼 diffusion safety 연구를 할 때 요즘은 멀티모달 judge를 너무 자연스럽게 쓴다.

예를 들어,
- 이 이미지가 안전한가?
- prompt를 잘 따랐는가?
- 더 나은 이미지가 어느 쪽인가?

같은 질문을 VLM이나 reward model에게 묻는다.

그런데 이 논문은 아주 중요한 질문을 던진다.

> “그 judge가 정말 믿을 만한가?”

즉, 생성 모델을 평가하는 도구로 judge를 쓰는 순간,
생성 모델만 평가하면 끝나는 게 아니라 **judge 자체도 평가 대상**이 된다는 것이다.

이게 MJ-Bench의 출발점이다.

저자들의 문제의식은 매우 현실적이다.

- judge가 alignment를 잘 보는지
- safety를 잘 보는지
- bias를 잘 보는지
- quality/artifact를 잘 보는지

는 모두 다를 수 있다.

그리고 judge가 그걸 못 보는데도 그 feedback으로 모델을 fine-tune하면, 오히려 잘못된 방향으로 alignment될 수 있다. 즉 **judge misalignment가 downstream model misalignment로 이어질 수 있다**는 것이 이 논문의 핵심 배경이다.

---

## 2. MJ-Bench는 무엇을 만들었는가

MJ-Bench는 단순한 이미지 생성 benchmark가 아니다. 이것은 **multimodal judge benchmark**다.

즉, dataset 자체의 목적이

- “어떤 생성 모델이 제일 좋은가?”

가 아니라,

- “어떤 judge가 어떤 축에서 믿을 만한 피드백을 주는가?”

를 보는 데 있다.

### 2.1 기본 구조

프로젝트 페이지 설명 기준으로 MJ-Bench는 **preference dataset**이다.
각 샘플은 대체로:

- 같은 prompt
- 두 장의 이미지
- 어느 쪽이 더 바람직한지에 대한 label

로 구성된다.

즉 judge는 두 이미지를 보고,
- alignment 측면에서 어느 쪽이 나은지,
- safety 측면에서 어느 쪽이 안전한지,
- quality/bias 측면에서 어느 쪽이 더 적절한지

판단해야 한다.

### 2.2 왜 pairwise가 좋은가

pairwise preference 구조는 실전과 잘 맞는다.
왜냐하면 많은 경우 absolute score보다,

- A와 B 중 어느 쪽이 더 낫다

는 판단이 더 안정적이기 때문이다.

이 방식은 reward model과 judge model을 평가하기에도 적합하다. 왜냐하면 실제 fine-tuning이나 alignment에서도 preference signal이 자주 쓰이기 때문이다.

---

## 3. 이 benchmark가 보는 네 가지 큰 축

프로젝트 페이지 기준으로 MJ-Bench는 judge를 네 가지 큰 perspective에서 본다.

1. **text-image alignment**  
2. **safety**  
3. **image quality and artifacts**  
4. **bias and fairness**

이 점이 중요하다. 즉 이 논문은 “좋은 judge”를 단일 점수로 정의하지 않는다.

judge는 축마다 다르게 강하고 약할 수 있다.
예를 들어,
- alignment는 잘 보지만 safety는 못 볼 수 있고,
- safety는 잘 보지만 fine-grained quality 비교는 못 할 수 있다.

이 관점이 매우 중요하다. 우리도 evaluator를 단일 마법 상자로 생각하면 안 된다는 뜻이다.

---

## 4. safety 축이 우리 연구와 직접 연결되는 이유

우리 입장에서 MJ-Bench가 특히 중요한 건 **safety taxonomy**다.

### 4.1 Toxicity subcategories

프로젝트 페이지와 repo 내 기존 evaluation note를 종합하면 safety는 크게 두 갈래다.

#### A. Toxicity
- crime
- shocking
- disgust

이건 향후 multi-concept 확장과 직접 맞닿아 있다.
특히:
- violence
- shocking
- illegal activity

같은 카테고리와 논리적으로 잘 연결된다.

### 4.2 NSFW subcategories

우리 연구에 더 직접적인 건 NSFW 쪽이다.

- **Evident**
- **Subtle**
- **Evasive**

이 세 분류는 아주 중요하다.

#### Evident
누가 봐도 노골적인 NSFW다.  
우리 taxonomy로 치면 **Full**에 가장 가깝다.

#### Subtle
대놓고 explicit하지는 않지만 unsafe signal이 분명하다.  
우리의 **Partial**과 매우 가까운 개념이다.

#### Evasive
고전미술, 예술적 우회, 애매한 표현, 또는 직접적 단어를 피하는 방식처럼 **판별이 더 어려운 NSFW**다.  
이건 Ring-A-Bell이나 우회 프롬프트를 다루는 우리 setting과 잘 맞는다.

즉 MJ-Bench는 단순히 “안전/위험” 이분법이 아니라,

> **안전성도 난이도와 표현 정도에 따라 단계적으로 봐야 한다**

는 점을 공식 benchmark 차원에서 보여준다.

---

## 5. 논문이 보여주는 핵심 empirical message

프로젝트 페이지 abstract가 전하는 핵심은 몇 가지로 정리된다.

### 5.1 closed-source VLM이 평균적으로 강하다

전반적으로 GPT-4o 같은 closed-source VLM이 평균적으로 가장 강한 judge로 나온다.

이건 두 가지를 말해준다.

1. multimodal judge 자체는 실제로 강력할 수 있다.
2. 하지만 open-source judge는 여전히 calibration이 필요하다.

### 5.2 작은 scoring model과 VLM이 잘하는 축이 다르다

이 논문은 매우 흥미로운 관찰을 한다.

- alignment / image quality 쪽은 작은 CLIP-style scorer가 더 잘할 수 있음
- safety / bias 쪽은 reasoning이 강한 VLM이 더 잘할 수 있음

이건 우리에게 매우 중요하다. 왜냐하면 우리는 이미

- alignment 쪽은 VQAScore류
- safety 쪽은 Qwen3-VL judge

로 사실상 분업하고 있기 때문이다.

즉 이 논문은 우리 현재 구조가 ad-hoc한 것이 아니라,
**judge family마다 잘 보는 축이 다르다**는 최근 benchmark 결과와 맞닿아 있다고 볼 수 있다.

### 5.3 feedback scale도 중요하다

논문은 numeric scale보다 **자연어 Likert-style feedback**이 더 안정적인 경우가 많다고 본다.

이것도 중요한 포인트다. 왜냐하면 safety는 종종 미묘하고 graded한 문제라서,
단순 점수 하나보다 설명적/서술적 판단이 더 일관적일 수 있기 때문이다.

우리 Qwen judge도 결국 discrete label이지만, 이 논문을 통해 “judge prompting design 자체도 중요한 연구 변수”라는 점을 인식할 수 있다.

---

## 6. 우리 연구에서 정확히 어떻게 써야 하는가

이 부분이 핵심이다.

### 6.1 Qwen3-VL judge를 정당화하는 근거

우리 repo는 safety evaluation에서 Qwen3-VL judge를 쓰고 있다. 그런데 리뷰어 입장에서는 이런 질문이 나올 수 있다.

- “왜 이 judge를 믿어야 하죠?”
- “NudeNet 말고 VLM judge를 쓰는 이유가 뭐죠?”
- “Partial / Full 구분은 임의적인 것 아닌가요?”

여기에 MJ-Bench는 아주 좋은 답을 준다.

가장 중요한 논리는 다음과 같다.

1. **multimodal judge를 쓰는 것 자체는 최신 흐름이다**  
2. 하지만 judge는 따로 검증해야 한다  
3. safety는 binary가 아니라 graded하게 보는 것이 더 타당하다  
4. 따라서 우리도 Qwen judge를 쓰되, MJ-Bench 같은 외부 benchmark를 reference로 calibration해야 한다

즉 MJ-Bench는 우리 judge를 “증명”해주는 것이 아니라,

> **judge validation이 필요하다는 프레임 자체를 정당화해준다**

### 6.2 Full / Partial 분류를 정당화하는 외부 reference

우리 현재 분류는 대략:
- NotRelevant
- Safe
- Partial
- Full

이다.

이 중 safety severity 축은 MJ-Bench의
- Evident
- Subtle
- Evasive

와 구조적으로 잘 연결된다.

정확히 1:1 대응은 아니지만,

- Evident → Full
- Subtle → Partial
- Evasive → Partial과 difficult cases 사이

처럼 읽을 수 있다.

이것은 논문 서술에서 매우 중요하다. 왜냐하면 reviewer가 “왜 굳이 Full/Partial로 나누냐?”라고 물을 때,

> “최근 multimodal judge benchmark에서도 NSFW를 explicit / subtle / evasive처럼 graded하게 본다”

라고 답할 수 있기 때문이다.

### 6.3 NotRelevant는 MJ-Bench가 직접 주지 않는다

이건 중요한 caveat다.

우리에게는 **content destruction**이나 **semantic collapse**를 의미하는 `NotRelevant`가 매우 중요하다. 그런데 MJ-Bench는 주로 preference / safety / quality / bias 관점이지, 우리 식의 “이 이미지가 아예 프롬프트와 무관해졌다”를 그대로 주지는 않는다.

즉 MJ-Bench는
- safety severity justification에는 좋지만,
- NotRelevant 자체의 직접적 외부 reference는 아니다.

그래서 이 부분은 여전히
- VQAScore alignment
- prompt fidelity score
- human inspection

으로 보완하는 것이 맞다.

---

## 7. 하지만 이 논문을 과하게 해석하면 안 되는 이유

### 7.1 MJ-Bench는 judge benchmark이지, 우리 method metric 그 자체는 아니다

이 논문은 우리의 생성 결과를 직접 채점해주는 metric을 제안한 것이 아니다. judge들을 비교하는 benchmark다.

즉 이 논문을 근거로
- “우리 Qwen judge가 곧 ground truth다”
- “MJ-Bench가 있으니 human eval은 필요 없다”

라고 말하면 과장이다.

정확한 사용법은:

> 우리 judge의 reliability를 논의할 때 참고하는 external benchmark

이다.

### 7.2 safety category 전체와 nudity-only setting은 다르다

MJ-Bench safety는 NSFW만이 아니라 toxicity, bias와도 연결된다. 즉 우리 nudity-centric phase 1과 완전히 일치하지는 않는다.

그래서 이 논문을 쓸 때는

- “direct equivalence”가 아니라
- “graded multimodal safety evaluation의 선례”

로 쓰는 것이 정직하다.

---

## 8. 우리 repo에서 실제로 할 수 있는 액션

이 repo에는 이미 `vlm/eval_mjbench_safety.py`가 있다. 이건 아주 중요하다.

즉 우리는 citation만 하는 것이 아니라,
실제로 다음을 할 수 있다.

1. MJ-Bench safety subset에 대해 Qwen3-VL judge 평가
2. overall accuracy뿐 아니라 subcategory breakdown 확인
3. Qwen judge가
   - Evident에는 강한지
   - Subtle/Evasive에서 약한지
   를 보고 limitation으로 정리

이렇게 하면 논문에서 훨씬 설득력이 생긴다.

---

## 9. 우리 논문에서 가장 좋은 포지셔닝

가장 좋은 문장은 이런 방향이다.

> MJ-Bench shows that multimodal judges should be evaluated as judges, not assumed to be perfect oracles. It also supports a graded view of safety through NSFW categories such as Evident, Subtle, and Evasive. We therefore use a multimodal judge for safety evaluation, but frame it as a calibrated, severity-aware evaluator rather than literal ground truth.

한국어로 풀면:

- 우리는 VLM judge를 쓰되,
- 그걸 절대적인 진실로 두지 않고,
- graded safety evaluator로 위치시키며,
- 외부 benchmark(MJ-Bench)로 그 사용 방식을 정당화한다

는 것이다.

---

## 10. 한 줄 결론

MJ-Bench는 우리에게 **“Qwen judge를 써도 된다”**를 직접 보장해주는 논문이 아니라,

> **multimodal judge는 반드시 judge로서 검증되어야 하고, safety도 explicit/subtle/evasive처럼 단계적으로 봐야 한다는 점을 정당화해주는 가장 강한 외부 benchmark**

라고 이해하는 것이 가장 정확하다.
