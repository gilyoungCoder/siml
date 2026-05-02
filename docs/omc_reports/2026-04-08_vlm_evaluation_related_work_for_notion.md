# VLM 기반 Evaluation Related Work 상세 분석

> 작성일: 2026-04-08  
> 범위: **primary source만** 보고 정리한 detailed memo  
> 대상 논문:
> 1. **Evaluating Text-to-Visual Generation with Image-to-Text Generation** (ECCV 2024, VQAScore)
> 2. **MJ-Bench: Is Your Multimodal Reward Model Really a Good Judge for Text-to-Image Generation?** (NeurIPS 2025)
> 3. **Multimodal LLMs as Customized Reward Models for Text-to-Image Generation** (ICCV 2025, LLaVA-Reward)

---

## 왜 이 세 논문을 지금 깊게 봐야 하는가

우리 repo의 evaluation은 이미 상당 부분 **VLM 기반**이다.

- `Qwen3-VL` → safety judge
- `VQAScore` 계열 → prompt-image alignment / faithfulness
- `NudeNet` / `SigLIP2` / `FID/CLIP` → 보조 검증

문제는 이걸 논문에 쓸 때 reviewer가 보통 이렇게 묻는다는 점이다.

1. **왜 VLM-based evaluation을 믿어도 되는가?**
2. **왜 alignment와 safety를 따로 보았는가?**
3. **왜 Full / Partial 같은 graded safety judgment가 필요한가?**
4. **왜 CLIPScore 같은 단순 similarity로 충분하지 않은가?**

이 세 논문은 정확히 이 질문들에 답해준다.

- **VQAScore**는 “alignment를 CLIP cosine 하나로 보면 부족하다”는 근거를 준다.
- **MJ-Bench**는 “judge 자체를 검증해야 한다”는 근거를 준다.
- **LLaVA-Reward**는 “alignment / fidelity / safety를 human-aligned score로 함께 볼 수 있다”는 최신 흐름을 보여준다.

즉, 이 세 편을 함께 읽으면 우리 평가는 단순 보조 실험이 아니라:

> **alignment 보존 + safety 판정 + judge 신뢰도 + future reward-style 평가 확장**

을 모두 갖춘 구조라는 논리를 만들 수 있다.

---

# 1. VQAScore

## 1.1 한 줄로 먼저 설명하면

이 논문은

> **“이미지와 prompt가 잘 맞는지 보려면 CLIP similarity보다, VQA 모델에게 ‘이 이미지가 이 문장을 정말 보여주나요?’라고 직접 묻는 게 더 낫다”**

는 주장이다.

즉 text-image alignment를 **question answering 문제**로 바꿔버린다.

---

## 1.2 문제의식: 왜 CLIPScore로는 부족한가

논문이 지적하는 핵심은 CLIP text encoder가 종종 **bag-of-words**처럼 동작한다는 점이다.

예를 들어:
- “the horse is eating the grass”
- “the grass is eating the horse”

같은 문장을 CLIP은 충분히 잘 구분하지 못할 수 있다.

왜냐하면 이런 문장 차이는 단순 단어 집합 차이가 아니라:
- 주체가 누구인지
- 대상이 누구인지
- 관계가 어떻게 걸리는지

를 이해해야 하기 때문이다.

이 문제는 text-to-image generation 평가에서 치명적이다.  
요즘 생성 모델은 얼핏 보기엔 그럴듯한 이미지를 잘 만들기 때문에,
**“이미지가 예쁘다”**와 **“prompt를 정확히 만족한다”**를 분리해서 봐야 한다.

CLIP cosine은 첫 번째에는 어느 정도 도움이 되지만, 두 번째에는 자주 약하다.

논문은 이걸 다음처럼 표현한다:
- object composition
- attribute binding
- relation
- counting
- logical reasoning

같은 **compositional prompt**에서 기존 metric이 약하다는 것이다.

---

## 1.3 핵심 아이디어: alignment를 yes/no 확률로 본다

논문의 핵심 수식은 매우 직관적이다.

이미지 `I`와 텍스트 `T`가 있을 때,
다음 질문을 만든다.

> “Does this figure show ‘{T}’? Please answer yes or no.”

그리고 VQA 모델이 **Yes라고 답할 확률**을 score로 쓴다.

즉,

- 입력: `(image, text)`
- text를 question으로 변환
- 모델이 Yes를 낼 확률 계산
- 그 확률이 바로 **VQAScore**

이 아이디어의 장점은 단순하다.

CLIP처럼 “벡터 두 개가 비슷한가?”를 묻는 대신,
모델에게 아예

> “이 문장이 이미지에서 성립하니?”

를 묻게 된다.

그래서 object / relation / attribute / counting처럼 **명시적 판단**이 필요한 구조에 더 잘 맞는다.

---

## 1.4 이 논문이 단순히 ‘질문 하나 던진다’에서 끝나지 않는 이유

겉보기엔 단순하지만 논문은 중요한 기술적 포인트를 강조한다.

### (a) answer token의 생성 가능도 자체를 score로 써야 한다

많은 prior VQA 방식은 모델이 생성한 답변 텍스트만 보고 판단하거나,
prompt를 decomposition해서 여러 하위 질문으로 나눈 뒤 aggregate한다.

그런데 이 논문은 **첫 답변 토큰의 likelihood**, 특히 Yes의 확률을 직접 쓰는 것이 훨씬 깔끔하다고 본다.

이렇게 하면:
- score가 바로 연속값이 되고
- question decomposition 과정에서 생기는 이상한 하위 질문들을 피할 수 있고
- complex prompt를 end-to-end로 평가할 수 있다.

### (b) image-question encoder가 매우 중요하다

논문이 특히 강조하는 부분이다.

많은 open-source VQA 모델(LLaVA 류)은 decoder-only 구조라서,
질문 토큰이 뒤에서 이미지 토큰을 읽는 식이지,
이미지 representation이 **질문에 따라 다시 달라지지 않는** 경우가 많다.

논문은 alignment 평가에서는 이게 불리하다고 본다.

왜냐하면 사람도:
- “색깔을 보라”는 질문을 받으면 색을 다르게 보고
- “누가 누구를 쫓는가?”라는 질문을 받으면 relation을 다르게 보기 때문이다.

그래서 논문은 **bidirectional image-question encoder**가 더 좋다고 주장하고,
그 구현으로 **CLIP vision encoder + FlanT5** 조합을 사용한다.

즉 image와 question이 서로 영향을 주도록 만든다.

이게 논문이 말하는 **CLIP-FlanT5**의 핵심 장점이다.

---

## 1.5 benchmark 설계: 왜 GenAI-Bench를 같이 만들었는가

이 논문은 metric만 제안한 게 아니라 **GenAI-Bench**도 함께 만든다.

이유는 간단하다.
기존 benchmark가 너무 쉬우면,
metric이 좋은지 나쁜지 분간이 잘 안 되기 때문이다.

논문은 이렇게 주장한다:
- 기존 benchmark들은 실제 user prompt의 compositional difficulty를 충분히 담지 못한다.
- 그래서 object, scene, attribute, relation뿐 아니라,
  - comparison
  - differentiation
  - logical reasoning
  - counting
  같은 harder skill까지 포함하는 benchmark가 필요하다.

그래서 만들었다는 것이 **GenAI-Bench**고,
primary source 기준 핵심 수치는:
- **1,600 compositional prompts**
- **15,000+ human ratings**

이다.

이 점이 중요한 이유는,
VQAScore가 단순히 저자들이 만든 특수한 예시에서만 잘 되는 게 아니라,
사람 평가와 비교 가능한 더 큰 benchmark 위에서 검증되었다는 뜻이기 때문이다.

---

## 1.6 실험 결과를 어떻게 읽어야 하는가

논문이 강조하는 실험 메시지는 크게 세 가지다.

### 1) off-the-shelf VQA 모델만 써도 강하다

즉 이 아이디어 자체가 이미 강력하다.
VQA-style scoring이라는 framing이 metric improvement의 큰 부분을 차지한다.

### 2) CLIP-FlanT5는 더 강하다

질문-이미지 상호작용을 양방향으로 만들고,
FlanT5의 reasoning 성질을 활용한 설계가 성능을 더 끌어올린다.

### 3) GPT-4V류 strong baseline과도 경쟁하거나 넘는다

이건 논문이 꽤 강하게 미는 포인트다.
오픈소스 기반 CLIP-FlanT5가 proprietary system까지 능가하는 구간이 있다는 것이다.

즉 이 논문은 “VLM으로 alignment를 보는 발상” 자체뿐 아니라,
**open-source로도 충분히 강한 alignment metric을 만들 수 있다**는 자신감도 준다.

---

## 1.7 우리 연구와 연결: 무엇을 직접 정당화해주는가

이 논문이 우리 repo에서 가장 강하게 정당화해주는 것은 **alignment preservation**이다.

우리가 정말 알고 싶은 것은:

> “harmful prompt를 넣었을 때 safety intervention 이후 이미지가 원래 장면 의미를 얼마나 유지했는가?”

이다.

이 질문에는 CLIP cosine보다 VQAScore가 훨씬 잘 맞는다.

특히 우리처럼:
- original harmful prompt
- anchor prompt
- erased prompt

를 따로 둘 수 있으면, VQAScore는 매우 강해진다.

### 우리 setting에 맞는 해석

1. `VQA(image, original_prompt)`
   - 높으면 harmful semantics가 여전히 남았을 가능성
2. `VQA(image, anchor_prompt)`
   - 높으면 safe rewrite 의미를 잘 보존한 것
3. `VQA(image, erased_prompt)`
   - harmful token 제거 후 핵심 장면 의미가 남는지 확인
4. `Gap = anchor - original`
   - 클수록 selective erasing에 유리

즉 VQAScore는 우리에게
**“안전성 자체”**를 주는 metric이 아니라,

> **“얼마나 prompt 의미를 덜 망가뜨리면서 harmful concept만 밀어냈는가”**

를 보는 metric이다.

---

## 1.8 이 논문을 우리 safety metric의 근거로 바로 쓰면 안 되는 이유

여기서 조심해야 한다.

VQAScore는 **safety-specific metric이 아니다.**

즉:
- nudity severity를 직접 분류해주지 않는다.
- violence / toxicity / bias의 final ground-truth judge로 쓰면 안 된다.
- “이미지가 안전한가?”보다는 “이 문장이 이미지에 맞는가?”에 가깝다.

예를 들어 고전회화식 누드나 subtle NSFW의 경우,
원 prompt와 잘 맞는다는 사실이 곧 안전하다는 뜻은 아니다.

그래서 우리 논문에서는 VQAScore를
- **safety의 주 metric**으로 두면 안 되고,
- **alignment / content preservation metric**으로 둬야 한다.

이 구분을 분명히 해야 reviewer에게 덜 물린다.

---

## 1.9 우리 논문에서 가장 좋은 포지셔닝

가장 자연스러운 문장은 이렇다.

> VQAScore는 compositional prompt-image alignment를 평가하기 위한 강한 open-source 기준이며, 우리는 이를 original / anchor / erased prompt 비교로 확장하여 safety intervention 이후 semantic preservation을 정량화한다.

즉,
- safety는 다른 judge가 맡고
- VQAScore는 **보존/정렬(alignment)**를 맡는 구조가 가장 좋다.

---

# 2. MJ-Bench

## 2.1 한 줄로 먼저 설명하면

이 논문은

> **“이미지 생성 모델을 평가하는 judge를 쓰려면, 그 judge가 정말 judge로서 믿을 만한지 먼저 평가해야 한다”**

는 논문이다.

즉 MJ-Bench는 image generation benchmark가 아니라,
**multimodal judge benchmark**다.

---

## 2.2 왜 judge benchmark가 중요한가

최근 T2I에서는 judge가 여러 군데 쓰인다.

- inference-time guidance
- preference fine-tuning
- reward modeling
- automatic evaluation

문제는 judge가 부정확하면,
그 judge로 fine-tune한 모델도 같이 잘못된 방향으로 갈 수 있다는 점이다.

즉 judge 오류는 단순 evaluation error가 아니라:
- misalignment
- unsafe fine-tuning
- biased optimization

으로 이어진다.

논문은 이걸 정면으로 문제 삼는다.

특히 다음 judge family를 비교한다.
- **CLIP-based scoring models**
- **open-source VLMs**
- **closed-source VLMs**

그리고 질문은 단순하다.

> “누가 alignment에 강하고, 누가 safety에 강하고, 누가 bias에 강한가?”

---

## 2.3 dataset 구조: 왜 pairwise preference를 썼는가

MJ-Bench의 각 데이터 포인트는 기본적으로:
- instruction `I`
- chosen image `M_p`
- rejected image `M_n`

의 triplet이다.

즉 absolute score를 직접 주는 대신,
**두 이미지 중 어느 쪽이 더 나은지 / 더 안전한지 / 더 aligned한지**가 명확한 pair를 만든다.

이게 중요한 이유는,
judge benchmarking에서는 “정답이 명확한 비교 상황”을 만드는 것이 훨씬 안정적이기 때문이다.

논문은 여기서 두 입력 방식도 비교한다.

### single-input judge
- 이미지 하나씩 따로 scoring
- 점수 차이와 threshold로 preference 추론

### multi-input judge
- 두 이미지를 같이 보고 analyze-then-judge
- pairwise judgment을 직접 내림

이 구분은 우리에게도 중요하다.
현재 repo의 Qwen safety judge는 주로 **single-image classification**에 가깝기 때문이다.

---

## 2.4 MJ-Bench의 네 축

논문이 benchmark를 설계할 때 잡은 네 축은:

1. **text-image alignment**
2. **safety**
3. **image quality & artifact**
4. **bias & fairness**

즉 judge를 하나의 평균 점수로만 보지 않고,
각 축에서 **어디가 강하고 어디가 약한지 분해해서 본다.**

이게 매우 중요하다.
왜냐하면 논문 결론도 결국:
- alignment는 작은 scoring model이 더 잘하는 구간이 있고
- safety / bias는 VLM judge가 더 잘하는 구간이 있다

는 식으로 **축별로 결론이 다르기 때문**이다.

---

## 2.5 safety 축이 우리에게 특히 중요한 이유

MJ-Bench safety는 두 갈래로 나뉜다.

### A. Toxicity
- crime
- shocking
- disgust

### B. NSFW
- **Evident**
- **Subtle**
- **Evasive**

우리 연구 입장에서는 이 NSFW 세분화가 핵심이다.

### Evident
누가 봐도 대놓고 unsafe한 경우.
우리 쪽 taxonomy로 치면 **Full**에 가장 가깝다.

### Subtle
직접적이진 않지만 unsafe signal이 분명 있는 경우.
우리의 **Partial**에 가장 가깝다.

### Evasive
고전예술, 우회적 표현, 간접적 묘사처럼,
“노골적이지 않지만 안전하지 않다” 혹은 “판단이 더 어렵다”는 영역.
이건 Ring-A-Bell 스타일 adversarial / evasive prompt와 매우 닮아 있다.

즉 MJ-Bench는 우리에게 다음 메시지를 준다.

> NSFW는 binary로만 보면 안 되고, 적어도 evident / subtle / evasive처럼 graded하게 볼 필요가 있다.

이건 우리 **NotRel / Safe / Partial / Full** 중에서,
특히 **Partial / Full 구분**을 정당화하는 외부 근거로 아주 좋다.

---

## 2.6 논문의 핵심 실험 메시지

primary source 기준으로 논문은 대략 다음을 주장한다.

### 1) closed-source VLM이 평균적으로 가장 강하다
GPT-4o가 평균적으로 가장 좋다고 보고한다.

### 2) 작은 scoring model이 항상 약한 건 아니다
alignment나 image quality에서는 CLIP-based scoring model이
open-source VLM보다 더 나은 구간이 있다.

### 3) safety와 bias는 VLM이 강하다
논문은 그 이유를 **reasoning capability**에서 찾는다.
즉 safety는 단순 similarity보다 “상황을 읽고 해석하는 능력”이 중요하다는 것이다.

### 4) feedback scale이 중요하다
논문은 numerical score뿐 아니라 **Likert-scale / 자연어형 피드백**도 본다.
그리고 VLM judge는 종종 natural-language scale에서 더 안정적이라고 본다.

이건 우리에게 매우 중요한 시사점을 준다.
우리도 safety judge를 사용할 때,
너무 딱딱한 숫자 점수 하나보다
**설명 가능한 class decision** 혹은 richer prompting이 더 나을 수 있다는 뜻이다.

---

## 2.7 우리 repo와 직접 연결: 무엇을 정당화해주는가

MJ-Bench가 우리에게 주는 가장 큰 가치는 두 가지다.

### (a) Qwen3-VL judge를 ‘검증 대상’으로 보게 해준다

우리도 지금 Qwen3-VL을 safety judge로 쓴다.
그런데 reviewer가 물을 수 있다.

> “왜 그 judge를 믿나요?”

MJ-Bench는 바로 여기에 대한 답이 된다.

즉 우리는 논문에서:
- Qwen judge를 그냥 쓰는 게 아니라
- **judge benchmarking 관점에서 calibrate해야 한다**

고 주장할 수 있다.

repo에도 이미 `vlm/eval_mjbench_safety.py`가 있으므로,
이건 단순 citation이 아니라 실제 evaluation path와 연결된다.

### (b) graded safety taxonomy를 정당화해준다

우리 current taxonomy:
- NotRelevant
- Safe
- Partial
- Full

여기서 safety severity 쪽은
MJ-Bench NSFW taxonomy를 reference로 삼을 수 있다.

물론 완전 동일하진 않다.
하지만:
- Evident ≈ Full
- Subtle ≈ Partial
- Evasive ≈ harder / adversarial NSFW case

라는 구조적 대응은 충분히 설득력 있다.

즉 reviewer에게:

> “우리가 safety를 severity-aware하게 보는 것은 임의적 설계가 아니라, multimodal judge benchmark에서도 이미 graded NSFW taxonomy가 쓰이고 있다”

고 말할 수 있다.

---

## 2.8 이 논문을 우리 generated image metric으로 곧바로 쓰면 안 되는 이유

여기서도 주의가 필요하다.

MJ-Bench는 **judge benchmark**다.
즉 이것만으로 우리 method 성능이 자동으로 평가되지는 않는다.

이 논문이 직접 주는 것은:
- judge reliability evidence
- taxonomy justification
- 어떤 judge family가 어떤 축에 강한지에 대한 reference

이지,
우리 output image에 대한 final metric 그 자체는 아니다.

또한 pairwise preference benchmark이기 때문에,
우리 single-image 4-way classification과 완전히 같다고 말하면 안 된다.

즉 MJ-Bench는
**주 metric의 대체물**이 아니라,
**주 metric을 정당화하고 calibration하는 외부 benchmark**로 쓰는 것이 맞다.

---

## 2.9 우리 논문에서 가장 좋은 포지셔닝

가장 좋은 문장은 이렇다.

> MJ-Bench demonstrates that multimodal judges themselves should be benchmarked, and that safety judgment is inherently graded. We use this as external support for validating our Qwen-based judge and for motivating severity-aware safety labels.

즉 MJ-Bench는:
- **judge validation**
- **graded safety rationale**

를 맡는 논문으로 배치하는 게 가장 좋다.

---

# 3. LLaVA-Reward

## 3.1 한 줄로 먼저 설명하면

이 논문은

> **“alignment / fidelity / safety를 따로따로 heuristic으로 재지 말고, pretrained MLLM hidden state를 이용해 human-aligned reward model로 직접 배우자”**

는 논문이다.

즉 judge를 넘어서 **reward model**로 간다.

---

## 3.2 문제의식: 기존 judge / reward 방식이 왜 불만족스러운가

논문은 기존 방식을 크게 두 부류로 본다.

### A. CLIP / BLIP 계열
- CLIPScore
- PickScore
- HPS
- ImageReward

장점은 빠르고 단순하지만,
- bag-of-words 성향
- 복잡한 text-image relation에 약함
- generalization이 충분하지 않음

이라는 약점이 있다.

### B. VQA / token-probability 기반 MLLM judge
- 긴 instruction prompt 필요
- “yes/no”, “good/bad” 같은 특정 토큰 likelihood에 너무 의존
- pairwise preference처럼 작은 quality gap을 fine-grained하게 배우기 어려움
- discrete supervision에 끌려 bias된 score가 나올 수 있음

논문은 이 두 부류의 약점을 합쳐서 다음처럼 본다.

> “기존 방식은 너무 prompt-heavy 하거나, 너무 token-heavy 하거나, 특정 perspective에만 최적화돼 있다.”

그래서 대안으로
**hidden-state 기반 reward modeling**을 제안한다.

---

## 3.3 핵심 아이디어: hidden state에서 reward를 바로 읽는다

LLaVA-Reward의 핵심은:
- `(text, image)` 쌍을 pretrained MLLM에 넣고
- 긴 평가 문장을 생성시키지 않고
- 마지막 hidden state와 visual representation을 활용해
- 바로 scalar reward 혹은 ranking score를 뽑는 것이다.

즉 질문-응답형 judge보다 훨씬 direct하다.

이 발상의 장점은:
- training/inference가 더 효율적이고
- paired preference를 더 자연스럽게 학습할 수 있고
- alignment / fidelity / safety / ranking 같은 여러 관점을 하나의 reward framework 안에 넣을 수 있다는 점이다.

---

## 3.4 SkipCA: 이 논문에서 제일 중요한 구조적 포인트

논문이 단순히 “hidden state 읽자”에 그치지 않는 이유가 **SkipCA**다.

문제의식은 이렇다.

decoder-only MLLM에서는 깊은 층으로 갈수록 visual token 정보가 희석될 수 있다.
그러면 마지막 hidden state만 보고 reward를 만들 때,
정작 중요한 visual signal이 약해질 수 있다.

그래서 논문은:
- early-stage visual feature
- late-stage hidden representation

을 다시 연결하는 **Skip-connection Cross Attention**을 붙인다.

쉽게 말하면,

> “모델이 끝부분에서 reward를 결정할 때, 초반에 봤던 이미지 단서를 다시 참고하게 하자”

는 설계다.

논문 primary source 기준으로 이 모듈은:
- projected visual token `e_v`
- final hidden state `e_h`

를 연결해서 reward head로 보낸다.

이 구조가 중요한 이유는,
text-image evaluation은 단순 텍스트 reasoning이 아니라
**이미지와 텍스트를 끝까지 같이 생각해야 하는 문제**이기 때문이다.

논문도 safety set에서 SkipCA가 특히 유의미하다고 본다.

---

## 3.5 학습 방식: paired와 unpaired를 모두 받는다

논문은 reward model training을 꽤 실용적으로 설계했다.

### paired preference data
chosen image와 rejected image가 있는 경우,
**Bradley–Terry ranking loss**를 쓴다.

즉 같은 prompt 아래에서:
- 어떤 이미지가 더 선호되는지
- score 차이를 통해 배우게 한다.

이건 “미세한 quality gap”을 배우기에 좋다.

### unpaired / binary labeled data
예를 들어 safety처럼
safe vs unsafe의 binary label만 있는 경우,
**cross-entropy loss**로 학습한다.

즉 paired preference가 없는 safety 데이터도 유연하게 쓸 수 있다.

이게 중요하다.
왜냐하면 safety 데이터는 अक्सर fine-grained ranking보다
binary labeled dataset 형태로 더 많이 존재하기 때문이다.

---

## 3.6 논문이 다루는 네 가지 perspective

primary source 기준으로 논문은 아래 네 perspective를 학습한다.

- **text-image alignment**
- **artifact / fidelity**
- **safety**
- **overall ranking / inference-time scaling**

그리고 사용한 데이터 규모는 대략:
- IR-alignment **158k** pairwise
- IR-fidelity **84k** pairwise
- safety **UnsafeBench 8.1k** binary
- inference-time scaling **IR-ranking 136k** pairwise

즉 이 논문은 “safety فقط”가 아니라,
**여러 evaluation objective를 서로 다른 head / perspective로 다루는 멀티-오브젝트 reward framework**라고 보는 게 정확하다.

---

## 3.7 base model과 tuning 철학

논문은 base MLLM으로 **Phi-3.5-vision 4.2B**를 선택하고,
full fine-tuning 대신 **LoRA adapter** 중심으로 효율 fine-tuning을 한다.

이 점이 중요한 이유는,
이 논문이 막대한 compute를 쓰는 giant proprietary judge가 아니라,
비교적 실용적인 크기에서 human-aligned reward modeling을 하려는 시도이기 때문이다.

즉 메시지는:

> “충분히 강한 pretrained MLLM이 있으면, 긴 instruction-engineering 없이도 reward model로 바꿔 쓸 수 있다.”

이다.

---

## 3.8 실험 결과는 어떻게 읽어야 하나

논문 메시지는 크게 두 덩어리다.

### (a) auto-evaluation 성능
LLaVA-Reward는 alignment / fidelity / safety 같은 public benchmark에서 strong result를 낸다고 주장한다.
특히 논문은:
- CLIP/BLIP 계열
- 다른 MLLM-based judge

보다 더 human-aligned score를 준다고 밀고 있다.

즉 이 논문의 핵심 승부는
**“judge로서 더 사람 취향에 맞다”**이다.

### (b) inference-time scaling / FK steering
이 논문은 evaluation-only에서 멈추지 않고,
reward model을 **FK steering**에 넣는다.

쉽게 말하면,
생성 중 intermediate candidate들 중에서
reward가 더 높은 방향을 sequential Monte Carlo 식으로 고르며 진행하는 것이다.

논문은 DrawBench / GenEval prompt에서
LLaVA-Reward 기반 FK steering이:
- ImageReward
- CLIPScore
- 다른 grader

보다 더 prompt alignment를 잘 끌어올리는 qualitative/quantitative evidence를 제시한다.

즉 이 reward model은 단순 평가기를 넘어서,
**training-free steering signal**로도 활용될 수 있음을 보여준다.

이건 우리 repo 입장에서 꽤 흥미롭다.
우리는 이미 inference-time safe guidance를 하는 연구이기 때문이다.

---

## 3.9 우리 연구와 연결: 왜 이 논문이 좋은 secondary evaluator reference인가

이 논문이 우리에게 주는 핵심 가치는 다음이다.

### 1) alignment와 safety를 완전히 분리된 세계로 보지 않아도 된다는 점

우리 현재 stack은
- alignment: VQAScore
- safety: Qwen judge

로 분리되어 있다.

이건 아주 합리적이지만,
reviewer가 “너무 metric이 분절된 것 아니냐”고 물을 수 있다.

LLaVA-Reward는 여기에 대해:

> “최근에는 하나의 multimodal reward framework 안에서 alignment / fidelity / safety를 함께 다루는 흐름도 있다”

는 reference가 된다.

즉 우리는 이 논문을 이용해,
현재 metric 분리가 잘못된 것이 아니라,
**향후 unified reward-style evaluation로 확장 가능한 중간 단계**라고 말할 수 있다.

### 2) continuous score의 중요성

우리 current safety judge는 class-based다.
그런데 subtle case에서는:
- safe와 partial 사이
- partial과 full 사이

에 연속적인 정도 차이가 존재할 수 있다.

LLaVA-Reward는 이걸 **continuous reward**로 다룰 가능성을 보여준다.

즉 우리 repo에서도 장기적으로는:
- Qwen class
- VQAScore
- reward-style continuous safety score

를 같이 보고, judge disagreement를 분석할 수 있다.

### 3) inference-time steering과의 연결

이건 당장 evaluation보다 future work 쪽이지만,
LLaVA-Reward는 reward model이 실제 generation control에도 들어갈 수 있음을 보여준다.

우리 method가 training-free safety guidance라면,
장기적으로는
- CAS / spatial mask / safe CFG
- plus reward-style reranking or scaling

같은 hybrid 방향도 상상할 수 있다.

---

## 3.10 이 논문을 우리의 main metric 근거로 바로 쓰면 안 되는 이유

여기서도 주의가 필요하다.

LLaVA-Reward는 **학습된 reward model**이다.
즉 평가기의 편향과 취약성이 여전히 남는다.

논문 자체도 다음 리스크를 인정한다.
- training data quality 부족
- reward hacking 가능성
- 소수 source model preference에서 온 편향

즉 이 모델은 “진실 판정기”가 아니라,
잘 학습된 **human-aligned scorer**에 가깝다.

그래서 우리 논문에서 이걸 쓰더라도,
단독 main metric으로 두면 오히려 위험하다.

가장 좋은 위치는:
- **secondary cross-check**
- **future auxiliary evaluator**
- **reward-style unified evaluation의 reference**

이다.

---

## 3.11 우리 논문에서 가장 좋은 포지셔닝

가장 자연스러운 문장은 이렇다.

> LLaVA-Reward shows that a multimodal reward model can jointly score alignment, fidelity, and safety in a human-aligned manner. While we do not rely on it as a sole metric, it supports the broader legitimacy of VLM-based multi-objective evaluation and motivates future continuous reward-style assessment in our setting.

즉 이 논문은
- 현재 main metric의 직접 근거라기보다는
- **한 단계 더 나아간 최신 evaluation 흐름**을 보여주는 논문이다.

---

# 4. 세 논문을 합치면 우리 evaluation story는 어떻게 정리되는가

이제 세 편을 합치면 역할이 매우 명확해진다.

## 4.1 역할 분담

### VQAScore
- 맡는 역할: **alignment / semantic preservation**
- 왜 필요한가: harmful concept을 지운 뒤에도 prompt 의미를 보존했는지 봐야 함

### MJ-Bench
- 맡는 역할: **judge validation + graded safety taxonomy justification**
- 왜 필요한가: safety judge는 그냥 쓰면 안 되고, judge로서 검증되어야 함

### LLaVA-Reward
- 맡는 역할: **secondary reward-style cross-check / future unified evaluator**
- 왜 필요한가: alignment / fidelity / safety를 하나의 human-aligned score framework로 보는 최신 흐름 제시

---

## 4.2 우리 repo에 대한 가장 설득력 있는 최종 해석

따라서 우리 evaluation은 이렇게 쓰는 게 가장 좋다.

1. **Qwen3-VL** = primary safety judge  
2. **VQAScore** = prompt/image alignment 및 content preservation  
3. **MJ-Bench** = judge calibration 및 graded safety taxonomy의 외부 근거  
4. **LLaVA-Reward** = future auxiliary learned reward judge

즉 reviewer에게는:

> “우리는 단순히 안전 여부만 보는 것이 아니라, judge reliability와 semantic preservation까지 포함하는 구조화된 multimodal evaluation을 사용한다”

라고 말할 수 있다.

이게 이번 related work에서 가장 중요한 메시지다.

---

## 4.3 우리가 논문에서 절대 헷갈리면 안 되는 포인트

### 헷갈리면 안 되는 것 1
**VQAScore = safety metric** 이라고 쓰면 안 됨
- VQAScore는 alignment metric이다.
- 우리에게선 “safe rewrite와 가까운가”를 보는 보조 축이다.

### 헷갈리면 안 되는 것 2
**MJ-Bench = 우리 generated image benchmark** 라고 쓰면 안 됨
- MJ-Bench는 judge benchmark이다.
- 우리 judge를 calibration하는 reference다.

### 헷갈리면 안 되는 것 3
**LLaVA-Reward = truth oracle** 라고 쓰면 안 됨
- 이건 learned reward model이다.
- strong reference이지만 data dependence와 reward hacking risk가 있다.

이 세 구분만 명확히 하면 related work / evaluation section이 훨씬 단단해진다.

---

## 4.4 바로 실행 가능한 액션 아이템

### 즉시
- `vlm/eval_vqascore_alignment.py` 결과를 표로 정리
- `vlm/eval_mjbench_safety.py`로 Qwen judge calibration 실행
- 논문 evaluation-related work에 세 논문을 각각 다른 역할로 인용

### 다음 단계
- Qwen class judgment와 VQAScore gap의 correlation 보기
- 가능하면 LLaVA-Reward를 보조 score로 붙여 disagreement case 분석

---

## Source links
- VQAScore ECCV 2024 poster: https://eccv.ecva.net/virtual/2024/poster/2239
- VQAScore PDF: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01435.pdf
- MJ-Bench project: https://mj-bench.github.io/
- MJ-Bench arXiv/PDF: https://arxiv.org/abs/2407.04842
- LLaVA-Reward ICCV 2025 open access page: https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_Multimodal_LLMs_as_Customized_Reward_Models_for_Text-to-Image_Generation_ICCV_2025_paper.html
- LLaVA-Reward PDF: https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_Multimodal_LLMs_as_Customized_Reward_Models_for_Text-to-Image_Generation_ICCV_2025_paper.pdf
