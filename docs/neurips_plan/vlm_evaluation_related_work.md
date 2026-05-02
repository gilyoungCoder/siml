# VLM 기반 평가 논문 3편 상세 분석 — NeurIPS용 diffusion safety evaluation 관점

> 목적: 우리 repo의 VLM-based evaluation을 **정당화**하고, 동시에 **어디까지 믿어야 하고 어디부터는 조심해야 하는지**까지 논문 수준으로 정리한다.  
> 다루는 논문:
> 1. **VQAScore** — ECCV 2024  
> 2. **MJ-Bench** — NeurIPS 2025  
> 3. **LLaVA-Reward** — ICCV 2025

---

## 먼저 큰 그림: 왜 이 세 논문을 같이 봐야 하나?

우리 연구에서 평가하고 싶은 것은 사실 한 가지가 아니다.

1. **이미지가 원래 prompt 의미를 얼마나 보존하는가?**  
2. **harmful concept이 실제로 사라졌는가?**  
3. **이 safety 판단을 내려주는 judge 자체를 믿어도 되는가?**  
4. **alignment / safety / quality를 따로따로 보지 말고 더 통합적으로 볼 수 있는가?**

이 네 질문에 대해 각 논문이 맡는 역할이 다르다.

- **VQAScore**는 1번에 강하다.  
  → “prompt-image alignment를 어떻게 더 정교하게 볼 것인가?”
- **MJ-Bench**는 3번에 강하다.  
  → “multimodal judge를 그냥 쓰면 안 되고, judge도 benchmark로 검증해야 한다.”
- **LLaVA-Reward**는 4번에 강하다.  
  → “alignment / fidelity / safety를 하나의 learned reward 체계로 볼 수도 있다.”

즉 이 세 논문을 같이 읽으면, 우리 평가 체계는 아래처럼 정리된다.

- **VQAScore** = prompt fidelity / semantic preservation
- **Qwen3-VL judge** = 주 safety 판정기
- **MJ-Bench** = 그 judge의 신뢰성 점검 + graded safety taxonomy 근거
- **LLaVA-Reward** = 보조적인 reward-style cross-check 혹은 future extension

이제 각 논문을 자세히 본다.

---

# 1. VQAScore 상세 분석

## 1.1 기본 정보

- **논문 제목**: *Evaluating Text-to-Visual Generation with Image-to-Text Generation*
- **학회**: ECCV 2024 poster
- **핵심 키워드**: text-image alignment, VQA-based scoring, compositional prompt evaluation, GenAI-Bench
- **주요 링크**:
  - ECCV poster: https://eccv.ecva.net/virtual/2024/poster/2239
  - PDF: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01435.pdf
  - arXiv: https://arxiv.org/abs/2404.01291
  - code/project: https://github.com/linzhiqiu/t2v_metrics

## 1.2 이 논문이 풀려는 정확한 문제

기존 text-to-image 자동 평가에서 가장 많이 쓰이는 축은 CLIPScore 계열이다.  
문제는 CLIPScore가 빠르고 편하지만, **복잡한 compositional prompt에서는 너무 약하다**는 점이다.

왜 약하냐?

- CLIP text encoder는 종종 문장을 **bag-of-words 비슷하게** 처리한다.
- 즉 문장 안의 **관계(relation)**, **속성(attribute)**, **개수(counting)**, **주체-객체 순서**를 충분히 반영하지 못한다.
- 예를 들어:
  - “말이 풀을 먹는다”와 “풀이 말을 먹는다”는 단어는 비슷하지만 의미는 완전히 다르다.
  - “세 개의 흰 달걀과 두 개의 갈색 달걀”도 단순 단어 일치만으로는 정확히 판정하기 어렵다.

이 논문은 이 문제를 매우 직관적으로 바꿔 말한다.

> alignment를 벡터 similarity로 볼 게 아니라,  
> “이 이미지가 정말 이 문장을 보여주고 있나?”를 질문-응답 형태로 물어보면 더 낫지 않나?

즉 핵심 문제는:

> **prompt-image alignment를 더 문장 의미에 가깝게 평가하는 방법이 필요하다.**

## 1.3 핵심 아이디어를 아주 쉽게 설명하면

논문은 prompt를 그대로 similarity 계산에 넣지 않고, **질문**으로 바꾼다.

예를 들어 prompt가

> “A cat eating a burger like a person.”

이면, 평가 모델에게 이렇게 묻는다.

> “Does this figure show a cat eating a burger like a person?”

그리고 모델이 **Yes라고 대답할 확률**을 점수로 쓴다.

즉 VQAScore는 사실상:

> **P(Yes | image, question(prompt))**

이다.

이 접근의 좋은 점은, 모델이 단순히 단어가 비슷한지만 보는 게 아니라
- 누가 무엇을 하는지,
- 어떤 속성이 붙었는지,
- 수량이 맞는지,
- 관계 구조가 맞는지

를 더 직접적으로 판단하게 만든다는 점이다.

## 1.4 수식/정의 수준에서 보면

논문은 prompt `t`와 image `i`에 대해, VQA 모델이 생성하는 “yes”의 확률을 alignment score로 둔다.

직관적으로는:

- 질문: `Does this image show {prompt}?`
- 출력: `yes` / `no`
- score: `P(yes)`

중요한 건, 이 score가 단순 텍스트-이미지 cosine similarity가 아니라 **질문에 대한 응답 확률**이라는 점이다.

이 차이가 왜 크냐면:

- CLIPScore는 embedding 두 개를 바로 비교한다.
- VQAScore는 prompt를 **명시적 판별 과제**로 바꾼다.
- 즉 평가가 “비슷하다”에서 “정말 맞다/아니다”로 이동한다.

## 1.5 논문이 제안하는 벤치마크 관점: GenAI-Bench

이 논문은 metric만 제안한 것이 아니라, 그 metric을 검증하기 위한 benchmark도 같이 민다.

핵심은 **GenAI-Bench**다.

- 1,600개의 compositional prompt
- 15,000개 이상의 human rating
- object / attribute / action / relation / counting 등 복합적 reasoning이 필요한 prompt 다수 포함

이 논문이 말하고 싶은 것은 단순하다.

> 쉬운 prompt에서는 아무 metric이나 얼추 그럴듯해 보일 수 있다.  
> 하지만 compositional prompt로 가면 metric의 수준 차이가 바로 드러난다.

이건 우리 연구에도 매우 중요하다.  
왜냐하면 safe generation에서 실패는 종종 **노골적 prompt**가 아니라 **우회적 / 구성적 / 함축적 prompt**에서 더 잘 드러나기 때문이다.

## 1.6 실험에서 무엇을 보여주나

논문은 challenging matching benchmark들(예: Winoground, EqBen)에서 VQAScore가 매우 강하다고 보고한다.

핵심 메시지:
- CLIPScore, PickScore 같은 방법은 compositional benchmark에서 거의 chance 수준에 가깝게 무너질 수 있다.
- VQAScore는 훨씬 높은 text/image/group score를 보인다.
- 심지어 일부 proprietary/대형 모델과 견줘도 매우 경쟁력 있다.

여기서 중요한 건 단순한 숫자 자체보다도,

> **VQAScore가 “복잡한 의미 일치”를 평가하는 데 훨씬 더 적합하다**

는 점이다.

## 1.7 우리 연구와 연결하면 정확히 무엇이 좋은가

이 repo에서 VQAScore는 이미 도입돼 있다.

- `vlm/eval_vqascore.py`
- `vlm/eval_vqascore_alignment.py`

특히 우리는 일반적인 “prompt faithfulness”보다 한 단계 더 나아간 사용을 하고 있다.

### 우리 쪽의 가장 좋은 사용 방식

우리는 safe generation 결과 이미지 `x_safe`에 대해 세 가지 prompt와 비교할 수 있다.

1. **original harmful prompt**
2. **anchor safe prompt**
3. **erased prompt** (harmful token 제거 버전)

이렇게 하면 단순히 “prompt에 잘 맞는가?”가 아니라,

- harmful 의미에는 멀어졌는지
- safe rewrite 의미에는 가까워졌는지
- 이미지 전체 semantic은 살아있는지

를 같이 볼 수 있다.

### 우리 setting에서 해석 예시

- `VQA(x_safe, original_prompt)`가 **낮다**  
  → harmful meaning이 잘 사라졌을 가능성
- `VQA(x_safe, anchor_prompt)`가 **높다**  
  → safe한 rewrite 의미는 잘 살아 있음
- `Gap = VQA(anchor) - VQA(original)`가 **크다**  
  → harmful만 선택적으로 밀어낸 것

이건 굉장히 좋은 설계다.  
왜냐하면 reviewer가 늘 묻는 질문,

> “safe해지긴 했는데, 그냥 이미지가 망가진 것 아닌가?”

에 대해 직접 답할 수 있기 때문이다.

## 1.8 하지만 이 논문이 우리에게 **해주지 못하는 것**

매우 중요하다. VQAScore는 강하지만 **safety metric 자체는 아니다**.

이 논문은 주로 alignment를 다룬다.
즉 아래 질문에는 좋다.

- “이 이미지가 이 문장을 보여주나?”
- “원래 의미를 어느 정도 보존했나?”

하지만 아래 질문에는 직접 답하지 않는다.

- “이 이미지가 객관적으로 unsafe한가?”
- “partial nudity와 full nudity를 잘 구분하는가?”
- “artistic evasive NSFW를 제대로 잡는가?”

즉 VQAScore만으로 safety를 재면 논리적으로 위험하다.

### 우리 논문에서의 정확한 포지션

VQAScore는 이렇게 써야 한다.

> **alignment / meaning preservation metric**

이지,

> **ultimate safety judge**

가 아니다.

## 1.9 reviewer 대응 문장으로 바꾸면

가장 설득력 있는 표현은 다음과 같다.

- 우리는 VQAScore를 사용해 **prompt-image alignment preservation**을 평가한다.
- 특히 safe intervention 이후에도 이미지가 **원래 장면 의미를 유지하는지**를 보기 위해 original / anchor / erased prompt 삼중 비교를 한다.
- VQAScore는 ECCV 2024에서 compositional prompt evaluation에 대해 강한 근거를 제공하므로, CLIP류보다 더 적합한 alignment metric이다.
- 그러나 safety 최종 판정은 별도의 judge와 함께 사용한다.

즉 VQAScore는 우리 evaluation에서 **의미 보존 축**을 담당한다.

---

# 2. MJ-Bench 상세 분석

## 2.1 기본 정보

- **논문 제목**: *MJ-Bench: Is Your Multimodal Reward Model Really a Good Judge for Text-to-Image Generation?*
- **학회**: NeurIPS 2025 poster
- **핵심 키워드**: judge benchmarking, multimodal evaluator, safety taxonomy, preference dataset, image generation judge calibration
- **주요 링크**:
  - NeurIPS poster: https://neurips.cc/virtual/2025/poster/121393
  - arXiv: https://arxiv.org/abs/2407.04842
  - project: https://mj-bench.github.io/
  - dataset: https://huggingface.co/datasets/MJ-Bench/MJ-Bench

## 2.2 이 논문이 풀려는 정확한 문제

이 논문은 생성 모델 자체보다 **judge**를 문제로 본다.

지금 이미지 생성 쪽에서는 reward model이나 multimodal judge를 많이 쓴다.
그런데 정작 연구자들은 종종 아래를 그냥 넘겨버린다.

- 이 judge가 alignment에는 강한가?
- safety에는 강한가?
- bias에는 강한가?
- numeric score가 좋은가, Likert가 좋은가?
- single-image input이 좋은가, multi-image pairwise가 좋은가?

즉 judge를 도구로만 쓰고, **judge 자체의 한계**는 잘 안 본다.

MJ-Bench는 이걸 정면으로 겨냥한다.

> “multimodal judge는 곧 evaluation pipeline의 핵심인데,  
> judge 자체를 benchmark하지 않으면 평가 전체가 흔들린다.”

이건 우리한테 매우 중요하다.  
왜냐하면 우리는 이미 Qwen3-VL 같은 multimodal judge를 주 평가 축으로 쓰고 있기 때문이다.

## 2.3 데이터셋 구조를 쉽게 설명하면

MJ-Bench의 각 샘플은 기본적으로:

- 하나의 instruction/prompt
- 두 개의 image
- 어떤 이미지가 더 바람직한지에 대한 label

로 되어 있다.

즉 pairwise preference benchmark다.

하지만 중요한 건 pairwise 형식보다도, **무엇을 기준으로 비교하느냐**다.  
논문은 judge를 4개의 큰 차원에서 평가한다.

1. **Alignment**
2. **Safety**
3. **Quality & Artifact**
4. **Bias & Fairness**

이 네 축 각각이 다시 여러 subcategory로 분해되어 있다.

### Alignment 하위 예시
- object
- attribute
- action
- spatial
- location
- object count

즉 judge가 단순히 “그럴듯하다”가 아니라, 어떤 종류의 의미 오류에 강하고 약한지까지 본다.

### Safety 하위 예시
- Toxicity-crime
- Toxicity-shocking
- Toxicity-disgust
- NSFW-Evident
- NSFW-Subtle
- NSFW-Evasive

바로 이 부분이 우리와 가장 밀접하다.

## 2.4 NSFW Evident / Subtle / Evasive를 왜 중요하게 봐야 하나

이 taxonomy는 우리 논문에 거의 직접적으로 연결된다.

### Evident
- 누가 봐도 노골적인 NSFW
- 직접적이고 명시적
- 우리 기준으로는 **Full**에 가장 가까움

### Subtle
- 노골적이지 않지만 unsafe 신호가 분명 존재
- 의상, 포즈, 노출 정도, 맥락 때문에 경계선에 걸침
- 우리 기준으로는 **Partial**에 매우 가까움

### Evasive
- 더 흥미로운 범주
- 노골적인 단어를 쓰지 않으면서도 NSFW를 유도하는 우회적 설정
- 예술, 고전회화, 문화적 표현, 은유적 묘사, indirect prompt 등이 포함될 수 있음
- 우리 쪽의 Ring-A-Bell / adversarial / art-style prompt 정당화에 특히 좋다

논문 PDF는 evasive caption construction까지 따로 설명한다.  
즉 “명시적 단어를 피하면서도 NSFW를 유도하는 prompt”를 일부러 만들어 judge를 시험한다.

이게 왜 좋냐면, 우리가 실제로 맞닥뜨리는 어려운 케이스가 바로 이런 류이기 때문이다.

## 2.5 이 논문이 보여주는 핵심 실험 메시지

논문의 가장 중요한 결론은 단순히 “GPT-4o가 잘한다”가 아니다.
정말 중요한 메시지는 다음이다.

### 메시지 1: judge는 축마다 다르게 강하다

- **CLIP-based scoring model**은 alignment와 image quality에서 강할 수 있다.
- **VLM judge**는 safety와 bias에서 더 강할 수 있다.

즉 judge도万能이 아니다.  
무엇을 재고 있느냐에 따라 더 적합한 judge가 달라진다.

이건 우리에게 매우 큰 힌트다.

> alignment는 VQAScore 같은 축으로 보고,  
> safety는 Qwen3-VL 같은 reasoning-heavy judge로 보는 현재 구조가 오히려 합리적이다.

### 메시지 2: feedback scale 자체도 중요하다

논문은 judge에게
- 숫자 점수로 답하게 할지,
- Likert scale로 답하게 할지,
- 자연어 reasoning을 시킬지,
- pairwise 비교를 시킬지

에 따라 성능 차이가 난다고 본다.

특히 open-source VLM은 숫자 정량화보다 **Likert-style / 자연어 기반 graded judgment**에서 더 안정적인 경우가 있다.

이것도 우리에게 유용하다.
왜냐하면 우리는 이미 단순 binary가 아니라
- NotRelevant
- Safe
- Partial
- Full

같은 **graded categorical safety judgment**를 하고 있기 때문이다.

### 메시지 3: judge 검증 없이 judge를 쓰면 위험하다

이 논문이 사실상 가장 강하게 말하는 건 이것이다.

> 생성 결과 평가에서 multimodal judge를 쓰는 것 자체는 괜찮지만,  
> 그 judge가 어떤 유형에서 약한지 먼저 밝혀야 한다.

즉 MJ-Bench는 “judge benchmark”이지 “generation benchmark”가 아니다.

이 distinction이 논문 작성에서 매우 중요하다.

## 2.6 우리 연구와 연결하면 정확히 무엇이 좋은가

### (A) Qwen3-VL judge를 믿을 근거를 만들 수 있다

우리는 safety 판정에 Qwen3-VL을 쓰고 있다.  
reviewer는 당연히 이렇게 물을 수 있다.

- 왜 Qwen인가?
- 얼마나 믿을 수 있나?
- explicit에는 강하고 evasive에는 약한 것 아닌가?

MJ-Bench는 이 질문에 답할 근거를 준다.

가장 좋은 전략은:

1. MJ-Bench safety subset에서 Qwen3-VL을 한번 돌린다.
2. NSFW Evident/Subtle/Evasive와 toxicity subset별 성능을 본다.
3. 그 결과를 appendix나 evaluation section 앞부분에 넣는다.

그러면 우리는 이렇게 말할 수 있다.

> “우리는 Qwen3-VL을 그냥 임의로 고른 것이 아니라,  
> MJ-Bench-style judge calibration 관점에서 최소한의 external validation을 했다.”

### (B) Full / Partial taxonomy의 외부 근거가 생긴다

우리의 `Full / Partial / Safe / NotRelevant` taxonomy는 실용적이지만, 외부 reference가 없으면 자의적으로 보일 수 있다.

MJ-Bench의:
- Evident
- Subtle
- Evasive

는 정확히 same taxonomy는 아니지만,
**severity-aware safety categorization**의 훌륭한 선례다.

그래서 논문에서는 이렇게 말할 수 있다.

- 우리는 safety를 binary로 보지 않는다.
- 최근 benchmark들도 NSFW를 여러 난이도/명시성 수준으로 나눈다.
- 우리의 Full/Partial 구분은 이런 graded-safety evaluation 흐름과 정합적이다.

### (C) Evasive category가 우리의 adversarial setting을 정당화해 준다

특히 이게 중요하다.

우리쪽 prompt 중에는:
- 직접적인 nude/naked가 없는 경우
- 고전회화풍 / 예술풍 / 은유형 prompt
- 우회적으로 unsafe를 유도하는 케이스

가 있다.

이때 reviewer가 “이건 너무 특이한 케이스 아니냐?”고 물을 수 있는데,  
MJ-Bench는 바로 그런 케이스를 **Evasive**라는 정식 safety difficulty로 다룬다.

즉 우리 setting은 weird한 예외가 아니라,
최신 benchmark 관점에서도 충분히 중요한 failure mode다.

## 2.7 하지만 MJ-Bench가 우리에게 **해주지 못하는 것**

이것도 분명히 해야 한다.

### 1) MJ-Bench는 judge benchmark다

즉 MJ-Bench는 우리 method의 safety 성능을 직접 재는 benchmark가 아니다.

그 역할은:
- judge calibration
- taxonomy justification

이지,
- “우리 method가 더 좋다”를 직접 증명해주는 건 아니다.

### 2) pairwise preference와 single-image 4-way classification은 다르다

MJ-Bench는 주로 두 이미지 중 어떤 것이 더 바람직한지 판단한다.  
반면 우리는 한 이미지를 보고
- Full
- Partial
- Safe
- NotRelevant

같은 절대 라벨을 붙인다.

즉 평가 설정이 완전히 동일하지 않다.  
따라서 “MJ-Bench에서 이런 결과가 나왔으니 우리 4-way judge도 자동으로 타당하다”고 말하면 과장이다.

### 3) NotRelevant를 직접 정당화하지는 못한다

MJ-Bench는 alignment / safety / quality / bias를 보지만,  
우리의 `NotRelevant`는 일종의 **content destruction / semantic collapse**를 분리해 잡기 위한 category다.

이건 MJ-Bench와 직접 대응되지는 않는다.
즉 `NotRelevant`는 여전히 우리 쪽의 별도 실용적 taxonomy다.

## 2.8 reviewer 대응 문장으로 바꾸면

가장 설득력 있는 표현은 다음과 같다.

- 우리는 multimodal judge를 main safety evaluator로 사용하지만, recent work shows that judges themselves must be benchmarked.
- MJ-Bench는 judge reliability를 alignment / safety / quality / bias로 분해해 보여주는 benchmark다.
- 특히 MJ-Bench의 NSFW Evident / Subtle / Evasive taxonomy는 safety severity를 단계적으로 보는 최근 흐름을 잘 보여준다.
- 따라서 우리의 Full / Partial style evaluation은 ad-hoc이 아니라 graded multimodal safety judging의 broader trend와 맞닿아 있다.

즉 MJ-Bench는 우리 논문에서 **judge validation + taxonomy justification** 역할을 맡는다.

---

# 3. LLaVA-Reward 상세 분석

## 3.1 기본 정보

- **논문 제목**: *Multimodal LLMs as Customized Reward Models for Text-to-Image Generation*
- **학회**: ICCV 2025
- **핵심 키워드**: reward model, multimodal LLM, hidden-state scoring, SkipCA, safety-aware evaluation, FK steering
- **주요 링크**:
  - ICCV poster: https://iccv.thecvf.com/virtual/2025/poster/286
  - open access page: https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_Multimodal_LLMs_as_Customized_Reward_Models_for_Text-to-Image_Generation_ICCV_2025_paper.html
  - PDF: https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_Multimodal_LLMs_as_Customized_Reward_Models_for_Text-to-Image_Generation_ICCV_2025_paper.pdf
  - arXiv: https://arxiv.org/abs/2507.21391

## 3.2 이 논문이 풀려는 정확한 문제

기존 text-to-image reward/evaluation model은 크게 두 부류다.

1. **CLIP-based reward**
   - 빠르지만 bag-of-words 성향이 강하고, 복합적 의미 판단이 약할 수 있음

2. **instruction-following MLLM judge**
   - reasoning은 좋지만, 긴 prompt / 긴 생성 / 높은 latency / tie 문제 등 비효율이 있음

이 논문은 질문을 이렇게 던진다.

> “MLLM의 hidden state 자체를 reward로 쓰면,  
> 길게 설명시키지 않아도 alignment / fidelity / safety를 잘 점수화할 수 있지 않을까?”

즉 핵심 목표는:

> **사람 선호와 잘 맞는 multimodal reward model을, MLLM 기반으로 더 효율적으로 만들자.**

## 3.3 핵심 구조를 쉽게 설명하면

논문이 제안하는 LLaVA-Reward의 아이디어는 크게 세 가지다.

### 아이디어 1: 답변 텍스트를 길게 생성하지 말고 hidden state에서 바로 점수를 읽자

기존 MLLM judge는 종종
- 긴 instruction을 주고
- reasoning을 시키고
- 마지막에 점수나 선호를 읽는다.

이건 느리고 불안정할 수 있다.

LLaVA-Reward는 대신,
- multimodal LLM에 image-text pair를 넣고
- 마지막 hidden representation을 뽑고
- reward head를 붙여 scalar reward를 낸다.

즉 “설명문 생성”보다 “hidden-state scoring”으로 간다.

### 아이디어 2: visual token 정보가 뒤로 갈수록 희석되니 SkipCA로 다시 연결하자

논문은 decoder-only MLLM에서, 깊은 층으로 갈수록 시각 정보가 희석될 수 있다고 본다.
특히 safety나 fidelity처럼 **이미지 쪽을 제대로 봐야 하는 task**에서 이게 문제가 된다.

그래서 reward head를 단순 MLP로 두지 않고 **Skip-connection Cross Attention (SkipCA)**를 넣는다.

쉽게 말하면:
- 깊은 층의 EOS hidden state를 query로 쓰고
- visual projector 쪽 token을 key/value로 다시 가져와서
- 마지막 reward 계산 직전에 **시각 정보를 다시 강하게 연결**한다.

이게 이 논문의 구조적 핵심이다.

### 아이디어 3: 하나의 모델을 여러 평가 관점으로 LoRA-adapt 하자

논문은 evaluation perspective를 여러 개로 나눈다.

- alignment
- fidelity / artifact
- safety
- overall ranking

그리고 base MLLM 위에 **LoRA adapter**를 perspective별로 붙여, 같은 backbone으로 여러 평가를 하게 만든다.

즉 “모델 하나 = 평가 하나”가 아니라,
**같은 멀티모달 backbone을 여러 reward head / adapter 관점으로 재사용**한다.

## 3.4 학습 objective와 데이터

논문은 reward model을 human preference로 학습한다.

### paired preference data
같은 prompt에서 두 이미지가 있고,
- 하나는 chosen
- 하나는 rejected

이면 Bradley–Terry ranking loss를 사용한다.

즉 모델이 chosen image에 더 높은 reward를 주도록 학습한다.

### unpaired / binary data
특히 safety의 경우 UnsafeBench 같은 binary labeled data도 쓴다.
즉 safety 쪽은 classification 성격도 있다.

논문 본문에 나온 training data 통계/설명에서 중요한 건:
- alignment / fidelity는 ImageReward류 data에서 변환한 preference pair를 사용
- safety는 **UnsafeBench** training set 사용
- alignment 쪽은 MJ-Bench alignment set과 겹치지 않게 filtering하고, hard negative pair를 추가해 더 어렵게 만듦

즉 단순히 공개 데이터 가져다 쓰는 게 아니라,
**evaluation leakage를 줄이고 hard negative를 추가해 reward model을 더 빡세게 훈련**한다.

## 3.5 이 논문이 보여주는 핵심 실험 메시지

### 메시지 1: LLaVA-Reward는 여러 public benchmark에서 강하다

논문은 alignment / fidelity / safety 축에서 SOTA급이라고 주장한다.
핵심 메시지는 다음과 같다.

- CLIP/BLIP 계열보다 일반화가 좋다.
- VQA-style MLLM judge보다 tie에 덜 갇히고, 작은 차이를 더 잘 구분한다.
- SkipCA를 넣지 않은 버전보다 특히 safety에서 좋아진다.

### 메시지 2: safety에서도 hidden-state reward + SkipCA가 실제로 의미 있다

논문은 MJ-Bench safety set과 SMID 같은 데이터에서,
- visual token을 다시 연결하는 SkipCA가 도움이 된다고 본다.
- 즉 reward model이 단순 text-heavy reasoning이 아니라, **시각적 근거를 더 잘 본다**는 주장을 실험으로 뒷받침한다.

이건 우리에게 중요하다.  
왜냐하면 safety 판정은 text alignment보다 이미지 자체의 시각 증거를 훨씬 더 많이 보기 때문이다.

### 메시지 3: evaluation model을 inference-time steering에도 쓸 수 있다

논문은 reward model을 evaluation으로만 끝내지 않고,
**FK steering** 같은 inference-time scaling에 연결한다.

즉 reward model이 잘 만들어지면,
- 평가에만 쓰는 게 아니라
- 생성 과정에서 더 좋은 이미지를 고르는 guide로도 쓸 수 있다는 것이다.

논문은 SD2.1, SDXL 같은 모델에서,
LLaVA-Reward 기반 FK steering이 다른 reward model보다 더 좋은 alignment를 보인다고 주장한다.

## 3.6 우리 연구와 연결하면 정확히 무엇이 좋은가

### (A) alignment / safety / quality를 하나의 learned judge family로 묶는 선례가 된다

지금 우리 평가는 여러 metric으로 쪼개져 있다.

- VQAScore
- Qwen3-VL
- NudeNet
- SigLIP2
- FID / CLIP

이 자체는 나쁘지 않다. 오히려 현실적이다.
하지만 reviewer는 이렇게 물을 수 있다.

> “평가가 너무 파편적이지 않나?”

이때 LLaVA-Reward는 중요한 반론 자료가 된다.

- 최근에는 alignment, fidelity, safety를 아우르는 **통합형 multimodal evaluator** 연구도 존재한다.
- 즉 우리가 여러 축을 나눠 보는 건 구식이 아니라, 오히려 요즘 흐름과 연결된다.
- 다만 우리는 learned reward model 하나에 다 맡기기보다, metric family를 분리해 쓰는 보수적 설계를 택한 것이라고 설명할 수 있다.

### (B) reward-style secondary cross-check로 쓸 수 있다

우리의 주 safety judge는 Qwen3-VL이고,
alignment는 VQAScore가 담당하는 게 가장 자연스럽다.

그런데 만약 reviewer가
- “이 결과가 특정 judge에만 맞춘 것 아닌가?”
- “judge-specific bias 아닌가?”

라고 묻는다면,
LLaVA-Reward는 아주 좋은 **보조 cross-check**가 될 수 있다.

즉:
- main result는 Qwen3-VL / VQAScore로 보고
- appendix나 additional experiment에서 LLaVA-Reward safety/alignment score를 같이 보고
- 두 evaluator가 대체로 같은 방향이면 평가 신뢰도가 올라간다.

### (C) 장기적으로는 reward-guided safe generation 연구로도 이어질 수 있다

이건 immediate need는 아니지만, research direction으로 좋다.

우리 method는 training-free safe generation이다.
LLaVA-Reward는 learned reward model이고, FK steering까지 보였다.

즉 미래에는:
- CAS + spatial CFG 같은 explicit method
- reward-guided selection/steering 같은 implicit evaluator-guided method

를 결합하는 연구도 가능하다.

논문 related work / future work에서 이 연결고리를 깔아두는 건 꽤 좋다.

## 3.7 하지만 LLaVA-Reward가 우리에게 **해주지 못하는 것**

여기도 분명히 선을 그어야 한다.

### 1) learned reward model은 ground truth가 아니다

논문도 스스로 말하듯,
- training data quality 문제
- reward hacking 가능성
- source model 편향

이 있다.

즉 reward model은 강력하지만,
**truth oracle**처럼 쓰면 안 된다.

### 2) 우리 taxonomy와 바로 일치하지 않는다

우리는:
- NotRelevant
- Safe
- Partial
- Full

같은 discrete class를 중요하게 본다.

반면 LLaVA-Reward는 continuous reward / ranking 중심이다.
즉 바로 치환되지는 않는다.

### 3) 실제 integration cost가 있다

이 논문은 멋지지만, repo에 바로 붙이려면
- 모델 로드
- adapter 선택
- env 설정
- batch inference
- score calibration

같은 practical work가 필요하다.

즉 “좋은 reference”와 “당장 메인 metric으로 도입 가능”은 다르다.

## 3.8 reviewer 대응 문장으로 바꾸면

가장 설득력 있는 표현은 다음과 같다.

- Recent work such as LLaVA-Reward shows that multimodal LLMs can be turned into human-aligned reward models spanning alignment, fidelity, and safety.
- We do not rely on a single learned reward model as ground truth, but this line of work supports the broader validity of multimodal automatic evaluation beyond hand-crafted metrics alone.
- In our setting, such reward-style evaluators are best interpreted as a secondary cross-check rather than the sole arbiter of safety.

즉 LLaVA-Reward는 **통합형 learned evaluator의 최신 reference**다.

---

# 4. 세 논문을 합쳐서 우리 논문 evaluation story로 다시 쓰면

이제 가장 중요한 부분이다.  
이 세 논문을 읽고 나면, 우리 논문의 평가 서사는 아래처럼 구성하는 게 가장 좋다.

## 4.1 역할 분담을 명확히 하자

### VQAScore
담당 질문:
> “안전하게 만들었더니 원래 prompt 의미까지 다 망가뜨린 것 아닌가?”

즉 **semantic preservation / alignment** 담당.

### Qwen3-VL judge
담당 질문:
> “이 이미지가 실제로 얼마나 unsafe한가?”

즉 **primary safety classification** 담당.

### MJ-Bench
담당 질문:
> “그 safety judge를 왜 믿어야 하나?”

즉 **judge validation + graded taxonomy justification** 담당.

### LLaVA-Reward
담당 질문:
> “alignment / safety / quality를 더 통합적으로 보는 learned evaluator 흐름은 없는가?”

즉 **secondary cross-check + future direction** 담당.

## 4.2 우리가 논문에서 하면 좋은 주장

가장 자연스러운 주장은 다음 네 줄이다.

1. 우리는 safety를 단순 binary로 보지 않는다.
2. 우리는 semantic preservation도 별도 축으로 정량화한다.
3. 우리는 multimodal judge를 쓰되, judge reliability 자체도 문제라는 점을 인식한다.
4. 우리는 discrete judge와 alignment metric을 결합하고, recent reward-model work를 보조 근거로 둔다.

이렇게 쓰면 evaluation이 훨씬 성숙해 보인다.

## 4.3 “무엇을 주장하지 말아야 하는가”도 중요하다

다음은 과장이다.

- “VQAScore가 safety를 측정해 준다” → 과장
- “MJ-Bench가 우리 method 성능을 직접 증명한다” → 과장
- “LLaVA-Reward면 진실을 다 알 수 있다” → 과장

대신 이렇게 써야 한다.

- VQAScore는 **alignment preservation** 근거
- MJ-Bench는 **judge validation / graded safety framing** 근거
- LLaVA-Reward는 **auxiliary learned evaluator** 근거

즉 세 논문의 역할을 분리해야 reviewer 설득력이 올라간다.

---

# 5. 우리 repo 기준으로 바로 실행할 실무 제안

## 5.1 지금 당장 해야 할 것

### 1) VQAScore alignment 결과를 main table/ablation으로 끌어올리기
이미 repo에는
- `vlm/eval_vqascore.py`
- `vlm/eval_vqascore_alignment.py`

가 있다.

이 결과를 appendix가 아니라 **main preservation evidence**로 승격시키는 게 좋다.

### 2) MJ-Bench로 Qwen3-VL judge sanity check 하기
이미 repo에는
- `vlm/eval_mjbench_safety.py`

가 있다.

이걸 돌려서 최소한
- overall
- NSFW subset
- toxicity subset
- 가능하면 Evident/Subtle/Evasive breakdown

을 넣어두면 judge 신뢰성 서사가 크게 좋아진다.

### 3) LLaVA-Reward는 바로 main metric으로 쓰지 말고 후보로 둬라
LLaVA-Reward는 좋아 보이지만, 현재 단계에서는:
- “추가 cross-check”
- “future extension”
- “appendix experiment”

정도가 가장 적절하다.

## 5.2 논문 문장 수준 제안

### evaluation section용
- “We evaluate semantic preservation with VQAScore, a VLM-based alignment metric shown to be stronger than CLIP-style similarity on compositional prompts.”
- “We evaluate image safety with a multimodal judge and calibrate that judge against recent judge-benchmarking work such as MJ-Bench.”
- “We adopt a graded safety taxonomy rather than a binary one, consistent with recent multimodal evaluation benchmarks that distinguish explicit, subtle, and evasive unsafe content.”

### related work용
- “VQAScore motivates our use of VLM-based alignment scoring.”
- “MJ-Bench motivates validating the judge itself and supports severity-aware safety evaluation.”
- “LLaVA-Reward indicates that multimodal evaluation can be learned as a reward model spanning alignment, fidelity, and safety.”

---

# 6. 최종 결론

이 세 논문을 깊게 읽고 나면, 우리 평가 체계는 아래처럼 정리하는 것이 가장 정확하다.

> **VQAScore는 의미 보존을, MJ-Bench는 judge의 신뢰성과 graded safety framing을, LLaVA-Reward는 통합형 learned evaluator의 가능성을 담당한다.**

즉 우리의 VLM-based evaluation은 단순히 “요즘 다들 VLM 쓰니까 우리도 쓴다”가 아니라,

- alignment는 왜 VLM으로 보는 게 맞는지,
- safety judge는 왜 calibration이 필요한지,
- graded safety는 왜 중요한지,
- learned reward evaluator는 어디까지 참고할 수 있는지

를 모두 설명할 수 있는 구조가 된다.

논문 관점에서 가장 중요한 한 줄은 이거다.

> **우리의 평가 프레임은 safety만 보는 단순 judge가 아니라, semantic preservation, graded safety judgment, and judge reliability를 함께 고려하는 구조화된 multimodal evaluation이다.**

---

## Primary sources used

- VQAScore ECCV 2024 poster: https://eccv.ecva.net/virtual/2024/poster/2239
- VQAScore PDF: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01435.pdf
- VQAScore repo: https://github.com/linzhiqiu/t2v_metrics
- MJ-Bench arXiv: https://arxiv.org/abs/2407.04842
- MJ-Bench project: https://mj-bench.github.io/
- MJ-Bench dataset: https://huggingface.co/datasets/MJ-Bench/MJ-Bench
- LLaVA-Reward arXiv: https://arxiv.org/abs/2507.21391
- LLaVA-Reward ICCV open-access page: https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_Multimodal_LLMs_as_Customized_Reward_Models_for_Text-to-Image_Generation_ICCV_2025_paper.html
- LLaVA-Reward PDF: https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_Multimodal_LLMs_as_Customized_Reward_Models_for_Text-to-Image_Generation_ICCV_2025_paper.pdf
