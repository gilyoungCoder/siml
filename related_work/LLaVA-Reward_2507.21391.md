# LLaVA-Reward: Multimodal LLMs as Customized Reward Models for Text-to-Image Generation

**논문**: *Multimodal LLMs as Customized Reward Models for Text-to-Image Generation*  
**학회**: ICCV 2025  
**arXiv**: https://arxiv.org/abs/2507.21391  
**CVF open-access page**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_Multimodal_LLMs_as_Customized_Reward_Models_for_Text-to-Image_Generation_ICCV_2025_paper.html  
**공식 PDF**: https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_Multimodal_LLMs_as_Customized_Reward_Models_for_Text-to-Image_Generation_ICCV_2025_paper.pdf

---

## 1. 이 논문이 풀려는 문제를 쉽게 말하면

요즘 text-to-image 평가에는 judge가 많이 들어간다. 그런데 judge에도 여러 종류가 있다.

- CLIPScore 같은 embedding similarity
- reward model
- MLLM/VLM judge
- VQA 기반 metric

이 논문은 그중에서도 다음 문제를 겨냥한다.

> “alignment, fidelity, safety를 각각 따로따로 보는 대신, 하나의 multimodal reward model이 여러 축을 같이 잘 볼 수는 없을까?”

기존 CLIP 기반 reward model은 빠르지만,
- bag-of-words처럼 동작하기 쉽고
- compositional semantics를 충분히 반영하지 못하며
- safety 같은 고차원적인 판정을 하기에 한계가 있다.

반대로 MLLM은 reasoning은 강하지만,
- 긴 instruction prompt가 필요하고
- 텍스트를 생성해 판단하는 방식은 느리며
- 점수화가 번거롭다.

이 논문은 그 중간을 노린다.

> MLLM의 내부 표현(hidden state)을 직접 reward로 읽어내면, reasoning 능력은 유지하면서도 효율적인 reward model을 만들 수 있다.

---

## 2. 핵심 아이디어: “출력 문장”보다 “내부 표현”을 reward로 쓰자

이 논문의 핵심은 MLLM에게 긴 문장을 생성하게 만드는 대신,
**마지막 hidden state와 시각 토큰 정보를 이용해 직접 reward를 예측**하는 것이다.

즉,

- 입력: `(text, image)`
- MLLM이 둘을 인코딩
- hidden state를 읽음
- reward head가 alignment/safety/fidelity 등의 점수를 냄

이렇게 하면 judge로서 더 빠르고, 학습 가능한 reward model이 된다.

---

## 3. 왜 그냥 마지막 hidden state만 쓰지 않고 SkipCA를 넣었는가

이 논문에서 가장 기술적으로 중요한 부분이 **SkipCA (Skip-connection Cross Attention)** 다.

### 3.1 문제 배경

decoder-only MLLM에서는 깊은 층으로 갈수록 시각 정보가 약해질 수 있다. 즉 마지막 hidden state만 보고 reward를 만들면,

- 텍스트 reasoning은 남아 있지만
- 실제 visual evidence와의 결합이 충분히 강하지 않을 수 있다.

특히 safety나 artifact는 image evidence가 중요하기 때문에 이 문제가 더 크게 다가온다.

### 3.2 SkipCA의 직관

저자들은 다음 생각을 한다.

- 마지막 layer의 hidden state는 “최종 판단”에 가깝다.
- 하지만 초기 visual token은 실제 이미지 정보를 더 강하게 담고 있다.

그러면 둘을 다시 연결해주면 좋지 않을까?

그래서 SkipCA는
- 마지막 hidden state(EOS 중심 표현)
- visual projector 뒤의 visual token

을 cross-attention으로 다시 연결해, reward head가 시각 정보를 더 잘 활용하게 만든다.

쉽게 말하면,

> “최종 판단 벡터가 이미지 증거를 다시 한 번 참조하게 만들자”

는 설계다.

### 3.3 논문이 주장하는 효과

논문은 이 설계 덕분에
- text-image reasoning이 더 좋아지고
- safety/fidelity 같은 시각 의존적 판단도 더 좋아진다고 본다.

PDF 후반부의 ablation도 visual projector 직후의 visual token을 쓰는 것이 더 낫다는 쪽을 지지한다.

---

## 4. 이 reward model은 어떤 관점을 학습하는가

이 논문은 reward를 하나만 학습하는 것이 아니라, 여러 perspective를 나눠서 본다.

핵심 perspective는 다음 네 가지다.

1. **text-image alignment**  
2. **fidelity / artifact**  
3. **safety**  
4. **overall ranking**

이 점이 매우 중요하다. 즉 LLaVA-Reward는 “좋은 이미지”를 한 줄 점수로만 보지 않고,

- prompt를 잘 따랐는지
- 이미지가 artifact 없이 자연스러운지
- 안전한지
- 전반적으로 선호되는지

를 분해해서 본다.

우리 연구 입장에서는 이게 굉장히 매력적이다. 왜냐하면 우리도 사실 같은 고민을 하고 있기 때문이다.

- safe하긴 한데 fidelity가 무너질 수 있고
- alignment는 유지됐지만 harmful concept이 남을 수 있고
- content preservation과 safety 사이 trade-off가 있기 때문이다.

---

## 5. 학습 데이터와 목표는 어떻게 구성되는가

논문은 여러 preference 데이터셋을 사용한다.

본문 기준으로 대략 다음 규모가 언급된다.

- IR-alignment: 158k pairwise
- IR-fidelity: 84k pairwise
- Safety(UnsafeBench): 8.1k binary
- IR-ranking: 136k pairwise

즉 reward model은 단순히 “이미지 좋음/나쁨” 하나만 배우는 게 아니라,
각 perspective에 맞는 supervision을 따로 받는다.

학습 목표는 주로
- pairwise ranking
- binary classification
- preference learning

성격을 가진다.

여기서 중요한 건, safety도 이 reward model 안에 **명시적인 학습 축**으로 들어간다는 점이다.

즉 이 논문은 safety를 단순 부가 옵션으로 보는 것이 아니라,
reward modeling 안에서 독립적인 평가 관점으로 다룬다.

---

## 6. 이 논문이 VQAScore나 MJ-Bench와 다른 점

이 세 논문은 모두 VLM/MLLM 기반 평가를 다루지만 역할이 다르다.

### VQAScore와의 차이

- VQAScore는 alignment metric이다.
- 질문 기반 yes/no 확률을 사용한다.
- 주로 prompt-image faithfulness에 집중한다.

반면 LLaVA-Reward는
- learned reward model이고
- alignment뿐 아니라 fidelity, safety까지 함께 본다.

즉 VQAScore가 “한 축을 잘 보는 도구”라면,
LLaVA-Reward는 “여러 축을 같이 보는 learned judge”에 가깝다.

### MJ-Bench와의 차이

- MJ-Bench는 judge benchmark다.
- judge를 평가하는 데이터셋이다.

반면 LLaVA-Reward는
- 그 benchmark 위에서 경쟁하는 **judge/reward model 자체**다.

즉 MJ-Bench가 “시험장”이라면,
LLaVA-Reward는 그 시험장에 들어가는 “응시자 모델” 중 하나라고 보면 된다.

---

## 7. 실험 결과를 어떻게 읽어야 하는가

이 논문의 핵심 메시지는 “MLLM hidden state 기반 reward model이 alignment, fidelity, safety에서 강하다”는 것이다.

### 7.1 alignment / fidelity / safety를 모두 본다

논문은 MJ-Bench 관련 subset, TIFA160, UnsafeDiff, SMID 등 여러 evaluation으로 모델을 본다.

즉 단일 benchmark 하나에서 좋았다는 수준이 아니라,
- alignment
- fidelity
- safety

여러 축에서 강한 generality를 보이려고 한다.

### 7.2 safety 성능도 경쟁력이 있다

PDF의 safety evaluation table을 보면 LLaVA-Reward 변형은 ImageGuard나 LlavaGuard 같은 safety judge와 비교해도 경쟁력이 있다.

이건 중요하다. 왜냐하면 이 논문이 단순히 “미학적 reward”만 잘 보는 모델이 아니라,
**safety까지도 reward-model 관점에서 꽤 잘 다룰 수 있다**는 점을 보여주기 때문이다.

### 7.3 FK steering까지 연결한다

더 흥미로운 부분은 이 reward model을 evaluation으로만 끝내지 않고,
**FK steering** 같은 inference-time scaling에도 연결한다는 점이다.

즉 reward를 보고,
- 더 alignment 좋은 쪽,
- 더 안전한 쪽,
- 더 fidelity 높은 쪽

으로 sampling을 유도할 수 있다는 것이다.

이건 우리 연구에는 직접 main contribution은 아니지만, future work 레벨에서 매우 흥미롭다. 왜냐하면 우리도 training-free guidance를 하고 있기 때문이다.

---

## 8. 우리 연구에 구체적으로 어떤 의미가 있는가

### 8.1 가장 먼저 드는 의미: “alignment와 safety를 함께 보는 learned judge”의 선례

우리 현재 평가는 사실 분리형이다.

- safety: Qwen3-VL / NudeNet / SigLIP2
- alignment: VQAScore
- quality: FID/CLIP 등

이건 합리적이지만, reviewer가 이렇게 물을 수도 있다.

- “왜 이렇게 metric이 조각나 있나요?”
- “하나의 더 통합적인 judge는 없나요?”

여기에 LLaVA-Reward는 아주 좋은 related-work reference다.

> 최근에는 alignment, fidelity, safety를 함께 보는 learned multimodal reward model도 등장하고 있다.

즉 우리 평가를 바로 대체하는 것은 아니지만,
우리 논문이 너무 뒤처진 framing이 아니라는 걸 보여준다.

### 8.2 secondary cross-check로 쓸 수 있다

우리 setting에서 가장 현실적인 사용법은 다음이다.

- main safety metric은 그대로 Qwen3-VL
- main alignment metric은 그대로 VQAScore
- 여기에 **LLaVA-Reward safety/alignment score를 보조적 cross-check**로 붙인다

이렇게 하면 장점이 있다.

1. 특정 judge에만 의존한다는 비판을 줄일 수 있다.
2. class label이 아니라 continuous reward를 같이 볼 수 있다.
3. alignment–safety trade-off를 다른 관점에서 다시 볼 수 있다.

### 8.3 future work로는 더 흥미롭다

장기적으로는 이 reward를 활용해,
- 어떤 generated sample이 더 안전하고,
- 어떤 sample이 원래 의미를 덜 훼손했는지

reranking이나 inference-time steering으로 연결할 수도 있다.

우리 연구는 현재 training-free selective guidance이므로,
이 reward model은 훗날 “evaluation-only”를 넘어 “guidance companion”으로도 흥미롭다.

---

## 9. 하지만 이 논문을 그대로 믿으면 안 되는 이유

이 부분이 매우 중요하다.

### 9.1 learned reward model의 본질적 한계

논문 자체도 분명히 인정하듯, learned reward model은
- training data 품질
- preference 수집 방식
- source model 편향
- reward hacking 가능성

의 영향을 받는다.

즉 LLaVA-Reward는 강력한 judge일 수 있지만,
**절대적 진실판정기**는 아니다.

### 9.2 safety를 binary/learned preference로만 보면 놓치는 것들

우리 연구는 `Full / Partial / Safe / NotRelevant` 같이 nuanced한 분류를 중요하게 본다. 그런데 learned reward는 종종

- 하나의 scalar
- 혹은 binary-safe/bad 성향

으로 수렴할 위험이 있다.

이 경우,
- subtle unsafe case
- semantic collapse
- prompt 불일치인데 safe하다고만 나오는 경우

를 섬세하게 설명하기 어렵다.

### 9.3 그래서 우리 논문에서의 올바른 위치는

정확히는 이렇다.

- **main metric**이 아니라
- **reward-style auxiliary evaluator**
- 혹은 **future extension reference**

로 두는 것이 가장 정직하다.

---

## 10. 우리 논문에서 가장 좋은 포지셔닝

가장 설득력 있는 framing은 다음과 같다.

> LLaVA-Reward demonstrates that multimodal evaluation can move beyond discrete class labels and into learned, human-aligned reward modeling across alignment, fidelity, and safety. We do not use it as our sole arbiter, but it provides a strong precedent for auxiliary cross-checking and future unified evaluation.

한국어로 풀면,

- 최근에는 alignment와 safety를 함께 보는 learned judge도 등장했고,
- 우리는 현재 그 정도까지는 쓰지 않지만,
- 우리 evaluation이 앞으로 확장될 수 있는 방향을 보여주는 강한 reference다.

이게 가장 적절하다.

---

## 11. 한 줄 결론

LLaVA-Reward는 우리에게

> **“안전성 평가도 결국 더 통합적이고, learned이며, continuous한 multimodal reward 방향으로 갈 수 있다”는 최신 흐름의 강한 근거**

를 주는 논문이다.

다만 현재 우리 논문에서는 **주 지표**가 아니라,
**보조 judge / cross-check / future work reference**로 쓰는 것이 가장 적절하다.
