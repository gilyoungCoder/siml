# Jailbreaking 관련 3개 논문 종합 정리

대상 논문:
1. **Red-Teaming Text-to-Image Systems by Rule-based Preference Modeling** (arXiv:2505.21074)
2. **Jailbreaking Prompt Attack: A Controllable Adversarial Attack against Diffusion Models** (arXiv:2404.02928)
3. **The Illusion of Forgetting: Attack Unlearned Diffusion via Initial Latent Variable Optimization** (arXiv:2602.00175)

---

## 1. 먼저 결론
사용자 관찰:
> “diffusion guidance를 직접 주는 방향은 거의 없고, 아직까지는 대부분 prompt 최적화 + evaluator feedback 구조인 것 같다”

내 최종 정리는 다음과 같다.

### 짧은 답
- **일반적인 T2I jailbreak / commercial API red-teaming** 에서는 이 관찰이 **대체로 맞다**.
- 다만 **unlearning defense를 깨는 맥락** 에서는 prompt만이 아니라 **latent optimization** 이 강한 대안 축으로 등장한다.

즉,
- **주류 쪽**: prompt-side optimization / feedback-driven red-teaming
- **예외적이지만 중요**: latent-side attack on unlearned diffusion

---

## 2. 세 논문의 위치를 한 줄씩 정리
### 2505.21074 RPG-RT
- **가장 evaluator/feedback-loop에 가깝다**
- iterative prompt modification
- system feedback → rule-based preference modeling
- DPO fine-tuning으로 attacker가 target system에 적응

### 2404.02928 JPA
- **가장 전형적인 prompt optimization 논문**
- embedding space에서 NSFW concept direction 탐색
- discrete prefix prompt optimization
- semantic fidelity + speed를 강조

### 2602.00175 IVO
- **가장 latent-side 공격에 가깝다**
- unlearned diffusion의 dormant memory를 initial latent optimization으로 재활성화
- 일반 API jailbreak라기보다 unlearning robustness 공격 성격이 강함

---

## 3. “evaluator를 두고 prompt를 깎는다”는 관찰은 어디까지 맞나?
### 맞는 부분
RPG-RT는 거의 정확히 그 그림이다.
- prompt 수정
- target response 관찰
- harmfulness / semantic similarity 평가
- preference 생성
- attacker 업데이트

즉, 사용자의 서술은 RPG-RT를 잘 포착한다.

### 조금 수정이 필요한 부분
JPA는 evaluator를 쓰긴 하지만,
핵심은 iterative feedback adaptation이 아니라
**embedding-guided optimization** 이다.

즉 JPA는
- “계속 feedback 받고 attacker를 바꾼다” 보다는
- “embedding 공간에서 목표 concept를 잡고, 그걸 discrete prompt로 역투영한다”
에 가깝다.

### 명확히 다른 부분
IVO는 거의 이 프레임 밖에 있다.
- prompt refinement가 메인이 아님
- latent가 메인 공격 변수
- target도 일반 safety checker가 아니라 **unlearned DM**

그래서 IVO는 사용자의 일반화에 대한 **중요한 예외**로 보는 게 좋다.

---

## 4. 공격 surface 기준으로 정리
### A. Prompt surface
- 대표: **JPA**
- 핵심 질문: 어떤 프롬프트/프리픽스를 넣으면 unsafe concept를 semantic fidelity 있게 끌어올 수 있는가?
- 장점: black-box 친화적, 상용 API에 현실적
- 한계: internal unlearning/latent-level defense가 강하면 ceiling 존재

### B. Prompt + feedback/adaptation surface
- 대표: **RPG-RT**
- 핵심 질문: target system의 비공개 방어 패턴을 interaction으로 학습할 수 있는가?
- 장점: unknown defense / commercial API setting에 강함
- 한계: query cost, evaluator design 의존성

### C. Latent surface
- 대표: **IVO**
- 핵심 질문: prompt 대신 latent를 조절해 dormant memory를 깨울 수 있는가?
- 장점: unlearning robustness에 매우 강력
- 한계: 일반 end-user black-box jailbreak threat model과는 거리 있음

---

## 5. 논문별 핵심 차이
| 관점 | RPG-RT | JPA | IVO |
|---|---|---|---|
| 주된 공격 변수 | prompt modification + feedback loop | prefix prompt optimization | initial latent optimization |
| target setting | commercial black-box T2I/API | black-box T2I + online services | unlearned diffusion models |
| 핵심 메커니즘 | rule-based preference + DPO | CLIP/text embedding direction + discrete search | DDIM inversion + latent optimization |
| evaluator 역할 | 매우 핵심 | 보조적/평가 중심 | 성능 측정 및 objective 일부 |
| 사용자의 관찰과의 부합도 | 매우 높음 | 중간 | 낮음 |

---

## 6. 연구적으로 중요한 해석
### 6.1 commercial red-teaming 관점
실제 상용 서비스에서는 내부 접근이 불가하므로,
당분간은 prompt-side 공격이 계속 중심일 가능성이 높다.
이때 더 강해지는 방향은
- 더 좋은 prompt search
- 더 좋은 feedback modeling
- 더 좋은 adaptive loop
이다.

이 점에서 RPG-RT는 꽤 자연스러운 진화 방향이다.

### 6.2 unlearning 평가 관점
하지만 unlearning robustness를 진지하게 따질 때는,
prompt-side 공격만 보면 defense를 과대평가할 수 있다.
IVO는 바로 그 점을 찌른다.

즉,
- prompt attack이 약하다고 해서 truly forgotten은 아님
- latent-level trigger가 있으면 dormant memory가 다시 살아날 수 있음

이건 unlearning 논문 읽을 때 중요한 포인트다.

---

## 7. 내 최종 판단
### 사용자의 원래 직관에 대한 판정
**대체로 맞다.**
다만 더 정교하게 말하면:

1. **일반 jailbreak** 에서는 아직도 prompt optimization이 중심축이다.
2. 그 안에서도 최근에는 단순 휴리스틱보다 **feedback-aware / evaluator-aware 구조** 가 강해지는 흐름이 있다.
3. 하지만 **unlearning robustness** 문맥으로 가면 latent optimization 같은 비-prompt 축이 강하게 등장한다.

즉,
> “대부분 prompt를 깎는다”는 건 맞고,
> “그중 일부는 evaluator-feedback loop까지 간다”도 맞고,
> “latent 공격은 아직 예외적이지만 무시하면 안 된다”가 최종 정리다.

---

## 8. 실무적으로 어떤 논문을 기준점으로 삼을까?
- **실제 black-box 상용 API red-teaming** 을 생각하면: **RPG-RT**
- **빠르고 강한 prompt-space baseline** 을 생각하면: **JPA**
- **unlearning defense가 진짜로 지웠는지 검증** 하려면: **IVO**

---

## 9. 추천 메모
앞으로 이 주제로 related work를 더 묶을 때는 아래처럼 분류하면 깔끔하다.

1. **Prompt-only jailbreaks**
   - JPA 류
2. **Feedback/adaptive red-teaming**
   - RPG-RT 류
3. **Latent / internal-state attacks on unlearning**
   - IVO 류

이렇게 나누면 “왜 대부분 prompt 계열처럼 보이는지”와 “왜 latent 계열이 따로 중요한지”가 동시에 설명된다.
