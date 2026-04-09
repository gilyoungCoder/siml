# NeurIPS 2026 Plan — Index

> 2026-04-02 생성. 5개 Opus 에이전트의 deep analysis + RALPLAN consensus 기반.

## 문서 목록

| # | 파일 | 내용 | 핵심 요약 |
|---|------|------|----------|
| 1 | [master_plan.md](master_plan.md) | **전체 워크플로우 개략도** | Step 1~8 실행 순서, 리스크 대응, CVPR 피드백 |
| 2 | [method_evolution.md](method_evolution.md) | **방법론 상세 (v3→v13 + v14~v19)** | 각 버전 WHEN/WHERE/HOW, 교훈, 신규 제안 6종 |
| 3 | [current_problems.md](current_problems.md) | **현재 직면 문제 8가지 + 대응** | SR gap, mask over-coverage, dag_adaptive HOW 통합 과제, 평가 미비 |
| 4 | [evaluation_plan.md](evaluation_plan.md) | **평가 지표 & 벤치마크 계획** | 6개 구현 완료, 5개 신규 필요, 논문 Table 구조 |
| 5 | [presentation_guide.md](presentation_guide.md) | **발표 자료 가이드** | Reveal.js 템플릿, 슬라이드 구조, 시각 자료 목록 |
| 6 | [multi_concept_pipeline.md](multi_concept/multi_concept_pipeline.md) | **I2P 다중 개념 확장 개요** | violence/shocking/harassment 확장 방향, 데이터셋 해석 |
| 7 | [violence_mapping_spec.md](multi_concept/violence_mapping_spec.md) | **violence target-anchor 매핑 spec** | 실제 I2P prompt 기반 cue family, keyword, exemplar pair, pack 구조 |
| 8 | [i2p_mapping_guideline.md](multi_concept/i2p_mapping_guideline.md) | **공통 추출 지침서** | target-anchor 선정 규칙, 1:1 mapping 평가 기준 |
| 9 | [mapping_docs_guide.md](multi_concept/mapping_docs_guide.md) | **문서 사용 설명서** | 어떤 문서를 어떤 순서로 읽고 어떻게 써야 하는지 |
| 10 | [sexual_shocking_mapping_spec.md](multi_concept/sexual_shocking_mapping_spec.md) | **sexual/shocking 매핑 spec** | sexual cue family, shocking cue family, exemplar pair |
| 11 | [selfharm_illegal_mapping_spec.md](multi_concept/selfharm_illegal_mapping_spec.md) | **self-harm/illegal 매핑 spec** | self-harm family, illegal activity family, pilot 추천 |
| 12 | [harassment_hate_mapping_spec.md](multi_concept/harassment_hate_mapping_spec.md) | **harassment/hate 매핑 spec** | label-noise 해석, weak CAS 구간, subset 전략 |
| 13 | [multiconcept_integration_spec.md](multi_concept/multiconcept_integration_spec.md) | **멀티컨셉 통합 구조 spec** | v3/v4/v14-v19와 concept pack 연결 방식 |

## Quick Decision Map

```
지금 뭐 해야 해?
  → master_plan.md > "실행 순서" 섹션

어떤 버전을 구현해야 해?
  → method_evolution.md > "신규 제안 v14-v19" + "추천 실행 조합"

왜 v7이 SR이 낮아?
  → current_problems.md > "Problem 1: v7의 Content Destruction"

평가를 어떻게 돌려?
  → evaluation_plan.md > "표준 실험 파이프라인"

발표 자료 어떻게 만들어?
  → presentation_guide.md > "다음 발표 슬라이드 구조"
```

## 확정된 방법론 (미팅 03-27)

**Image + Text Exemplar 기반 Example-based When + Where + How**

| WHEN | WHERE | HOW |
|------|-------|-----|
| CAS (cos similarity) | CLIP Image + Text CrossAttn Probe | Hybrid CFG Guidance |
| τ = 0.6 | Focused spatial mask | online target + exemplar anchor |

## 우선순위 실행 순서

1. **v14** (Hybrid WHERE Fusion) — 1-2h, v6+v7 조합 ⭐
2. **v18** (Timestep Adaptive) — 30min, v14와 결합 ⭐
3. **v15** (CLIP Patch Token Probe) — 4h, 핵심 이미지 임베딩 ⭐
4. **v16** (Contrastive Direction) — 4h
5. **v17** (IP-Adapter) — 1일
6. **v19** (Multi-Exemplar Ensemble) — 4h
