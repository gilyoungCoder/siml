# Repository Layout Guide

Use this lightweight map when navigating the repository.

## Recommended reading order

1. `README.md` — project summary and current orientation
2. `docs/repo-taxonomy.md` — active vs baseline vs legacy folders
3. `docs/active-workflow.md` — canonical current workflow
4. `docs/evaluation_alignment_and_mjbench.md` — current evaluation notes
5. `CLAUDE.md` — local research workflow guidance

## Canonical path for current work

### 1. Method development
- `CAS_SpatialCFG/`
- `SAFREE/`

### 2. Evaluation and aggregation
- `vlm/`

### 3. Multi-method orchestration
- `scripts/`

## Large baseline/comparison areas

- `AMG/`
- `SoftDelete+CG/`

## Historical branches

Most classifier-family and SDXL-family directories at the top level are best
treated as legacy branches unless the task explicitly targets them.

## Practical rule

If a task is ambiguous, start by checking:

- `CAS_SpatialCFG/` for current method logic
- `vlm/` for evaluation logic
- `scripts/` for end-to-end pipeline wiring
