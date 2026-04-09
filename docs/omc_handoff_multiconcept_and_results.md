# OMC Handoff: Multi-Concept Plumbing and Result Reading

This note exists so a later OMC/OMX session can immediately understand what
changed, why it changed, and which files now matter.

## Why this handoff exists

Two practical issues were blocking non-GPU progress:

1. **Multi-concept code paths existed but were not safely usable**
   - `generate_v14.py` ~ `generate_v19.py` treated `ConceptPack` like a dict.
   - The loader returned a dataclass, so multi-concept execution could fail at runtime.

2. **Existing experiment outputs were hard to aggregate**
   - many directories contain legacy files such as:
     - `results.txt`
     - `categories_qwen2_vl.json`
   - but newer readers preferred:
     - `results_qwen_nudity.txt`
     - `categories_qwen_nudity.json`

The work in this handoff fixes those two issues **without launching new GPU jobs**.

---

## Files changed and what they now do

| File | What changed | Why it matters |
| --- | --- | --- |
| `CAS_SpatialCFG/concept_pack_loader.py` | Added compatibility accessors (`get`, `__getitem__`, `__contains__`) plus `target_concepts`, `anchor_concepts`, `target_words`, prompt loading, metadata passthrough | Makes existing multi-concept generator code usable without rewriting every `generate_v14~v19.py` file |
| `vlm/result_paths.py` | Added legacy Qwen2/`results.txt` candidates | Lets readers discover historical nudity-judge outputs |
| `scripts/lib/repo_env.sh` | Added fallback detection for `results.txt` and `categories_qwen2_vl.json` | Shell scripts can now recognize both canonical and legacy Qwen results |
| `vlm/opensource_vlm_nudity.py` | Now writes canonical nudity filenames while keeping minimal legacy compatibility for Qwen2-VL | Future outputs become readable by the active contract |
| `scripts/aggregate_nudity_results.py` | Rebuilt result discovery/parsing around canonical + legacy files | Existing outputs can be aggregated again |
| `CAS_SpatialCFG/scripts/analyze_results.py` | Rebuilt analysis to read legacy and canonical judge results, plus both SR and Relevant_SR | Version comparison works again |
| `CAS_SpatialCFG/scripts/run_remaining_siml04.sh` | Uses repo env helpers and smarter Qwen-result detection | Avoids re-evaluating directories that already have legacy judge outputs |
| `CAS_SpatialCFG/scripts/run_v14.sh` | Uses repo env helpers, canonical judge call, and fixed aggregate invocation | New v14 runs align better with the active result contract |
| `CAS_SpatialCFG/scripts/report_gridsearch_best_configs.py` | Builds best-config and Pareto reports from existing outputs | Gives OMC an immediate snapshot of what currently looks best without running GPUs |
| `CAS_SpatialCFG/scripts/check_concept_pack_completeness.py` | Audits concept-pack readiness and missing files/artifacts | Makes multi-concept readiness visible at a glance |
| `CAS_SpatialCFG/prepare_multi_concept.py` | Normalized pack-by-pack optional-artifact prep entrypoint | Makes exemplar / .pt artifact prep coherent when GPU time is available |
| `docs/omc_reports/current_optional_artifact_prep_playbook.md` | Operator-facing playbook for preparing optional concept-pack artifacts | Gives OMC a direct next-step recipe without re-reading all code |
| `tests/test_concept_pack_loader.py` | Added loader compatibility regression test | Protects the multi-concept compatibility shim |
| `tests/test_result_contract.py` | Added legacy `results.txt` parsing coverage | Protects the result parser against historical output formats |
| `tests/test_result_paths.py` | Added legacy Qwen2 candidate coverage | Protects output discovery logic |
| `tests/test_gridsearch_reports.py` | Added grid-search report coverage | Protects the best-config/Pareto reporting path |
| `tests/test_concept_pack_completeness.py` | Added concept-pack audit coverage | Protects the completeness checker |
| `tests/test_opensource_vlm_nudity.py` | Added canonical output-writing coverage | Protects canonical + legacy save behavior |

---

## What now works

### 1. Multi-concept loading

Existing generator code such as:

```python
pack.get("target_concepts")
"concept_directions" in pack
pack["anchor_directions"]
```

now works against `ConceptPack` objects loaded from:

```text
docs/neurips_plan/multi_concept/concept_packs/<concept>/
```

This is a compatibility layer, not a full redesign.

### 2. Legacy results are readable again

Readers now understand these legacy output pairs:

- `results.txt`
- `categories_qwen2_vl.json`

alongside the canonical files:

- `results_qwen_nudity.txt`
- `categories_qwen_nudity.json`

### 3. Version analysis now recovers SR

Use:

```bash
python3 CAS_SpatialCFG/scripts/analyze_results.py v14 v15
```

This now reports:

- `NN%`
- `SR%`
- `RelSR%`
- trigger rate
- mask area

instead of leaving SR as `N/A` for legacy outputs.

### 4. Ad-hoc aggregation works from the repo root

Example:

```bash
python3 scripts/aggregate_nudity_results.py \
  CAS_SpatialCFG/outputs/v14/ringabell_image_dag_adaptive_ss5.0_st0.2 \
  CAS_SpatialCFG/outputs/v15/ringabell_text_dag_adaptive_ss5.0_st0.2_np16
```

### 5. OMC-friendly report artifacts can be regenerated

```bash
python3 CAS_SpatialCFG/scripts/report_gridsearch_best_configs.py
python3 CAS_SpatialCFG/scripts/check_concept_pack_completeness.py
```

These scripts write intuitive artifacts under:

```text
docs/omc_reports/
  current_gridsearch_best_config_summary.md
  current_gridsearch_config_index.csv
  current_gridsearch_pareto_frontier.csv
  current_gridsearch_pareto_frontier.md
  current_concept_pack_completeness.md
  current_concept_pack_completeness.json
```

---

## Verification already done

- `pytest -q` → passing
- `python3 -m py_compile` on the main edited Python files → passing
- direct readback of existing `v14` / `v15` outputs via:
  - `CAS_SpatialCFG/scripts/analyze_results.py`
  - `scripts/aggregate_nudity_results.py`

---

## Remaining limits / risks

1. **This does not create missing judge outputs**
   - if a directory has no Qwen result at all, SR will still be unavailable.

2. **Multi-concept compatibility is plumbing-level**
   - loader/runtime compatibility was fixed
   - but no new GPU generation was run in this pass.

3. **Optional concept-pack artifacts are still not prepared**
   - every current pack still lacks `concept_directions.pt`, `clip_patch_tokens.pt`, `contrastive_embeddings.pt`, and `exemplar_images/`
   - so packs are now text/metadata-complete, but not image-artifact-complete
4. **No GPU work was launched in this pass**
   - the pipeline is now normalized and documented
   - actual exemplar/image/precomputed artifact generation is still pending execution time

---

## Recommended next non-GPU steps

1. Generate version-level comparison tables from current outputs
2. Add Pareto-style ranking for `NN` vs `SR`
3. Add a small inspection script for concept-pack completeness
4. If needed, standardize the remaining `run_v15~v19*.sh` scripts onto the same env/result helpers
5. Fill missing `harassment` / `hate` concept-pack required files

Update:
- items **1–4 are now done**
- item **5 is also done at the required text/metadata layer**
- the main remaining non-GPU gap is optional image/precomputed artifact preparation for all packs
