# OMX Handoff — 2026-04-24 — NeurIPS 2026 paper camera-ready alignment

## 1. User request context
The user asked to:
- ssh into `siml-09`
- inspect the current paper at `paper_neurips2026_sync`
- compare its overall feel / organization to the accepted NeurIPS paper at `paper_refs/paper2`
- treat `CAS_SpatialCFG/Formatting_Instructions_For_NeurIPS_2026` and the NeurIPS 2026 handbook as the **absolute formatting authority**
- read recent OMC context deeply **before editing**
- then move toward camera-ready-compliant revision

Important user constraint:
- **Template/handbook first.** The accepted paper is a style reference only, not a rule source.

## 2. Files and sources already read
### Current paper
- `paper_neurips2026_sync/main.tex`
- `paper_neurips2026_sync/sec/0_abstract.tex`
- `paper_neurips2026_sync/sec/1_introduction.tex`
- `paper_neurips2026_sync/sec/2_related_work.tex`
- `paper_neurips2026_sync/sec/3_method.tex`
- `paper_neurips2026_sync/sec/4_experiments.tex`
- `paper_neurips2026_sync/sec/5_conclusion.tex`
- `paper_neurips2026_sync/sec/A_appendix.tex`
- all main `paper_neurips2026_sync/table/*.tex`
- multiple `paper_neurips2026_sync/figures/*.tex`
- `paper_neurips2026_sync/checklist.tex`
- `paper_neurips2026_sync/main.pdf` metadata

### Accepted reference paper
- `paper_refs/paper2/main.tex`
- `paper_refs/paper2/Manuscript/Introduction.tex`
- `paper_refs/paper2/Manuscript/Preliminary.tex`
- `paper_refs/paper2/Manuscript/Method.tex`
- `paper_refs/paper2/Manuscript/Related_works.tex`
- `paper_refs/paper2/Manuscript/Experiments.tex`
- `paper_refs/paper2/Manuscript/Conclusion.tex`
- `paper_refs/paper2/Appendix/Limitations.tex`
- structure of `Appendix/`, `Figures/`, `Manuscript/`

### NeurIPS 2026 formatting authority
- `CAS_SpatialCFG/Formatting_Instructions_For_NeurIPS_2026/neurips_2026.tex`
- `CAS_SpatialCFG/Formatting_Instructions_For_NeurIPS_2026/neurips_2026.sty`
- `CAS_SpatialCFG/Formatting_Instructions_For_NeurIPS_2026/checklist.tex`
- user-provided NeurIPS 2026 handbook text in chat

### Recent OMC / repo context already read
- `docs/launch_0420_status.md`
- `docs/backbone_comparison.md`
- `docs/v2_final_results.md`
- `docs/v2_experiment_results.md`
- `docs/omc_reports/2026-04-05_progress_report_for_notion.md`
- `docs/neurips_plan/master_plan.md`
- `docs/neurips_plan/current_problems.md`
- `docs/neurips_plan/README.md`
- `docs/presentation_20260410.html`
- `meeting_status_2026-04-10.md`
- `.omc/project-memory.json`
- `.omc/state/deep-interview-state.json`
- recent `.omc/sessions/*.json` (note: these are metadata-only, not transcripts)

## 3. Most important formatting conclusions (treat as hard constraints)
### 3.1 Camera-ready PDF order
Per handbook + template intent, the combined PDF should be ordered as:
1. main paper content
2. references
3. appendices
4. NeurIPS paper checklist

Interpretation for current LaTeX:
- checklist should come **after appendix**, not before appendix
- checklist must remain included in the final PDF

### 3.2 Page budget
- submission: 9 content pages
- camera-ready: **10 content pages**
- references / appendices / checklist do **not** count as content pages

### 3.3 Mode selection
Camera-ready should use NeurIPS 2026 final mode, i.e. main-track final mode rather than default submission mode.

### 3.4 Style-file authority
Anything that risks violating style constraints must be treated as suspicious, including:
- direct style/notice suppression
- format hacks that alter reserved first-page behavior
- float/table environment redefinitions that effectively tweak formatting parameters
- manual attempts to outsmart the template unless clearly necessary and compliant

## 4. Current-paper issues already identified
### 4.1 Non-final mode / submission residue
Current `paper_neurips2026_sync/main.tex` is still effectively submission-shaped rather than camera-ready-shaped.
Key symptoms identified:
- package line is not in explicit main-track final form
- title still contains submission-style footnote text (`Submitted to ... Do not distribute.`)

### 4.2 Style-hack risk
Current `main.tex` contains custom formatting intervention that may be risky under NeurIPS rules, including:
- `table*` environment patching
- manual float spacing overrides
- `\@notice` suppression

These should be audited against the template-first rule before keeping any of them.

### 4.3 Checklist currently not wired in final order
Current `main.tex` includes appendix directly after references and does not currently include the paper checklist in the compiled order.
The repository *does* contain `paper_neurips2026_sync/checklist.tex`, so this looks like an integration omission rather than missing source content.

### 4.4 Draft smell that should be removed for camera-ready
Multiple tables/captions still contain wording like:
- preliminary
- pending
- post-patch re-evaluation ongoing
- not yet collected
- re-sweep
- intentional omission due to current bug / retune notes

These are useful research notes but not polished camera-ready paper language.

### 4.5 Author block is placeholder-like
Current author block still looks placeholder / not final-camera-ready quality.

## 5. What to borrow from `paper_refs/paper2` (soft reference only)
Use `paper2` only as a stylistic reference for final-paper feel, not as a formatting rule source.
Useful takeaways:
- strong first-page visual anchoring
- polished accepted-paper tone
- cleaner final artifact feel (no obvious in-progress notes)
- calmer paragraph rhythm in intro and experiments
- appendix/checklist/acknowledgment fully integrated as a final package

Do **not** copy:
- any 2025-specific assumptions that conflict with 2026 template/handbook
- any style-file behavior that the 2026 template no longer wants

## 6. OMC-context conclusions that should guide editing
### 6.1 Narrative identity to preserve
Latest repo context suggests the paper should be framed around:
- **EBSG / Example-Based Spatial Guidance**
- **When–Where–How** decomposition
- family-grouped exemplar packs
- training-free concept erasure
- adversarial safety / jailbreak-style prompt robustness
- cross-backbone transfer (SD1.4 / SD3 / FLUX)

### 6.2 Important recent-story emphasis
Recent OMC artifacts emphasize:
- example controllability / user-editable packs
- metaphorical / jailbreak prompt handling (MJA etc.)
- multi-concept support as meaningful but should not be oversold beyond evidence
- scientific headline is stronger around adversarial unsafe-generation defense than around a generic broad “safety framework” claim

### 6.3 Source-of-truth rule for numbers / claims
Old OMC documents contain earlier method phases and evolving conclusions.
Do **not** blindly merge stale older claims into the current paper.
Use this priority order:
1. latest `paper_neurips2026_sync` source files
2. latest directly supporting recent docs / evaluations
3. older planning docs only for historical context / motivation

Particular caution:
- some older documents strongly favor one HOW mode globally
- current sync paper instead presents anchor vs hybrid as a concept-dependent tradeoff
- newest paper sync should be treated as the current narrative unless new evidence explicitly overrides it

## 7. Important note about OMC session files
Recent `.omc/sessions/*.json` files are mostly session metadata only.
They do **not** preserve rich transcript content.
So future agents should reconstruct intent primarily from:
- `docs/*.md`
- `docs/omc_reports/*.md`
- meeting notes / status docs
- current paper source itself
- `.omc/project-memory.json`

## 8. Work status at handoff time
### Already done
- read current paper / reference paper / 2026 template / handbook excerpts
- read recent OMC-related context and extracted main narrative constraints
- identified camera-ready-critical formatting issues

### Not done yet
- no TeX edits applied yet in this session
- no checklist reintegration yet
- no final-mode switch yet
- no compile / page-layout verification after edits yet

## 9. Recommended next execution order
1. edit `paper_neurips2026_sync/main.tex` to align with NeurIPS 2026 camera-ready structure
2. place sections in final PDF order: content -> refs -> appendix -> checklist
3. remove submission residue and risky style hacks where possible
4. clean draft-smell wording from tables/captions/notes
5. verify compiled PDF layout and first-page behavior
6. only after compliance is stable, do optional stylistic polishing toward `paper2` feel

## 10. Practical commands for the next agent
### Read this handoff
```bash
ssh siml-09 'cd /mnt/home3/yhgil99/unlearning && sed -n "1,240p" docs/omx_handoff_2026-04-24_camera_ready.md'
```

### Quick section scan
```bash
ssh siml-09 'cd /mnt/home3/yhgil99/unlearning && grep -n "^##" docs/omx_handoff_2026-04-24_camera_ready.md'
```

### Re-open main paper entrypoint
```bash
ssh siml-09 'cd /mnt/home3/yhgil99/unlearning/paper_neurips2026_sync && nl -ba main.tex | sed -n "1,220p"'
```

### Re-open template authority
```bash
ssh siml-09 'cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/Formatting_Instructions_For_NeurIPS_2026 && nl -ba neurips_2026.tex | sed -n "1,260p"'
```
