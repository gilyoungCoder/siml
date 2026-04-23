# ML Figure Agent Guidelines

## Purpose
`ml-figure` is the repo-native OMX counterpart to the earlier OMC figure workflow: read the research context deeply, identify the exact visual claim, and produce a reproducible publication-quality figure rather than a one-off sketch.

## Repo-specific grounding
Before drawing, the agent should anchor on the current `unlearning` story:
- training-free unsafe concept removal for T2I diffusion models
- current paper narrative: **Example-Based Spatial Guidance (EBSG)** / **When–Where–How**
- evidence sources live in `paper_neurips2026_sync/`, `docs/`, `vlm/`, and active method code under `CAS_SpatialCFG/`
- current camera-ready handoff and formatting cautions live in `docs/omx_handoff_2026-04-24_camera_ready.md`

## Source-of-truth priority
When facts conflict, use this order:
1. latest `paper_neurips2026_sync` source files
2. latest directly supporting result/docs files (`docs/v2_final_results.md`, current table `.tex`, current figure scripts)
3. older planning / OMC notes only as historical context

Do **not** pull old exploratory numbers into a camera-ready figure unless the current paper or latest result docs still support them.

## Figure families

### 1. Method / pipeline diagrams
Use when the goal is to explain mechanism.
- Read the abstract, introduction, and method sections first.
- Preserve the paper's terminology exactly (`CAS`, `When`, `Where`, `How`, `family`, `anchor`, `hybrid`).
- Prefer clean flow and minimal text over dense blocks.
- Reuse established visual semantics when possible:
  - safe / decision / mask / probe colors
  - family or modality distinctions that are already used in the repository

### 2. Quantitative plots
Use when the goal is to show results, tradeoffs, or ablations.
- Pull numbers from current tables or generated result artifacts, not from memory.
- Make axis labels and legend text paper-ready.
- Avoid cherry-picking scales that exaggerate gains.
- If a point/table row is preliminary or missing, label it honestly or omit it.

### 3. Qualitative grids
Use when the goal is visual comparison across methods/prompts.
- Keep prompt/method ordering stable and readable.
- Ensure captions match the actual source images.
- Avoid mixing images from different experiment settings without explicit labeling.

### 4. Appendix / diagnostic figures
Use when the figure supports an ablation or a reviewer-facing clarification.
- Bias toward clarity and traceability over visual flourish.
- Include enough labeling that the figure can stand alone in the appendix.

## Output defaults
- Keep source code in version control whenever possible (`.py`, `.tex`, or other text-based generators).
- Default destinations:
  - `docs/figures/` for shared figure source + preview assets
  - `paper_neurips2026_sync/figures/` for assets wired into the synced paper
- Prefer vector-first export for paper figures (`.pdf` first, `.png` preview optional)
- If stochastic decoration is used, set a deterministic seed

## Style defaults
- Paper-facing figures: serif typography, restrained palette, compact but legible labels
- Presentation-facing figures: can be slightly bolder, but still technically grounded
- Preserve readability in grayscale / print when possible
- Do not overcrowd with unnecessary arrows, gradients, or legends
- Explain the figure's main message through layout, not caption rescue

## Verification checklist
Before claiming a figure is ready:
1. The source script runs cleanly.
2. The expected output files exist and are non-empty.
3. Labels, legends, and numbers match the source-of-truth files.
4. The figure is legible at its target size (paper column / full-width / slide).
5. If it is paper-facing, filenames and paths are stable for LaTeX inclusion.

## Deliverable contract
A good `ml-figure` task should usually leave behind:
- the source figure file
- the rendered asset(s)
- a short regeneration note or command
- an evidence note describing where the numbers / labels came from
