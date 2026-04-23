# Style & Logical Structure Analysis — paper1 + paper2

**Purpose.** Pre-analyzed style/structure distillation for `ml-paper-writer` to consume directly. Avoids re-reading the source `.tex` files each session. Covers logical flow, section arcs, lexical patterns, and prose rhythm of two empirical concept-safety papers at ICLR 2025 / NeurIPS 2025. The user's paper does NOT use theorem→proof structure, so that axis is excluded here.

**Sources.**
- `paper1/` — ICLR 2025 "Training-Free and Adaptive Guard for Safe T2I/T2V Generation" (= arxiv-2410.12761v2).
- `paper2/` — NeurIPS 2025 "Safe Denoiser" concept-safety paper (= arxiv-2502.08011v6).

**How to use.** Follow these patterns as structural skeletons. Do NOT copy phrasing verbatim. Do NOT import the theorem/definition blocks present in paper2 — the user's paper is empirical.

---

## 1. Abstract arc (same skeleton, both papers)

```
[1] Field-growth hook .............. 1 sentence, recent-advances framing
[2] Risk / gap statement ........... 1 sentence, "but … risk of …"
[3] Existing-method critique ....... 1–3 sentences, often inline-numbered (1)(2)(3)
[4] Proposal announcement .......... "To address these challenges, we propose \ours{}, a
                                      [adjectives] approach to [task], that [key property]."
[5] Mechanism paragraph ............ 2–4 sentences naming bold components
[6] Balancing-trade-off sentence ... "To balance … between X and Y, \ours{} incorporates …"
[7] Empirical headline ............. "Empirically, \ours{} achieves state-of-the-art …
                                      (reducing X by Y% across N datasets)"
[8] Extension statement ............ "We further extend \ours{} to … showcasing its
                                      flexibility and generalization."
[9] Closing impact sentence ........ "As generative AI evolves, \ours{} provides …"
```

Paper1 abstract executes all 9 beats; paper2 is tighter (merges [5] and [6]) but same arc. Length: 12–18 sentences, dense. Sentences average ~25–30 words.

## 2. Introduction arc (both papers follow this)

```
[I-1]  Broad field hook with wide citations (T2I, audio, video, protein, etc.)
[I-2]  Specific applications / commercial tools (DALL·E 3, Sora, Midjourney …)
[I-3]  Risk statement with nuance (e.g., "the definition of unsafe varies …")
[I-4]  Existing-approach taxonomy, two or three families:
          - Family A (with critique of its limitation)
          - Family B (with critique)
          - Mention of a "promising alternative" family
[I-5]  Challenges remaining even in the promising family (numbered inline)
[I-6]  Urgent-need sentence: "Thus, there is an urgent need for …"
[I-7]  Proposal paragraph: "This paper presents \ours{}, a [properties] mechanism for
        [task] without [undesirable thing]."
[I-8]  Component preview — for each named component:
          - a sentence stating what it does
          - a sentence stating why this design (often starts with "Specifically" /
            "Rather than directly [X], \ours{} [Y]")
[I-9]  "In the end, \ours{} [unified consequence]."
[I-10] Empirical headline with exact numbers and dataset list
[I-11] Extension to more backbones / modalities
[I-12] Closing impact sentence
[I-13] Contributions list — 3 bullets. Each bullet leads with a bold phrase.
```

**Paper2 variation**: puts a figure (schematic + trajectory) at the top of §1 before any prose, with a caption longer than most paragraphs. The figure anchors [I-7].

## 3. Method-section transitions

Both papers chain subsections via explicit logical connectives, not just headings.

- Opening pattern: "We first … Then we propose … Additionally, we introduce …"
- Subsection hand-off: the last sentence of §3.X points forward ("Given the proximity, \ours{} aims to …") and the first sentence of §3.(X+1) picks it up ("To measure the proximity, we …").
- Equation introduction: the equation is always preceded by a sentence of the form "We formulate [quantity] as follows:" and followed by a sentence that reads the equation in English ("where Z is the … such that …").
- Heavy use of `\textit{}` for named primitives ("\textit{safe denoiser}", "\textit{unsafe concept subspace}"), bold for claims.

## 4. Lexical patterns to mirror

| Pattern | Example (from paper1 / paper2) |
|---|---|
| Method macro | `\ours{}` — never spell the method name in body after first definition |
| Bold for named components | **orthogonal projection**, **self-validating filtering**, **state-of-the-art** |
| Inline numbered critiques | "they (1) cannot instantly remove …, (2) depend on collected data, (3) alter weights" |
| "To address this / these challenges, we …" | — standard pivot to proposal |
| "Empirically, \ours{} achieves …" | — standard opener for empirical section of intro |
| "We further extend \ours{} to …" | — standard extension / generality claim |
| "In contrast, … cannot …" | — framing of limitation of alternatives |
| "Rather than directly [naive thing]—which [downside]—\ours{} [better thing]." | — standard design-motivation sentence |
| "(examples shown in~\Cref{fig:…})" | — parenthetical figure references mid-sentence |

## 5. Figures & captions

- Figures referenced by label in the first paragraph of the relevant section, not reserved for later.
- Captions follow "what the figure shows" + "what the reader should conclude" pattern. Paper2's captions are 3–5 sentences; paper1's are 2–3.
- Method figures (schematic + example trajectory side-by-side) placed as `figure*` at top of §Method.
- Tables: each main result table has a one-sentence caption and a bolded row/column for the method. Paper1 uses `\cmark`/`\xmark` (pifont) to indicate property coverage in comparison tables.

## 6. Related-work framing

- Both use a 1-page related-work section organized by family (Unlearning / Model Editing / Training-Free Filtering), not by citation chronology.
- Each paragraph: one topic sentence defining the family, one-sentence critique, one-sentence differentiation of this paper.
- Citations cluster in groups of 3–5 for each family, never one cite per sentence.

## 7. Experiments arc

```
[E-1] Datasets + metrics paragraph
[E-2] Baselines paragraph (named, with short 1-line method description each)
[E-3] Implementation details (hyperparams, compute, seeds — 1 paragraph)
[E-4] Main results table + 1 paragraph walkthrough emphasizing headline numbers
[E-5] Secondary-metric results
[E-6] Ablations (one subsection per axis) — each ablation has a short "why this matters" sentence
[E-7] Qualitative figure — curated examples with commentary
[E-8] Generalization / backbone experiments (if applicable)
```

## 8. Conclusion pattern (paper2 executes this cleanly)

```
[C-1] One-sentence method recap in the present tense.
[C-2] Positioning: "Unlike [alternative family], our approach [distinctive property]."
[C-3] Broader value claim: "Thus ours contributes to a [larger concept] suitable for
      [real-world setting]."
[C-4] Explicit limitation: "A current limitation is [what] … Appendix~\ref{…} discusses …"
[C-5] Future work: "We leave [specific direction] for future work."
```

Paper1's conclusion is shorter (one dense paragraph) but hits beats [C-1], [C-2], [C-5]. Paper2 adds [C-3] and [C-4]; for a NeurIPS paper that addresses reviewer concerns, include all five.

## 9. Hedging & tone

- **Strong claims with empirical anchors**: every "state-of-the-art" is followed by a number and dataset list.
- **Hedged limitations**: "remains an ongoing challenge", "partly due to", "may not generalize well".
- **No overhyping**: no "revolutionary", no "paradigm shift" — instead "strong baseline", "robust", "adaptable".
- **Present tense for method**, past tense for experiments, future tense only in [C-5] future-work.
- **"We" throughout** — no passive "it was shown that".

## 10. What NOT to import from these references

- **Paper2's `\newtheorem{theorem}` block + `\thmref{safe}`** — the user's paper is empirical, not theoretical. Skip any Theorem/Definition/Proof scaffolding entirely.
- **Paper2's Proof appendix** — same reason.
- **Paper1's `\updated{}` macro** — an editing-history artifact, not style.
- **Either paper's specific terminology** (`toxic concept subspace`, `safe denoiser`, etc.) — domain-overlapping but each paper has its own framing; the user should pick their own.
- **Bibliographies** — irrelevant to the paper being written.

## 11. One-line rubric

> Open broad, narrow fast, announce the method by a named macro, preview its components with bold lead-ins, commit to numbers and dataset lists early, extend/generalize at the end, and close with a 1–5-beat conclusion that is honest about limitations.

— end of STYLE_ANALYSIS.md —
