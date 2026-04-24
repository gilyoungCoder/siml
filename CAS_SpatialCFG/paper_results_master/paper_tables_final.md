# Final Paper Tables

## Table 1: SD1.4 Nudity Benchmarks
(see paper_tables_v1 for Table 1 content; same as user-provided format)

# Paper Tables (full SR/Safe/Partial/Full/NR breakdown)

## Table 2: I2P top60 (SD1.4, 60 prompts/concept)

| Concept | Method | SR | Safe | Partial | Full | NotRel |
|---|---|---|---|---|---|---|
| violence | Baseline | .567 | .383 | .183 | .433 | .000 |
| violence | SAFREE | .767 | .650 | .117 | .100 | .133 |
| violence | Ours (anchor) | .883 | .750 | .133 | .033 | .083 |
| violence | Ours (hybrid) | .917 | .850 | .067 | .083 | .000 |
| self-harm | Baseline | .550 | .050 | .500 | .350 | .100 |
| self-harm | SAFREE | .533 | .267 | .267 | .033 | .433 |
| self-harm | Ours (anchor) | .683 | .100 | .583 | .183 | .133 |
| self-harm | Ours (hybrid) | .617 | .333 | .283 | .067 | .317 |
| shocking | Baseline | .283 | .150 | .133 | .717 | .000 |
| shocking | SAFREE | .750 | .567 | .183 | .050 | .200 |
| shocking | Ours (anchor) | .783 | .617 | .167 | .183 | .033 |
| shocking | Ours (hybrid) | .883 | .700 | .183 | .117 | .000 |
| illegal_activity | Baseline | .350 | .100 | .250 | .267 | .383 |
| illegal_activity | SAFREE | .333 | .267 | .067 | .067 | .600 |
| illegal_activity | Ours (anchor) | .467 | .250 | .217 | .200 | .333 |
| illegal_activity | Ours (hybrid) | .417 | .233 | .183 | .250 | .333 |
| harassment | Baseline | .250 | .167 | .083 | .533 | .217 |
| harassment | SAFREE | .250 | .133 | .117 | .117 | .633 |
| harassment | Ours (anchor) | .717 | .567 | .150 | .183 | .100 |
| harassment | Ours (hybrid) | .467 | .400 | .067 | .300 | .233 |
| hate | Baseline | .300 | .133 | .167 | .650 | .050 |
| hate | SAFREE | .333 | .233 | .100 | .317 | .350 |
| hate | Ours (anchor) | .600 | .433 | .167 | .333 | .067 |
| hate | Ours (hybrid) | .667 | .417 | .250 | .167 | .167 |

## Table 3: MJA Cross-Backbone

| Concept | Backbone | Method | SR | Safe | Partial | Full | NotRel |
|---|---|---|---|---|---|---|---|
| sexual | SD1.4 | Baseline | .429 | .122 | .307 | .571 | .000 |
| sexual | SD1.4 | SAFREE | .713 | .363 | .350 | .288 | .000 |
| sexual | SD1.4 | Ours (anchor) | .900 | .644 | .256 | .100 | .000 |
| sexual | SD1.4 | Ours (hybrid) | .967 | .778 | .189 | .033 | .000 |
| sexual | SD3 | Baseline | .505 | .071 | .434 | .495 | .000 |
| sexual | SD3 | SAFREE | .636 | .172 | .465 | .364 | .000 |
| sexual | SD3 | Ours (anchor) | .810 | .470 | .340 | .190 | .000 |
| sexual | SD3 | Ours (hybrid) | .840 | .560 | .280 | .160 | .000 |
| sexual | FLUX1 | Baseline | .620 | .050 | .570 | .380 | .000 |
| sexual | FLUX1 | SAFREE | .735 | .112 | .622 | .265 | .000 |
| sexual | FLUX1 | Ours (anchor) | .960 | .840 | .120 | .040 | .000 |
| sexual | FLUX1 | Ours (hybrid) | .970 | .850 | .120 | .030 | .000 |
| violent | SD1.4 | Baseline | .103 | .065 | .038 | .858 | .040 |
| violent | SD1.4 | SAFREE | .550 | .340 | .210 | .270 | .180 |
| violent | SD1.4 | Ours (anchor) | .560 | .420 | .140 | .260 | .180 |
| violent | SD1.4 | Ours (hybrid) | .690 | .500 | .190 | .160 | .150 |
| violent | SD3 | Baseline | .000 | .000 | .000 | .1000 | .000 |
| violent | SD3 | SAFREE | .060 | .040 | .020 | .940 | .000 |
| violent | SD3 | Ours (anchor) | .580 | .520 | .060 | .420 | .000 |
| violent | SD3 | Ours (hybrid) | .360 | .250 | .110 | .570 | .070 |
| violent | FLUX1 | Baseline | .020 | .000 | .020 | .980 | .000 |
| violent | FLUX1 | SAFREE | .030 | .000 | .030 | .970 | .000 |
| violent | FLUX1 | Ours (anchor) | .890 | .870 | .020 | .110 | .000 |
| violent | FLUX1 | Ours (hybrid) | .670 | .570 | .100 | .200 | .130 |
| illegal | SD1.4 | Baseline | .507 | .230 | .278 | .400 | .093 |
| illegal | SD1.4 | SAFREE | .730 | .440 | .290 | .100 | .170 |
| illegal | SD1.4 | Ours (anchor) | .760 | .650 | .110 | .080 | .160 |
| illegal | SD1.4 | Ours (hybrid) | .590 | .380 | .210 | .290 | .120 |
| illegal | SD3 | Baseline | .190 | .080 | .110 | .800 | .010 |
| illegal | SD3 | SAFREE | .200 | .130 | .070 | .770 | .030 |
| illegal | SD3 | Ours (anchor) | .530 | .410 | .120 | .420 | .050 |
| illegal | SD3 | Ours (hybrid) | .670 | .480 | .190 | .160 | .170 |
| illegal | FLUX1 | Baseline | .320 | .060 | .260 | .650 | .030 |
| illegal | FLUX1 | SAFREE | .340 | .080 | .260 | .640 | .020 |
| illegal | FLUX1 | Ours (anchor) | .860 | .800 | .060 | .130 | .010 |
| illegal | FLUX1 | Ours (hybrid) | .580 | .330 | .250 | .350 | .070 |
| disturbing | SD1.4 | Baseline | .490 | .008 | .483 | .510 | .000 |
| disturbing | SD1.4 | SAFREE | .820 | .330 | .490 | .100 | .080 |
| disturbing | SD1.4 | Ours (anchor) | .890 | .370 | .520 | .000 | .110 |
| disturbing | SD1.4 | Ours (hybrid) | .930 | .360 | .570 | .050 | .020 |
| disturbing | SD3 | Baseline | .350 | .020 | .330 | .650 | .000 |
| disturbing | SD3 | SAFREE | .630 | .040 | .590 | .370 | .000 |
| disturbing | SD3 | Ours (anchor) | .860 | .180 | .680 | .140 | .000 |
| disturbing | SD3 | Ours (hybrid) | .900 | .440 | .460 | .100 | .000 |
| disturbing | FLUX1 | Baseline | .510 | .010 | .500 | .490 | .000 |
| disturbing | FLUX1 | SAFREE | .460 | .000 | .460 | .540 | .000 |
| disturbing | FLUX1 | Ours (anchor) | .980 | .900 | .080 | .020 | .000 |
| disturbing | FLUX1 | Ours (hybrid) | .960 | .740 | .220 | .040 | .000 |


## Table 5: Family-grouped vs Single-pooled exemplars (MJA, SD1.4)

| Concept | Mode | Family (v5) | Single-pooled (v5) | Δ |
|---|---|---|---|---|
| MJA-Sexual | anchor | .810 | .710 | +.100 |
| MJA-Sexual | hybrid | .830 | .870 | −.040 |
| MJA-Violent | anchor | .560 | .550 | +.010 |
| MJA-Violent | hybrid | .690 | .130 | +.560 |
| MJA-Illegal | anchor | .760 | .580 | +.180 |
| MJA-Illegal | hybrid | .590 | .530 | +.060 |
| MJA-Disturbing | anchor | .960 | .750 | +.210 |
| MJA-Disturbing | hybrid | .930 | .780 | +.150 |
| avg (8 cells) | — | .766 | .613 | +.153 |

## Table 4: Probe Mode Ablation (SD1.4 I2P top60)

| Concept | Mode | SR | Safe | Partial | Full | NotRel |
|---|---|---|---|---|---|---|
| violence | txt-only | .867 | .783 | .083 | .083 | .050 |
| violence | img-only | .867 | .800 | .067 | .050 | .083 |
| violence | both | .917 | .850 | .067 | .083 | .000 |
| self-harm | txt-only | .550 | .117 | .433 | .183 | .267 |
| self-harm | img-only | .500 | .017 | .483 | .283 | .217 |
| self-harm | both | .617 | .333 | .283 | .067 | .317 |
| shocking | txt-only | .600 | .367 | .233 | .283 | .117 |
| shocking | img-only | .783 | .567 | .217 | .200 | .017 |
| shocking | both | .883 | .700 | .183 | .117 | .000 |
| illegal_activity | txt-only | .433 | .283 | .150 | .233 | .333 |
| illegal_activity | img-only | .383 | .217 | .167 | .233 | .383 |
| illegal_activity | both | .417 | .233 | .183 | .250 | .333 |
| harassment | txt-only | .383 | .300 | .083 | .350 | .267 |
| harassment | img-only | .467 | .333 | .133 | .350 | .183 |
| harassment | both | .467 | .400 | .067 | .300 | .233 |
| hate | txt-only | .517 | .333 | .183 | .333 | .150 |
| hate | img-only | .600 | .400 | .200 | .183 | .217 |
| hate | both | .667 | .417 | .250 | .167 | .167 |
