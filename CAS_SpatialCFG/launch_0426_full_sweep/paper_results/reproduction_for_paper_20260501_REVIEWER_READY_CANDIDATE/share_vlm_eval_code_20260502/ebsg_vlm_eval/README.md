# EBSG Qwen3-VL v5 evaluation code

This package contains the VLM evaluation code used for EBSG paper result tables.
It is intentionally self-contained for the evaluator logic: the wrapper scripts and the canonical  dependency chain are included together.

## Contents

==== EVAL I2P concept=sexual gpu=0 ====
==== EVAL nudity benchmark=unlearndiff gpu=0 ====
==== EVAL I2P concept=sexual gpu=0 ====

## Environment

Use a Python environment with the Qwen3-VL/Qwen-VL dependencies installed. The original run used a dedicated VLM environment.

Typical packages:



Depending on the Qwen3-VL checkpoint/version, you may also need the correct  version recommended by Qwen.

## Quick start: evaluate one image folder



The output files are written inside :



## Supported concepts

Use these evaluator concept names:



The wrapper also accepts these aliases:



## Direct canonical v5 usage

You can also call the canonical evaluator directly:



## Summarize result files

After running evaluations, collect result text files into one CSV:



CSV columns:



## Config-based usage, optional

If you also have the EBSG reproduction configs, you can evaluate by config:



For all I2P configs:

==== EVAL I2P concept=sexual gpu=0 ====

For all nudity configs:

==== EVAL nudity benchmark=unlearndiff gpu=0 ====

## Notes

- This package contains evaluation code only, not generated images or model weights.
- The v5 rubric is the one used for the final EBSG paper numbers.
-  is a wrapper; it delegates to  through  so there is no stale duplicate rubric.
