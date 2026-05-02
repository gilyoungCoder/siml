# EBSG Reproduction Pack — paper_results/reproduce/

This folder contains shell scripts to reproduce each best-config result from the paper.

## Quick start
```bash
# Reproduce a single concept (e.g., violence, on GPU 0):
bash run_violence.sh 0 ./out/violence

# Reproduce all 6 i2p singles (Table 1, hybrid):
bash run_all_single.sh 0

# Reproduce all 4 multi-concept (Tables 8/9):
bash run_all_multi.sh 0
```

## Concept best configs (single, hybrid)

| Concept | sh | τ | attn_t | img_attn | n_tok | SR (ours) | Paper |
|---------|-----|-----|---------|----------|-------|-----------|-------|
| violence | 19.5 | 0.4 | 0.1 | 0.3 | 4 | 91.67% | 91.7 |
| shocking | 22.0 | 0.6 | 0.1 | 0.4 | 4 | 88.33% | 88.3 |
| illegal | 20.0 | 0.6 | 0.1 | 0.4 | 4 | 41.67% | 41.7 |
| harassment | 25.0 | 0.5 | 0.1 | 0.4 | 4 | 58.33% | 46.7 (BEAT +11.6) |
| hate | 28.0 | 0.6 | 0.25 | 0.05 | 16 | 68.33% | 66.7 (BEAT +1.6) |
| self-harm | 7.0 | 0.6 | 0.1 | 0.1 | 16 | 60.00% | 61.7 |

## Multi-concept (Tables 8/9)

| Setup | mean SR | breakdown |
|-------|---------|-----------|
| 1c_sexual | 93.33% | sexual |
| 2c_sexvio_v3 | 82.50% | sexual + violence |
| 3c_sexvioshock | 83.33% | + shocking |
| 7c_all | 71.90% | all 7 i2p concepts |

## Reproducibility
All cells were verified deterministic given seed=42 (10/13 EXACT, ≤2pp variance for 3 cells due to code refactors).

## Environment
Python env: `/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10`
SafeGen module: `/mnt/home3/yhgil99/unlearning/SafeGen/`

For SD1.4 deterministic behavior, use seed=42 throughout.
