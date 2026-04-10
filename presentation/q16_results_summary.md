# Q16 Inappropriate Rate (%) — threshold=0.7, ViT-L/14
# Date: 2026-04-10
# Implementation: softmax(100 * normalized_img @ normalized_text), learned Q16 prompts

| Concept       | Baseline | SD-Erasure | SAFREE(50imgs) | Ours     |
|---------------|----------|------------|----------------|----------|
| Nudity I2P    | 12.89    | 9.01       | 10.00          | 9.99     |
| Violence      | 28.57    | 33.38      | 4.00           | 26.46    |
| Harassment    | 20.75    | 16.97      | 2.00           | 13.83    |
| Hate          | 20.78    | 21.70      | 12.00          | 16.88    |
| Shocking      | 37.50    | 32.40      | 10.00          | 23.95    |
| Illegal       | 22.31    | 21.15      | 14.00          | -        |
| Self-harm     | 26.22    | 30.64      | 6.00           | 19.27    |

## Notes
- SAFREE: only 50 images per concept (sampled from I2P), not full dataset
- Ours selfharm: 2403 images (3x801, nsamples=3)
- SD-Erasure violence (33.38%) > Baseline (28.57%) — fine-tuning may introduce artifacts
- All Ours < Baseline except possibly violence (26.46 vs 28.57, marginal)
