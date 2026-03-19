# Dataset & Generation Specification

All baselines should use the following datasets, generation parameters, and output structure.

## Datasets

| Name | CSV Path | Prompt Column | # Prompts |
|------|----------|---------------|-----------|
| **Ring-A-Bell** | `SAFREE/datasets/nudity-ring-a-bell.csv` | `sensitive prompt` | 79 |
| **I2P (sexual)** | `guided2-safe-diffusion/prompts/i2p/sexual.csv` | `prompt` | 931 |
| **P4D-N** | `SAFREE/datasets/p4dn_16_prompt.csv` | `prompt` | 16 |
| **UnlearnDiff** | `SAFREE/datasets/unlearn_diff_nudity.csv` | `prompt` | ? |
| **MMA-Diffusion** | `SAFREE/datasets/mma-diffusion-nsfw-adv-prompts.csv` | `adv_prompt` | ? |

> All paths are relative to `/mnt/home/yhgil99/unlearning/`.

## Generation Parameters (Fixed)

```
Model:           CompVis/stable-diffusion-v1-4
Seed:            42
Inference Steps: 50
CFG Scale:       7.5
Samples/Prompt:  1
Resolution:      512x512 (SD default)
Safety Checker:  Disabled
```

## Output Structure

All outputs go under `SoftDelete+CG/scg_outputs/final_{dataset}/{method_name}/`.

```
scg_outputs/
├── final_ringabell/
│   ├── sd_baseline/        # Vanilla SD v1.4
│   ├── safree/             # SAFREE
│   ├── sld/                # (other baselines)
│   ├── esd/                # ...
│   └── ...
├── final_i2p/
│   └── (same structure)
├── final_p4dn/
│   └── (same structure)
├── final_unlearndiff/
│   └── (same structure)
└── final_mma/
    └── (same structure)
```

## Image Naming

Images should be named `{index:06d}.png` (e.g., `000000.png`, `000001.png`, ...).
The index corresponds to the row order in the CSV (0-indexed, header excluded).

## CSV Parsing Example (Python)

```python
import csv

def load_prompts(csv_path):
    """Load prompts from CSV. Handles all 5 datasets."""
    column_priority = [
        'adv_prompt',       # MMA-Diffusion
        'sensitive prompt', # Ring-A-Bell
        'prompt',           # I2P, P4D-N, UnlearnDiff
    ]

    prompts = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        col = next((c for c in column_priority if c in reader.fieldnames), None)
        if col is None:
            raise ValueError(f"No prompt column found. Available: {reader.fieldnames}")
        for row in reader:
            val = row[col].strip()
            if val:
                prompts.append(val)
    return prompts
```

## Evaluation

After generation, run Qwen3-VL nudity evaluation:

```bash
cd /mnt/home/yhgil99/unlearning
bash vlm/batch_eval_qwen_multi_gpu.sh \
    0,1,2,3 \
    nudity \
    SoftDelete+CG/scg_outputs/final_ringabell
```

This produces `results_qwen3_vl_nudity.txt` in each method subdirectory.
