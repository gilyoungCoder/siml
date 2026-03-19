#!/usr/bin/env python
"""
Extract prompts from CSV files and save as txt files.
Supports various CSV formats:
- nudity-ring-a-bell.csv: 'sensitive prompt' column
- mma-diffusion-nsfw-adv-prompts.csv: 'adv_prompt' column
- nudity.csv: 'prompt' column
"""

import pandas as pd
import argparse
from pathlib import Path

def extract_prompts(csv_path: str, output_path: str = None, column: str = None, limit: int = None):
    """Extract prompts from CSV file."""
    csv_path = Path(csv_path)

    # Auto-detect column based on filename
    if column is None:
        if 'ring-a-bell' in csv_path.name:
            column = 'sensitive prompt'
        elif 'mma-diffusion' in csv_path.name:
            column = 'adv_prompt'
        elif csv_path.name == 'nudity.csv':
            column = 'prompt'
        else:
            # Try common column names
            df = pd.read_csv(csv_path, nrows=1)
            for col in ['prompt', 'text', 'sensitive prompt', 'adv_prompt', 'target_prompt']:
                if col in df.columns:
                    column = col
                    break
            if column is None:
                column = df.columns[0]

    print(f"Reading {csv_path}")
    print(f"Using column: '{column}'")

    df = pd.read_csv(csv_path)

    if column not in df.columns:
        print(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Column '{column}' not found in CSV")

    prompts = df[column].dropna().tolist()

    if limit:
        prompts = prompts[:limit]

    # Clean prompts
    cleaned = []
    for p in prompts:
        p = str(p).strip()
        if p and p.lower() != 'nan':
            cleaned.append(p)

    print(f"Extracted {len(cleaned)} prompts")

    # Output path
    if output_path is None:
        output_path = csv_path.with_suffix('.txt')
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for p in cleaned:
            f.write(p + '\n')

    print(f"Saved to {output_path}")
    return cleaned

def main():
    parser = argparse.ArgumentParser(description='Extract prompts from CSV files')
    parser.add_argument('csv_path', type=str, help='Path to CSV file')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output txt file path')
    parser.add_argument('--column', '-c', type=str, default=None, help='Column name containing prompts')
    parser.add_argument('--limit', '-n', type=int, default=None, help='Limit number of prompts')

    args = parser.parse_args()
    extract_prompts(args.csv_path, args.output, args.column, args.limit)

if __name__ == '__main__':
    main()
