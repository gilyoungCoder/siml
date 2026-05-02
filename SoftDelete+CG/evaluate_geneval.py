"""
GenEval Score Evaluation Script
================================

Evaluates generated images using GenEval framework for compositional text-to-image alignment.

This script wraps the original GenEval repository (https://github.com/djghosh13/geneval)
to evaluate images with their corresponding prompts.

Usage:
    python evaluate_geneval.py \
        --image_dir ./outputs/images \
        --prompt_file ./prompts/sexual_50.txt \
        --geneval_path ./geneval \
        --output_file ./results/geneval_results.json

Requirements:
    - GenEval repository cloned and set up
    - Object detector models downloaded
    - Images named in a consistent format
"""

import argparse
import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import shutil


def load_prompts(prompt_file: str) -> List[str]:
    """
    Load prompts from a text file.

    Args:
        prompt_file: Path to text file with one prompt per line

    Returns:
        List of prompts
    """
    prompts = []
    with open(prompt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                prompts.append(line)
    return prompts


def create_geneval_metadata(
    image_dir: str,
    prompts: List[str],
    output_jsonl: str,
    image_pattern: str = "*.png"
) -> int:
    """
    Create metadata file in GenEval format.

    GenEval expects a JSONL file where each line is:
    {
        "prompt": "a red apple and a blue box",
        "evaluation_type": "colors",
        "image_path": "path/to/image.png",
        ...
    }

    Args:
        image_dir: Directory containing generated images
        prompts: List of text prompts
        output_jsonl: Output JSONL file path
        image_pattern: Glob pattern for images

    Returns:
        Number of entries created
    """
    from glob import glob

    # Find all images
    image_paths = sorted(glob(os.path.join(image_dir, image_pattern)))

    print(f"Found {len(image_paths)} images in {image_dir}")
    print(f"Loaded {len(prompts)} prompts")

    # Create metadata entries
    entries = []

    # Map images to prompts
    # Assuming naming convention: prompt000_sample0_*.png or similar
    for img_path in image_paths:
        img_name = os.path.basename(img_path)

        # Try to extract prompt index from filename
        # Common patterns: prompt000, prompt_0, p0, etc.
        import re
        match = re.search(r'prompt(\d+)', img_name)
        if match:
            prompt_idx = int(match.group(1))
        else:
            # Fallback: use order
            prompt_idx = len(entries) % len(prompts)

        if prompt_idx < len(prompts):
            prompt = prompts[prompt_idx]

            entry = {
                "prompt": prompt,
                "evaluation_type": "overall",  # GenEval will auto-detect task type
                "image_path": os.path.abspath(img_path),
                "prompt_index": prompt_idx,
            }
            entries.append(entry)

    # Write JSONL file
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

    print(f"Created metadata file with {len(entries)} entries: {output_jsonl}")
    return len(entries)


def run_geneval_evaluation(
    geneval_path: str,
    metadata_file: str,
    output_dir: str,
    detector_path: str = None,
) -> str:
    """
    Run GenEval evaluation using the official repository.

    Args:
        geneval_path: Path to cloned GenEval repository
        metadata_file: Path to metadata JSONL file
        output_dir: Directory to save results
        detector_path: Path to object detector model (optional)

    Returns:
        Path to results JSONL file
    """
    geneval_path = os.path.abspath(geneval_path)

    # Check if GenEval repo exists
    if not os.path.exists(geneval_path):
        raise FileNotFoundError(
            f"GenEval repository not found at {geneval_path}.\n"
            f"Please clone it first:\n"
            f"  git clone https://github.com/djghosh13/geneval.git {geneval_path}"
        )

    eval_script = os.path.join(geneval_path, "evaluation", "evaluate_images.py")
    if not os.path.exists(eval_script):
        raise FileNotFoundError(
            f"GenEval evaluation script not found at {eval_script}.\n"
            f"Make sure the repository is complete."
        )

    # Setup detector path
    if detector_path is None:
        detector_path = os.path.join(geneval_path, "models")

    if not os.path.exists(detector_path):
        print(f"WARNING: Detector path {detector_path} not found.")
        print("You may need to download models first:")
        print(f"  cd {geneval_path}")
        print(f"  ./evaluation/download_models.sh {detector_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "geneval_results.jsonl")

    # Run evaluation
    print(f"\n{'='*60}")
    print("Running GenEval evaluation...")
    print(f"{'='*60}\n")

    cmd = [
        "python",
        eval_script,
        metadata_file,
        "--outfile", results_file,
        "--model-path", detector_path,
    ]

    print(f"Command: {' '.join(cmd)}\n")

    try:
        # Change to GenEval directory for imports
        result = subprocess.run(
            cmd,
            cwd=geneval_path,
            check=True,
            capture_output=False,
            text=True,
        )
        print(f"\nEvaluation complete! Results saved to: {results_file}")
        return results_file

    except subprocess.CalledProcessError as e:
        print(f"ERROR: GenEval evaluation failed with exit code {e.returncode}")
        raise


def compute_summary_scores(
    geneval_path: str,
    results_file: str,
) -> Dict[str, Any]:
    """
    Compute summary GenEval scores from results.

    Args:
        geneval_path: Path to GenEval repository
        results_file: Path to results JSONL file

    Returns:
        Dictionary with summary scores
    """
    summary_script = os.path.join(geneval_path, "evaluation", "summary_scores.py")

    if not os.path.exists(summary_script):
        raise FileNotFoundError(f"Summary script not found: {summary_script}")

    print(f"\n{'='*60}")
    print("Computing summary scores...")
    print(f"{'='*60}\n")

    cmd = [
        "python",
        summary_script,
        results_file,
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=geneval_path,
            check=True,
            capture_output=True,
            text=True,
        )

        # Parse output
        output = result.stdout
        print(output)

        # Extract scores from output
        # GenEval outputs in format like:
        # single_object: 0.95
        # two_object: 0.78
        # ...
        scores = {}
        for line in output.split('\n'):
            if ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    task = parts[0].strip()
                    try:
                        score = float(parts[1].strip())
                        scores[task] = score
                    except ValueError:
                        pass

        return scores

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Summary computation failed")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise


def parse_results_manually(results_file: str) -> Dict[str, Any]:
    """
    Manually parse GenEval results file if summary script fails.

    Args:
        results_file: Path to results JSONL file

    Returns:
        Dictionary with task-wise scores
    """
    task_results = {}

    with open(results_file, 'r') as f:
        for line in f:
            entry = json.loads(line)

            task = entry.get('evaluation_type', 'unknown')
            correct = entry.get('correct', False)

            if task not in task_results:
                task_results[task] = {'correct': 0, 'total': 0}

            task_results[task]['total'] += 1
            if correct:
                task_results[task]['correct'] += 1

    # Compute scores
    scores = {}
    for task, stats in task_results.items():
        if stats['total'] > 0:
            scores[task] = stats['correct'] / stats['total']
        else:
            scores[task] = 0.0

    # Overall score
    if scores:
        scores['overall'] = sum(scores.values()) / len(scores)

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated images using GenEval framework"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing generated images"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        help="Text file with prompts (one per line)"
    )
    parser.add_argument(
        "--geneval_path",
        type=str,
        default="./geneval",
        help="Path to GenEval repository (default: ./geneval)"
    )
    parser.add_argument(
        "--detector_path",
        type=str,
        default=None,
        help="Path to object detector models (default: <geneval_path>/models)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./geneval_results",
        help="Directory to save evaluation results (default: ./geneval_results)"
    )
    parser.add_argument(
        "--image_pattern",
        type=str,
        default="*.png",
        help="Glob pattern for images (default: *.png)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSON file for summary scores (default: <output_dir>/summary_scores.json)"
    )

    args = parser.parse_args()

    # Set default output file
    if args.output_file is None:
        args.output_file = os.path.join(args.output_dir, "summary_scores.json")

    print(f"\n{'='*60}")
    print("GenEval Evaluation")
    print(f"{'='*60}")
    print(f"Image directory: {args.image_dir}")
    print(f"Prompt file: {args.prompt_file}")
    print(f"GenEval path: {args.geneval_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")

    # Step 1: Load prompts
    prompts = load_prompts(args.prompt_file)

    # Step 2: Create metadata file
    os.makedirs(args.output_dir, exist_ok=True)
    metadata_file = os.path.join(args.output_dir, "metadata.jsonl")
    num_entries = create_geneval_metadata(
        args.image_dir,
        prompts,
        metadata_file,
        args.image_pattern,
    )

    if num_entries == 0:
        print("ERROR: No images found! Check image_dir and image_pattern.")
        return

    # Step 3: Run GenEval evaluation
    try:
        results_file = run_geneval_evaluation(
            args.geneval_path,
            metadata_file,
            args.output_dir,
            args.detector_path,
        )
    except Exception as e:
        print(f"\nERROR: GenEval evaluation failed: {e}")
        print("\nMake sure GenEval is properly installed:")
        print(f"  1. Clone: git clone https://github.com/djghosh13/geneval.git {args.geneval_path}")
        print(f"  2. Install dependencies (see GenEval README)")
        print(f"  3. Download models: cd {args.geneval_path} && ./evaluation/download_models.sh models/")
        return

    # Step 4: Compute summary scores
    try:
        scores = compute_summary_scores(args.geneval_path, results_file)
    except Exception as e:
        print(f"\nWARNING: Summary computation failed: {e}")
        print("Attempting manual parsing...")
        scores = parse_results_manually(results_file)

    # Save summary scores
    with open(args.output_file, 'w') as f:
        json.dump(scores, f, indent=2)

    print(f"\n{'='*60}")
    print("GenEval Summary Scores")
    print(f"{'='*60}")
    for task, score in sorted(scores.items()):
        print(f"{task:20s}: {score:.4f}")
    print(f"{'='*60}\n")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
