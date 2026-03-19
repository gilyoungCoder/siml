"""
GenEval with Original Mask2Former (Official Implementation)
공식 GenEval evaluate_images.py를 그대로 사용하되, 우리 데이터 형식에 맞게 wrapper 제공

이 스크립트는:
1. 우리의 img_dir + prompt_file 형식을 GenEval 형식으로 변환
2. 공식 evaluate_images.py 실행
3. 결과를 summary_scores.py로 집계

사용법:
python geneval_mask2former.py \
    --img_dir /path/to/images \
    --prompt_file /path/to/prompts.txt \
    --output results.json
"""

import os
import sys
import json
import argparse
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict
import tempfile

# GenEval official repo path
GENEVAL_REPO = "/mnt/home/yhgil99/unlearning/vlm/geneval_official_repo"
MODEL_PATH = "/mnt/home/yhgil99/unlearning/vlm/geneval_models"


def load_prompts(prompt_file: str) -> List[Dict]:
    """Load prompts from file"""
    prompts = []

    with open(prompt_file, 'r', encoding='utf-8') as f:
        if prompt_file.endswith('.jsonl'):
            # GenEval metadata format
            for line in f:
                if line.strip():
                    prompts.append(json.loads(line))
        elif prompt_file.endswith('.json'):
            # JSON array
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        prompts.append({'prompt': item})
                    else:
                        prompts.append(item)
            else:
                prompts = [data]
        else:
            # Plain text
            for line in f:
                if line.strip():
                    prompts.append({'prompt': line.strip()})

    return prompts


def parse_prompt_to_metadata(prompt: str, idx: int) -> Dict:
    """
    Parse plain text prompt to GenEval metadata format
    Simple COCO-class based parsing
    """
    import re

    prompt_lower = prompt.lower()

    metadata = {
        'prompt': prompt,
        'tag': 'unknown',
        'include': [],
        'exclude': []
    }

    # COCO classes
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
        'tv', 'laptop', 'mouse', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
        'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush', 'bench', 'fire hydrant', 'stop sign', 'parking meter'
    ]

    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'black', 'white']

    number_words = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'a': 1, 'an': 1
    }

    # Find COCO objects in prompt
    found_objects = []
    for cls in coco_classes:
        if cls in prompt_lower or cls + 's' in prompt_lower:
            # Find count
            count = 1
            count_match = re.search(rf'(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|an)\s+{cls}', prompt_lower)
            if count_match:
                count_str = count_match.group(1)
                count = number_words.get(count_str, int(count_str) if count_str.isdigit() else 1)

            # Find color
            color = None
            for col in colors:
                if f"{col} {cls}" in prompt_lower:
                    color = col
                    break

            obj_dict = {"class": cls, "count": count}
            if color:
                obj_dict["color"] = color

            found_objects.append(obj_dict)

    if not found_objects:
        # No COCO objects found
        return metadata

    metadata['include'] = found_objects

    # Determine tag
    if len(found_objects) == 1:
        if found_objects[0]['count'] > 1:
            metadata['tag'] = 'counting'
            # Add exclude for counting (n+1)
            metadata['exclude'].append({
                'class': found_objects[0]['class'],
                'count': found_objects[0]['count'] + 1
            })
        elif 'color' in found_objects[0]:
            metadata['tag'] = 'colors'
        else:
            metadata['tag'] = 'single_object'
    elif len(found_objects) == 2:
        if all('color' in obj for obj in found_objects):
            metadata['tag'] = 'color_attr'
        else:
            metadata['tag'] = 'two_object'

    return metadata


def create_geneval_structure(img_dir: str, prompts: List[Dict], temp_dir: str):
    """
    Create GenEval expected directory structure:
    temp_dir/
        00000/
            metadata.jsonl
            samples/
                0000.png
        00001/
            ...
    """
    # Get image files
    image_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if len(image_files) != len(prompts):
        print(f"Warning: {len(image_files)} images but {len(prompts)} prompts")
        min_len = min(len(image_files), len(prompts))
        image_files = image_files[:min_len]
        prompts = prompts[:min_len]

    # Create structure
    for idx, (img_file, prompt_data) in enumerate(zip(image_files, prompts)):
        folder_name = f"{idx:05d}"
        folder_path = os.path.join(temp_dir, folder_name)
        samples_path = os.path.join(folder_path, "samples")
        os.makedirs(samples_path, exist_ok=True)

        # Parse metadata if needed
        if 'include' not in prompt_data:
            metadata = parse_prompt_to_metadata(prompt_data['prompt'], idx)
        else:
            metadata = prompt_data

        # Write metadata.jsonl
        with open(os.path.join(folder_path, "metadata.jsonl"), 'w') as f:
            json.dump(metadata, f)

        # Copy image
        src_img = os.path.join(img_dir, img_file)
        dst_img = os.path.join(samples_path, "0000.png")
        shutil.copy(src_img, dst_img)

    print(f"Created GenEval structure with {len(prompts)} prompts in {temp_dir}")


def run_geneval_evaluation(temp_dir: str, output_file: str, model_path: str, device: str = "cuda"):
    """Run original GenEval evaluate_images.py"""
    eval_script = os.path.join(GENEVAL_REPO, "evaluation", "evaluate_images.py")
    results_file = os.path.join(temp_dir, "results.jsonl")

    # Check if script exists
    if not os.path.exists(eval_script):
        raise FileNotFoundError(f"GenEval evaluation script not found: {eval_script}")

    # Run evaluation
    cmd = [
        sys.executable,
        eval_script,
        temp_dir,
        "--outfile", results_file,
        "--model-path", model_path
    ]

    print(f"Running GenEval evaluation...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"GenEval evaluation failed with exit code {result.returncode}")

    print("GenEval evaluation completed!")
    return results_file


def compute_summary_scores(results_file: str) -> Dict:
    """Compute summary scores from results.jsonl (replicate summary_scores.py logic)"""
    with open(results_file, 'r') as f:
        results = [json.loads(line) for line in f if line.strip()]

    # Group by tag
    task_results = {}
    for result in results:
        tag = result['tag']
        if tag not in task_results:
            task_results[tag] = {'total': 0, 'correct': 0}
        task_results[tag]['total'] += 1
        if result['correct']:
            task_results[tag]['correct'] += 1

    # Calculate task scores
    task_scores = {
        tag: data['correct'] / data['total'] if data['total'] > 0 else 0
        for tag, data in task_results.items()
    }

    # GenEval score = average of all task scores
    geneval_score = sum(task_scores.values()) / len(task_scores) if task_scores else 0

    # Image accuracy
    total_images = len(results)
    correct_images = sum(1 for r in results if r['correct'])
    image_accuracy = correct_images / total_images if total_images > 0 else 0

    summary = {
        'total_images': total_images,
        'correct_images': correct_images,
        'image_accuracy': image_accuracy,
        'geneval_score': geneval_score,
        'task_scores': task_scores,
        'task_details': task_results
    }

    return summary, results


def print_summary(summary: Dict):
    """Print summary results"""
    print("\n" + "="*70)
    print("GenEval Evaluation Results (Original Mask2Former)")
    print("="*70)
    print(f"Total Images: {summary['total_images']}")
    print(f"Correct Images: {summary['correct_images']}")
    print(f"Image Accuracy: {summary['image_accuracy']:.4f}")
    print(f"\nGenEval Score: {summary['geneval_score']:.4f}")
    print(f"\nTask-wise Scores:")
    for tag in sorted(summary['task_scores'].keys()):
        score = summary['task_scores'][tag]
        details = summary['task_details'][tag]
        print(f"  {tag:20s}: {score:.4f} ({details['correct']}/{details['total']})")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='GenEval with Original Mask2Former')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--prompt_file', type=str, required=True, help='Prompt file')
    parser.add_argument('--output', type=str, default='geneval_mask2former_results.json', help='Output file')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Path to Mask2Former weights')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--keep_temp', action='store_true', help='Keep temporary directory')

    args = parser.parse_args()

    # Check model exists
    model_file = os.path.join(args.model_path, "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Mask2Former model not found: {model_file}")

    # Load prompts
    print(f"Loading prompts from {args.prompt_file}...")
    prompts = load_prompts(args.prompt_file)
    print(f"Loaded {len(prompts)} prompts")

    # Create temporary directory with GenEval structure
    temp_dir = tempfile.mkdtemp(prefix="geneval_")
    print(f"Creating temporary directory: {temp_dir}")

    try:
        # Create GenEval structure
        create_geneval_structure(args.img_dir, prompts, temp_dir)

        # Run GenEval evaluation
        results_file = run_geneval_evaluation(temp_dir, args.output, args.model_path, args.device)

        # Compute summary
        summary, results = compute_summary_scores(results_file)

        # Print results
        print_summary(summary)

        # Save output
        output_data = {
            'summary': summary,
            'results': results
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to {args.output}")

    finally:
        # Clean up
        if not args.keep_temp:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
        else:
            print(f"Temporary directory kept: {temp_dir}")


if __name__ == '__main__':
    main()
