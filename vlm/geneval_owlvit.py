"""
GenEval Implementation with OWL-ViT
오리지널 GenEval 평가 로직 + OWL-ViT 객체 탐지

Original GenEval: https://github.com/djghosh13/geneval
Differences from original:
- Object Detection: Mask2Former (original) → OWL-ViT (this)
- Color Classification: Uses original CLIP zero-shot approach
- Evaluation Logic: Same as original (counting, color, position, etc.)
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import re
from transformers import (
    OwlViTProcessor,
    OwlViTForObjectDetection,
    CLIPProcessor,
    CLIPModel
)


# COCO 80 classes (same as original GenEval)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Berlin-Kay basic colors (same as original)
COLORS = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'black', 'white']


class GenEvalOWLViT:
    """GenEval evaluator using OWL-ViT instead of Mask2Former"""

    def __init__(
        self,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        threshold=0.3,
        counting_threshold=0.9,
        max_objects=16,
        nms_threshold=1.0,
        position_threshold=0.1
    ):
        self.device = device
        self.threshold = threshold
        self.counting_threshold = counting_threshold
        self.max_objects = max_objects
        self.nms_threshold = nms_threshold
        self.position_threshold = position_threshold

        print(f"Loading models on {self.device}...")

        # OWL-ViT for object detection
        self.owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(self.device)

        # CLIP for color classification (same as original)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)

        self.color_classifiers = {}

        print("Models loaded successfully!")

    def detect_objects(self, image: Image.Image, metadata: Dict) -> Dict[str, List[Tuple[np.ndarray, float]]]:
        """
        Detect objects using OWL-ViT
        Returns: Dict[classname, List[(bbox, score)]]
        """
        # Use COCO classes as candidates
        text_queries = COCO_CLASSES

        inputs = self.owl_processor(
            text=text_queries,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.owl_model(**inputs)

        # Post-process
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.owl_processor.post_process_object_detection(
            outputs=outputs,
            threshold=0.0,  # We'll apply threshold later
            target_sizes=target_sizes
        )[0]

        detected = {}

        # Organize by class (same structure as original)
        for classname in COCO_CLASSES:
            class_idx = COCO_CLASSES.index(classname)

            # Get all detections for this class
            class_boxes = []
            class_scores = []

            for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
                if label.item() == class_idx:
                    class_boxes.append(box.cpu().numpy())
                    class_scores.append(score.item())

            if not class_boxes:
                continue

            # Sort by confidence (descending)
            boxes = np.array(class_boxes)
            scores = np.array(class_scores)
            ordering = np.argsort(scores)[::-1]

            # Apply threshold (different for counting vs others, same as original)
            confidence_threshold = self.counting_threshold if metadata.get('tag') == 'counting' else self.threshold
            ordering = ordering[scores[ordering] > confidence_threshold]

            # Limit max objects
            ordering = ordering[:self.max_objects]

            # Apply NMS (same as original)
            kept_objects = []
            ordering = ordering.tolist()

            while ordering:
                max_idx = ordering.pop(0)
                kept_objects.append((boxes[max_idx], scores[max_idx]))

                # Remove overlapping boxes
                if self.nms_threshold < 1.0:
                    ordering = [
                        idx for idx in ordering
                        if self._compute_iou(boxes[max_idx], boxes[idx]) < self.nms_threshold
                    ]

            if kept_objects:
                detected[classname] = kept_objects

        return detected

    def _compute_iou(self, box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Compute IoU between two boxes (same as original)"""
        area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)

        i_area = area_fn([
            max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
            min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
        ])

        u_area = area_fn(box_a) + area_fn(box_b) - i_area
        return i_area / u_area if u_area > 0 else 0

    def color_classification(
        self,
        image: Image.Image,
        bboxes: List[Tuple[np.ndarray, float]],
        classname: str
    ) -> List[str]:
        """
        Classify colors of detected objects (same approach as original)
        """
        # Create zero-shot classifier for this class if not cached
        if classname not in self.color_classifiers:
            # Same templates as original GenEval
            templates = [
                f"a photo of a {{c}} {classname}",
                f"a photo of a {{c}}-colored {classname}",
                f"a photo of a {{c}} object"
            ]

            # Create text inputs for all colors
            all_texts = []
            for color in COLORS:
                for template in templates:
                    all_texts.append(template.format(c=color))

            self.color_classifiers[classname] = all_texts

        # Crop and classify each object
        predicted_colors = []

        for bbox, score in bboxes:
            # Crop image to bbox
            x1, y1, x2, y2 = bbox[:4].astype(int)
            cropped = image.crop((x1, y1, x2, y2))

            # CLIP classification
            inputs = self.clip_processor(
                text=self.color_classifiers[classname],
                images=cropped,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

            # Get most likely color (average across templates)
            probs_reshaped = probs.view(len(COLORS), len(self.color_classifiers[classname]) // len(COLORS))
            color_probs = probs_reshaped.mean(dim=1)
            color_idx = color_probs.argmax().item()
            predicted_colors.append(COLORS[color_idx])

        return predicted_colors

    def relative_position(
        self,
        obj_a: Tuple[np.ndarray, float],
        obj_b: Tuple[np.ndarray, float]
    ) -> Set[str]:
        """
        Compute relative position of A relative to B (same as original)
        """
        boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
        center_a, center_b = boxes.mean(axis=-2)
        dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
        offset = center_a - center_b

        # Apply position threshold (same as original)
        revised_offset = np.maximum(
            np.abs(offset) - self.position_threshold * (dim_a + dim_b),
            0
        ) * np.sign(offset)

        if np.all(np.abs(revised_offset) < 1e-3):
            return set()

        # Determine direction
        dx, dy = revised_offset / np.linalg.norm(offset)
        relations = set()

        if dx < -0.5: relations.add("left of")
        if dx > 0.5: relations.add("right of")
        if dy < -0.5: relations.add("above")
        if dy > 0.5: relations.add("below")

        return relations

    def evaluate(
        self,
        image: Image.Image,
        objects: Dict[str, List[Tuple[np.ndarray, float]]],
        metadata: Dict
    ) -> Tuple[bool, str]:
        """
        Evaluate image against metadata (same logic as original)
        """
        correct = True
        reason = []
        matched_groups = []

        # Check for expected objects (include clauses)
        for req in metadata.get('include', []):
            classname = req['class']
            matched = True
            found_objects = objects.get(classname, [])[:req['count']]

            # Check count
            if len(found_objects) < req['count']:
                correct = matched = False
                reason.append(f"expected {classname}>={req['count']}, found {len(found_objects)}")
            else:
                # Check color if specified
                if 'color' in req:
                    colors = self.color_classification(image, found_objects, classname)
                    if colors.count(req['color']) < req['count']:
                        correct = matched = False
                        color_counts = ", ".join(f"{colors.count(c)} {c}" for c in COLORS if c in colors)
                        reason.append(
                            f"expected {req['color']} {classname}>={req['count']}, " +
                            f"found {colors.count(req['color'])} {req['color']}; and {color_counts}"
                        )

                # Check position if specified
                if 'position' in req and matched:
                    expected_rel, target_group = req['position']
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(f"no target for {classname} to be {expected_rel}")
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = self.relative_position(obj, target_obj)
                                if expected_rel not in true_rels:
                                    correct = matched = False
                                    rel_str = ' and '.join(true_rels) if true_rels else 'no relation'
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, " +
                                        f"found {rel_str} target"
                                    )
                                    break
                            if not matched:
                                break

            # Track matched groups for position evaluation
            if matched:
                matched_groups.append(found_objects)
            else:
                matched_groups.append(None)

        # Check for excluded objects (exclude clauses)
        for req in metadata.get('exclude', []):
            classname = req['class']
            if len(objects.get(classname, [])) >= req['count']:
                correct = False
                reason.append(f"expected {classname}<{req['count']}, found {len(objects[classname])}")

        return correct, "\n".join(reason)

    def evaluate_image(self, image_path: str, metadata: Dict) -> Dict:
        """Evaluate a single image"""
        image = Image.open(image_path).convert("RGB")

        # Detect objects
        detected = self.detect_objects(image, metadata)

        # Evaluate
        is_correct, reason = self.evaluate(image, detected, metadata)

        return {
            'image_path': image_path,
            'prompt': metadata['prompt'],
            'tag': metadata['tag'],
            'correct': is_correct,
            'reason': reason,
            'metadata': metadata,
            'details': {
                'detections': {
                    cls: [(box.tolist(), score) for box, score in objs]
                    for cls, objs in detected.items()
                }
            }
        }

    def parse_prompt_to_metadata(self, prompt: str) -> Dict:
        """
        Parse plain text prompt to GenEval metadata format
        Simple heuristic-based parsing
        """
        prompt = prompt.lower().strip()

        metadata = {
            'prompt': prompt,
            'tag': 'unknown',
            'include': [],
            'exclude': []
        }

        # Try to extract objects and attributes
        # Pattern: "a photo of [number] [color] [object](s) [position] [color] [object]"

        # Extract numbers
        number_words = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }

        found_objects = []

        # Simple pattern: find COCO objects in prompt
        for cls in COCO_CLASSES:
            if cls in prompt or cls + 's' in prompt:
                # Check for number before object
                count = 1
                for num_word, num_val in number_words.items():
                    if num_word + ' ' + cls in prompt:
                        count = num_val
                        break

                # Check for digit before object
                digit_match = re.search(rf'(\d+)\s+{cls}', prompt)
                if digit_match:
                    count = int(digit_match.group(1))

                # Check for color
                color = None
                for c in COLORS:
                    if c + ' ' + cls in prompt:
                        color = c
                        break

                obj_info = {'class': cls, 'count': count}
                if color:
                    obj_info['color'] = color

                found_objects.append(obj_info)

        if not found_objects:
            # No COCO objects found, return empty metadata
            return metadata

        metadata['include'] = found_objects

        # Determine tag
        if len(found_objects) == 1:
            if found_objects[0]['count'] > 1:
                metadata['tag'] = 'counting'
            elif 'color' in found_objects[0]:
                metadata['tag'] = 'colors'
            else:
                metadata['tag'] = 'single_object'
        elif len(found_objects) == 2:
            if all('color' in obj for obj in found_objects):
                metadata['tag'] = 'color_attr'
            else:
                metadata['tag'] = 'two_object'

        # Add exclude for counting (n+1)
        if metadata['tag'] == 'counting':
            for obj in found_objects:
                metadata['exclude'].append({
                    'class': obj['class'],
                    'count': obj['count'] + 1
                })

        return metadata


def load_prompts(prompt_file: str) -> List[Dict]:
    """Load prompts from various formats"""
    prompts = []

    with open(prompt_file, 'r') as f:
        if prompt_file.endswith('.jsonl'):
            # GenEval metadata format
            for line in f:
                if line.strip():
                    prompts.append(json.loads(line))
        elif prompt_file.endswith('.json'):
            # JSON array
            data = json.load(f)
            if isinstance(data, list):
                prompts = data
            else:
                prompts = [data]
        else:
            # Plain text
            for line in f:
                if line.strip():
                    prompts.append({'prompt': line.strip()})

    return prompts


def main():
    parser = argparse.ArgumentParser(description='GenEval with OWL-ViT')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--prompt_file', type=str, required=True, help='Prompt file (.txt, .json, or .jsonl)')
    parser.add_argument('--output', type=str, default='geneval_owlvit_results.json', help='Output file')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--threshold', type=float, default=0.3, help='Detection threshold')
    parser.add_argument('--counting_threshold', type=float, default=0.9, help='Threshold for counting task')
    parser.add_argument('--max_objects', type=int, default=16, help='Max objects per class')
    parser.add_argument('--nms_threshold', type=float, default=1.0, help='NMS IoU threshold')
    parser.add_argument('--position_threshold', type=float, default=0.1, help='Position threshold')

    args = parser.parse_args()

    # Load evaluator
    evaluator = GenEvalOWLViT(
        device=args.device,
        threshold=args.threshold,
        counting_threshold=args.counting_threshold,
        max_objects=args.max_objects,
        nms_threshold=args.nms_threshold,
        position_threshold=args.position_threshold
    )

    # Load prompts
    print(f"Loading prompts from {args.prompt_file}...")
    prompts = load_prompts(args.prompt_file)
    print(f"Loaded {len(prompts)} prompts")

    # Get image files
    image_files = sorted([
        os.path.join(args.img_dir, f)
        for f in os.listdir(args.img_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if len(image_files) != len(prompts):
        print(f"Warning: {len(image_files)} images but {len(prompts)} prompts")
        min_len = min(len(image_files), len(prompts))
        image_files = image_files[:min_len]
        prompts = prompts[:min_len]

    # Evaluate
    results = []
    correct_count = 0
    task_results = {}

    print(f"\nEvaluating {len(image_files)} images...")
    for img_path, prompt_data in tqdm(zip(image_files, prompts), total=len(image_files)):
        # Parse metadata if needed
        if 'include' not in prompt_data:
            metadata = evaluator.parse_prompt_to_metadata(prompt_data['prompt'])
        else:
            metadata = prompt_data

        # Evaluate
        result = evaluator.evaluate_image(img_path, metadata)
        results.append(result)

        if result['correct']:
            correct_count += 1

        # Track by task
        tag = result['tag']
        if tag not in task_results:
            task_results[tag] = {'total': 0, 'correct': 0}
        task_results[tag]['total'] += 1
        if result['correct']:
            task_results[tag]['correct'] += 1

    # Calculate scores
    task_scores = {
        tag: data['correct'] / data['total'] if data['total'] > 0 else 0
        for tag, data in task_results.items()
    }

    geneval_score = sum(task_scores.values()) / len(task_scores) if task_scores else 0

    # Print results
    print("\n" + "="*70)
    print("GenEval Evaluation Results (OWL-ViT)")
    print("="*70)
    print(f"Total Images: {len(results)}")
    print(f"Correct Images: {correct_count}")
    print(f"Image Accuracy: {correct_count / len(results):.4f}")
    print(f"\nGenEval Score: {geneval_score:.4f}")
    print(f"\nTask-wise Scores:")
    for tag in sorted(task_scores.keys()):
        print(f"  {tag:20s}: {task_scores[tag]:.4f}")
    print("="*70)

    # Save results
    output_data = {
        'summary': {
            'total_images': len(results),
            'correct_images': correct_count,
            'image_accuracy': correct_count / len(results),
            'geneval_score': geneval_score,
            'task_scores': task_scores
        },
        'results': results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
