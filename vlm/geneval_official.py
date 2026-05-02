"""
Official GenEval Implementation (Simplified)
공식 GenEval 구현을 기반으로 한 간소화 버전

이미지 디렉토리와 프롬프트 파일만으로 GenEval score 계산
프롬프트는 일반 텍스트 또는 GenEval 메타데이터 형식 지원

Reference: https://github.com/djghosh13/geneval
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import re


class GenEvalEvaluator:
    """공식 GenEval 기반 평가기"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")
        self._load_models()

        # COCO 클래스 (GenEval이 사용하는 객체 클래스)
        self.coco_classes = [
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

        # Berlin-Kay 기본 색상
        self.colors = ['red', 'orange', 'yellow', 'green', 'blue',
                      'purple', 'pink', 'brown', 'black', 'white']

    def _load_models(self):
        """모델 로드"""
        print("Loading models...")

        # CLIP 모델 (색상 분류용)
        try:
            from transformers import CLIPProcessor, CLIPModel
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_model.eval()
            print("✓ CLIP model loaded")
        except Exception as e:
            print(f"Error loading CLIP: {e}")
            raise

        # OWL-ViT 객체 탐지 모델
        try:
            from transformers import OwlViTProcessor, OwlViTForObjectDetection
            self.detector_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            self.detector_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(self.device)
            self.detector_model.eval()
            print("✓ OWL-ViT detector loaded")
        except Exception as e:
            print(f"Error loading detector: {e}")
            raise

    def detect_objects(self, image_path: str, query_classes: List[str], threshold: float = 0.1) -> Dict:
        """
        이미지에서 객체 탐지
        Returns: {class_name: [bbox1, bbox2, ...]}
        """
        try:
            image = Image.open(image_path).convert('RGB')

            inputs = self.detector_processor(
                text=[[f"a photo of a {cls}" for cls in query_classes]],
                images=image,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.detector_model(**inputs)

            # 후처리
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results = self.detector_processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=threshold
            )[0]

            # 클래스별로 그룹화
            detections = {cls: [] for cls in query_classes}
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                class_name = query_classes[label.item()]
                detections[class_name].append({
                    "box": box.cpu().tolist(),
                    "score": score.item()
                })

            return detections

        except Exception as e:
            print(f"Error in detection: {e}")
            return {cls: [] for cls in query_classes}

    def classify_color(self, image_path: str, bbox: List[float], class_name: str) -> str:
        """
        바운딩 박스 내 객체의 색상 분류
        """
        try:
            image = Image.open(image_path).convert('RGB')

            # 박스 영역 크롭
            x1, y1, x2, y2 = [int(c) for c in bbox]
            cropped = image.crop((x1, y1, x2, y2))

            # CLIP으로 색상 분류
            texts = [f"a photo of a {color} {class_name}" for color in self.colors]
            inputs = self.clip_processor(
                text=texts,
                images=cropped,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)[0]

            return self.colors[probs.argmax().item()]

        except Exception as e:
            print(f"Error in color classification: {e}")
            return "unknown"

    def get_relative_position(self, box1: List[float], box2: List[float]) -> Optional[str]:
        """
        두 박스의 상대적 위치 관계 계산
        Returns: 'left of', 'right of', 'above', 'below' or None
        """
        # 중심점 계산
        center1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
        center2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]

        dx = center1[0] - center2[0]
        dy = center1[1] - center2[1]

        # 임계값 (박스 크기의 10%)
        threshold = 0.1 * max(abs(box1[2] - box1[0]), abs(box1[3] - box1[1]))

        if abs(dx) > abs(dy) and abs(dx) > threshold:
            return 'right of' if dx > 0 else 'left of'
        elif abs(dy) > threshold:
            return 'below' if dy > 0 else 'above'

        return None

    def parse_prompt(self, prompt: str) -> Dict:
        """
        일반 프롬프트 텍스트를 파싱하여 GenEval 메타데이터 형식으로 변환
        Open-vocabulary: 모든 명사 추출 (COCO 클래스 제한 없음)
        """
        prompt_lower = prompt.lower()
        metadata = {
            "tag": "general",
            "prompt": prompt,
            "include": [],
            "exclude": []
        }

        # 개수 패턴 (예: "three cats", "2 dogs")
        number_words = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'a': 1, 'an': 1
        }

        # 일반적인 명사 추출 (Open Vocabulary)
        # 1. 먼저 COCO 클래스 확인
        found_objects = []
        for cls in self.coco_classes:
            if cls in prompt_lower:
                # 개수 찾기
                count_match = re.search(rf'(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|an)\s+{cls}', prompt_lower)
                count = 1
                if count_match:
                    count_str = count_match.group(1)
                    count = number_words.get(count_str, int(count_str) if count_str.isdigit() else 1)

                # 색상 찾기
                color = None
                for col in self.colors:
                    if f"{col} {cls}" in prompt_lower:
                        color = col
                        break

                obj_dict = {"class": cls, "count": count}
                if color:
                    obj_dict["color"] = color

                found_objects.append(obj_dict)

        # 2. COCO 클래스가 없으면 일반 명사 추출
        if not found_objects:
            # 일반적인 명사 후보 (관사/숫자 다음에 오는 단어)
            # 패턴: "a/an/the/숫자 + 명사"
            noun_pattern = r'\b(?:a|an|the|\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+([a-z]+(?:\s+[a-z]+)?)\b'
            noun_matches = re.findall(noun_pattern, prompt_lower)

            # 불용어 제거
            stopwords = {
                'photo', 'image', 'picture', 'drawing', 'painting', 'illustration',
                'digital', 'art', 'style', 'concept', 'highly', 'detailed', 'intricate',
                'realistic', 'photorealistic', 'trending', 'artstation', 'focus', 'sharp',
                'diffuse', 'lighting', 'fantasy', 'elegant', 'lifelike', 'smooth',
                'lot', 'group', 'bunch', 'set', 'collection', 'some', 'many', 'few',
                'quality', 'resolution', 'render', 'shot', 'view', 'scene'
            }

            # 명사 후보에서 객체 추출
            for noun in noun_matches:
                noun = noun.strip()
                # 불용어가 아닌 경우만 추가
                if noun not in stopwords and len(noun) > 2:
                    # 개수 찾기
                    count = 1
                    count_match = re.search(rf'(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|an)\s+{noun}', prompt_lower)
                    if count_match:
                        count_str = count_match.group(1)
                        count = number_words.get(count_str, int(count_str) if count_str.isdigit() else 1)

                    # 색상 찾기
                    color = None
                    for col in self.colors:
                        if f"{col} {noun}" in prompt_lower:
                            color = col
                            break

                    obj_dict = {"class": noun, "count": count}
                    if color:
                        obj_dict["color"] = color

                    # 중복 제거
                    if obj_dict not in found_objects:
                        found_objects.append(obj_dict)

        metadata["include"] = found_objects

        # 태그 결정
        if len(metadata["include"]) == 1:
            if metadata["include"][0]["count"] == 1:
                metadata["tag"] = "single_object"
            else:
                metadata["tag"] = "counting"
        elif len(metadata["include"]) == 2:
            metadata["tag"] = "two_object"
        elif len(metadata["include"]) > 2:
            metadata["tag"] = "multiple_objects"

        if any("color" in obj for obj in metadata["include"]):
            metadata["tag"] = "colors"

        # 위치 관계 확인
        position_keywords = ['left of', 'right of', 'above', 'below', 'on', 'under']
        if any(kw in prompt_lower for kw in position_keywords):
            metadata["tag"] = "position"

        return metadata

    def calculate_clip_score(self, image_path: str, prompt: str) -> float:
        """CLIP score 계산"""
        try:
            image = Image.open(image_path).convert('RGB')

            inputs = self.clip_processor(
                text=[prompt],
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.device)

            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                # CLIP logit 값 그대로 사용 (일반적으로 15~35 범위)
                clip_score = logits_per_image.squeeze().item()

            return clip_score

        except Exception as e:
            print(f"Error calculating CLIP score: {e}")
            return 0.0

    def evaluate_image(self, image_path: str, metadata: Dict) -> Dict:
        """
        단일 이미지를 메타데이터에 따라 평가
        """
        result = {
            "image_path": image_path,
            "prompt": metadata["prompt"],
            "tag": metadata["tag"],
            "correct": True,
            "reason": "",
            "details": {},
            "clip_score": 0.0
        }

        # CLIP Score 계산 (항상 수행)
        result["clip_score"] = self.calculate_clip_score(image_path, metadata["prompt"])

        # 필요한 객체 클래스 수집
        query_classes = list(set([obj["class"] for obj in metadata["include"]]))
        if not query_classes:
            # 객체 탐지 없이 CLIP Score만으로 평가
            result["reason"] = "No specific objects - evaluated by CLIP score only"
            # CLIP score가 일정 이상이면 correct로 간주
            result["correct"] = result["clip_score"] > 20.0  # 임계값
            result["details"]["evaluation_method"] = "clip_only"
            return result

        # 객체 탐지
        threshold = 0.1 if metadata["tag"] == "general" else 0.3
        if metadata["tag"] == "counting":
            threshold = 0.5

        detections = self.detect_objects(image_path, query_classes, threshold=threshold)
        result["details"]["detections"] = detections
        result["details"]["evaluation_method"] = "object_detection"

        # 각 요구사항 검사
        for requirement in metadata["include"]:
            cls = requirement["class"]
            required_count = requirement.get("count", 1)
            detected_count = len(detections.get(cls, []))

            # 개수 확인
            if detected_count < required_count:
                result["correct"] = False
                result["reason"] = f"Expected {required_count} {cls}, found {detected_count}"
                continue

            # 색상 확인
            if "color" in requirement and detections.get(cls):
                expected_color = requirement["color"]
                for detection in detections[cls][:required_count]:
                    detected_color = self.classify_color(
                        image_path,
                        detection["box"],
                        cls
                    )
                    if detected_color != expected_color:
                        result["correct"] = False
                        result["reason"] = f"Expected {expected_color} {cls}, detected {detected_color}"
                        break

        # 제외 조건 확인
        for exclusion in metadata.get("exclude", []):
            cls = exclusion["class"]
            max_count = exclusion.get("count", 1) - 1
            detected_count = len(detections.get(cls, []))

            if detected_count > max_count:
                result["correct"] = False
                result["reason"] = f"Should not have more than {max_count} {cls}, found {detected_count}"

        return result

    def evaluate(self, img_dir: str, prompt_file: str, output_file: str = None) -> Dict:
        """
        전체 평가 수행
        """
        print(f"Loading prompts from {prompt_file}...")
        prompts = self._load_prompts(prompt_file)

        print(f"Loading images from {img_dir}...")
        image_paths = self._load_images(img_dir)

        if len(prompts) != len(image_paths):
            print(f"Warning: {len(prompts)} prompts != {len(image_paths)} images")
            min_len = min(len(prompts), len(image_paths))
            prompts = prompts[:min_len]
            image_paths = image_paths[:min_len]

        print(f"Evaluating {len(image_paths)} images...")

        results = []
        for img_path, prompt in tqdm(zip(image_paths, prompts), total=len(image_paths)):
            # 프롬프트가 문자열이면 파싱, 딕셔너리면 메타데이터로 사용
            if isinstance(prompt, str):
                metadata = self.parse_prompt(prompt)
            else:
                metadata = prompt

            result = self.evaluate_image(img_path, metadata)
            results.append(result)

        # 점수 계산
        summary = self._calculate_scores(results)

        # 결과 출력
        self._print_results(summary)

        # 결과 저장
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "summary": summary,
                    "results": results
                }, f, indent=2, ensure_ascii=False)

            print(f"\nResults saved to {output_file}")

        return summary

    def _load_prompts(self, prompt_file: str) -> List:
        """프롬프트 파일 로드"""
        if prompt_file.endswith('.jsonl'):
            # GenEval 메타데이터 형식
            prompts = []
            with open(prompt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    prompts.append(json.loads(line.strip()))
            return prompts
        elif prompt_file.endswith('.json'):
            with open(prompt_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else list(data.values())
        else:
            # 일반 텍스트 파일
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]

    def _load_images(self, img_dir: str) -> List[str]:
        """이미지 파일 로드"""
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        img_paths = []

        img_dir_path = Path(img_dir)
        for ext in img_extensions:
            img_paths.extend(sorted(img_dir_path.glob(f'*{ext}')))
            img_paths.extend(sorted(img_dir_path.glob(f'*{ext.upper()}')))

        return sorted(list(set(str(p) for p in img_paths)))

    def _calculate_scores(self, results: List[Dict]) -> Dict:
        """GenEval 점수 계산"""
        total = len(results)
        correct = sum(1 for r in results if r["correct"])

        # CLIP scores 수집
        clip_scores = [r.get("clip_score", 0.0) for r in results]
        avg_clip_score = np.mean(clip_scores) if clip_scores else 0.0

        # 태그별 점수
        task_scores = {}
        for result in results:
            tag = result["tag"]
            if tag not in task_scores:
                task_scores[tag] = {"correct": 0, "total": 0}
            task_scores[tag]["total"] += 1
            if result["correct"]:
                task_scores[tag]["correct"] += 1

        # 각 태스크의 정확도
        task_accuracies = {}
        for tag, counts in task_scores.items():
            task_accuracies[tag] = counts["correct"] / counts["total"] if counts["total"] > 0 else 0

        # 전체 GenEval 점수 (태스크별 평균)
        geneval_score = np.mean(list(task_accuracies.values())) if task_accuracies else 0

        return {
            "total_images": total,
            "correct_images": correct,
            "image_accuracy": correct / total if total > 0 else 0,
            "avg_clip_score": avg_clip_score,
            "task_scores": task_accuracies,
            "geneval_score": geneval_score
        }

    def _print_results(self, summary: Dict):
        """결과 출력"""
        print("\n" + "="*70)
        print("GenEval Evaluation Results (Open Vocabulary)")
        print("="*70)
        print(f"Total Images: {summary['total_images']}")
        print(f"Correct Images: {summary['correct_images']}")
        print(f"Image Accuracy: {summary['image_accuracy']:.4f}")
        print(f"\nAverage CLIP Score: {summary['avg_clip_score']:.4f}")
        print(f"GenEval Score: {summary['geneval_score']:.4f}")
        print("\nTask-wise Scores:")
        for task, score in sorted(summary['task_scores'].items()):
            print(f"  {task:20s}: {score:.4f}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='GenEval Official Evaluation')
    parser.add_argument('--img_dir', type=str, required=True,
                       help='Directory containing generated images')
    parser.add_argument('--prompt_file', type=str, required=True,
                       help='Prompt file (txt, json, or jsonl with metadata)')
    parser.add_argument('--output', type=str, default='geneval_official_results.json',
                       help='Output JSON file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')

    args = parser.parse_args()

    evaluator = GenEvalEvaluator(device=args.device)
    evaluator.evaluate(
        img_dir=args.img_dir,
        prompt_file=args.prompt_file,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
