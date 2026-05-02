"""
GenEval Score Evaluation Script (Official Implementation Based)
공식 GenEval 구현을 기반으로 한 이미지 생성 품질 평가 스크립트

Reference: https://github.com/djghosh13/geneval
Paper: GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment
"""

import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class GenEvalEvaluator:
    """GenEval 점수를 계산하는 평가기 (공식 구현 기반)"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")
        self._load_models()

    def _load_models(self):
        """평가에 필요한 모델들을 로드합니다"""
        print(f"Loading models on {self.device}...")

        # CLIP 모델 로드 (색상 분류 및 유사도 측정용)
        try:
            from transformers import CLIPProcessor, CLIPModel
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_model.eval()
            print("✓ CLIP model loaded")
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            raise

        # OWL-ViT 모델 로드 (오픈 vocabulary 객체 탐지용)
        try:
            from transformers import OwlViTProcessor, OwlViTForObjectDetection
            self.owlvit_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            self.owlvit_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(self.device)
            self.owlvit_model.eval()
            print("✓ OWL-ViT object detector loaded")
        except Exception as e:
            print(f"Warning: Could not load OWL-ViT, using CLIP-based detection: {e}")
            self.owlvit_model = None

        # 기본 색상 리스트 (Berlin and Kay의 11가지 기본 색상)
        self.basic_colors = ['red', 'orange', 'yellow', 'green', 'blue',
                            'purple', 'pink', 'gray', 'brown', 'black', 'white']

    def load_prompts(self, prompt_file: str) -> List[str]:
        """프롬프트 파일을 로드합니다"""
        prompts = []

        if prompt_file.endswith('.json'):
            with open(prompt_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    prompts = data
                elif isinstance(data, dict):
                    prompts = list(data.values())
        elif prompt_file.endswith('.txt'):
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
            raise ValueError(f"Unsupported prompt file format: {prompt_file}")

        return prompts

    def load_images(self, img_dir: str) -> List[str]:
        """이미지 디렉토리에서 이미지 파일들을 로드합니다"""
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        img_paths = []

        img_dir_path = Path(img_dir)
        for ext in img_extensions:
            img_paths.extend(sorted(img_dir_path.glob(f'*{ext}')))
            img_paths.extend(sorted(img_dir_path.glob(f'*{ext.upper()}')))

        # 중복 제거 및 정렬
        img_paths = sorted(list(set(img_paths)))

        return [str(p) for p in img_paths]

    def detect_objects(self, image_path: str, text_queries: List[str],
                      threshold: float = 0.1) -> List[Dict]:
        """
        OWL-ViT를 사용한 오픈 vocabulary 객체 탐지
        text_queries: 탐지할 객체 이름 리스트 (예: ["cat", "dog"])
        """
        if self.owlvit_model is None:
            return []

        try:
            image = Image.open(image_path).convert('RGB')

            inputs = self.owlvit_processor(
                text=text_queries,
                images=image,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.owlvit_model(**inputs)

            # 결과 후처리
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results = self.owlvit_processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=threshold
            )[0]

            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                detections.append({
                    "score": score.item(),
                    "label": text_queries[label.item()],
                    "box": box.cpu().tolist()
                })

            return detections

        except Exception as e:
            print(f"Error in object detection for {image_path}: {e}")
            return []

    def classify_color(self, image_path: str, object_name: str,
                      bbox: Optional[List[float]] = None) -> Tuple[str, float]:
        """
        CLIP을 사용하여 객체의 색상을 분류합니다
        bbox: [x1, y1, x2, y2] 형식의 bounding box (선택적)
        """
        try:
            image = Image.open(image_path).convert('RGB')

            # bbox가 주어진 경우 해당 영역만 크롭
            if bbox is not None:
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                image = image.crop((x1, y1, x2, y2))

            # 각 색상에 대한 텍스트 프롬프트 생성
            text_prompts = [f"a photo of a {color} {object_name}" for color in self.basic_colors]

            inputs = self.clip_processor(
                text=text_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)[0]

            # 가장 높은 확률을 가진 색상 반환
            max_idx = probs.argmax().item()
            predicted_color = self.basic_colors[max_idx]
            confidence = probs[max_idx].item()

            return predicted_color, confidence

        except Exception as e:
            print(f"Error in color classification: {e}")
            return "unknown", 0.0

    def calculate_clip_score(self, image_path: str, prompt: str) -> float:
        """CLIP 점수를 계산합니다"""
        try:
            image = Image.open(image_path).convert('RGB')

            # 프롬프트가 너무 긴 경우 잘라내기 (CLIP max_length=77)
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
                # 이미지와 텍스트 feature 간 코사인 유사도
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds

                # 정규화
                image_features = F.normalize(image_features, p=2, dim=-1)
                text_features = F.normalize(text_features, p=2, dim=-1)

                # 코사인 유사도
                clip_score = (image_features @ text_features.T).squeeze().item()

                # 0-100 스케일로 변환 (일반적인 CLIP score 표현)
                clip_score = clip_score * 100

            return clip_score

        except Exception as e:
            print(f"Error calculating CLIP score for {image_path}: {e}")
            return 0.0

    def parse_prompt_for_objects(self, prompt: str) -> Dict:
        """
        프롬프트에서 객체, 개수, 색상, 위치 정보를 추출합니다
        간단한 휴리스틱 기반 파싱 (실제로는 더 정교한 NLP 파싱 필요)
        """
        import re

        info = {
            "objects": [],
            "counts": {},
            "colors": {},
            "positions": {}
        }

        # 숫자 단어를 정수로 변환
        number_words = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'a': 1, 'an': 1
        }

        # 개수 패턴 찾기 (예: "three cats", "5 dogs")
        count_pattern = r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(\w+s?)\b'
        for match in re.finditer(count_pattern, prompt.lower()):
            count_str, obj = match.groups()
            count = number_words.get(count_str, int(count_str) if count_str.isdigit() else 1)
            info["counts"][obj] = count
            if obj not in info["objects"]:
                info["objects"].append(obj)

        # 색상 패턴 찾기 (예: "red apple", "blue car")
        for color in self.basic_colors:
            color_pattern = rf'\b{color}\s+(\w+)\b'
            for match in re.finditer(color_pattern, prompt.lower()):
                obj = match.group(1)
                info["colors"][obj] = color
                if obj not in info["objects"]:
                    info["objects"].append(obj)

        # 기본 객체 추출 (명사로 추정되는 단어들)
        # 매우 간단한 휴리스틱: 일반적인 객체 이름들
        common_objects = ['cat', 'dog', 'bird', 'car', 'tree', 'house', 'person',
                         'apple', 'banana', 'chair', 'table', 'book', 'phone']
        for obj in common_objects:
            if obj in prompt.lower() and obj not in info["objects"]:
                info["objects"].append(obj)

        return info

    def evaluate_single_image(self, image_path: str, prompt: str) -> Dict:
        """단일 이미지에 대한 GenEval 평가를 수행합니다"""

        result = {
            "image_path": image_path,
            "prompt": prompt,
            "clip_score": 0.0,
            "object_detection": {},
            "color_accuracy": {},
            "evaluation": {}
        }

        # 1. CLIP Score 계산
        result["clip_score"] = self.calculate_clip_score(image_path, prompt)

        # 2. 프롬프트에서 객체 정보 추출
        prompt_info = self.parse_prompt_for_objects(prompt)
        expected_objects = prompt_info["objects"]

        # 3. 객체 탐지 수행
        if len(expected_objects) > 0 and self.owlvit_model is not None:
            detections = self.detect_objects(image_path, expected_objects, threshold=0.1)
            result["object_detection"] = {
                "expected": expected_objects,
                "detected": [d["label"] for d in detections],
                "details": detections
            }

            # 4. 객체별 색상 평가
            for obj_name, expected_color in prompt_info["colors"].items():
                # 해당 객체의 탐지 결과 찾기
                obj_detections = [d for d in detections if d["label"] == obj_name]

                if obj_detections:
                    # 가장 높은 confidence를 가진 탐지 결과 사용
                    best_detection = max(obj_detections, key=lambda x: x["score"])
                    predicted_color, color_conf = self.classify_color(
                        image_path, obj_name, best_detection["box"]
                    )

                    result["color_accuracy"][obj_name] = {
                        "expected": expected_color,
                        "predicted": predicted_color,
                        "confidence": color_conf,
                        "correct": predicted_color == expected_color
                    }

            # 5. 개수 평가
            for obj_name, expected_count in prompt_info["counts"].items():
                detected_count = sum(1 for d in detections if d["label"] == obj_name)
                result["evaluation"][f"count_{obj_name}"] = {
                    "expected": expected_count,
                    "detected": detected_count,
                    "correct": expected_count == detected_count
                }

        return result

    def evaluate(
        self,
        img_dir: str,
        prompt_file: str,
        output_file: str = None
    ) -> Dict:
        """전체 평가를 수행합니다"""

        print(f"Loading prompts from {prompt_file}...")
        prompts = self.load_prompts(prompt_file)

        print(f"Loading images from {img_dir}...")
        image_paths = self.load_images(img_dir)

        if len(prompts) != len(image_paths):
            print(f"Warning: Number of prompts ({len(prompts)}) != Number of images ({len(image_paths)})")
            min_len = min(len(prompts), len(image_paths))
            prompts = prompts[:min_len]
            image_paths = image_paths[:min_len]

        print(f"Evaluating {len(image_paths)} image-prompt pairs...")

        results = []
        clip_scores = []
        color_accuracies = []
        count_accuracies = []

        for idx, (img_path, prompt) in enumerate(tqdm(zip(image_paths, prompts), total=len(image_paths))):
            result = self.evaluate_single_image(img_path, prompt)
            result["index"] = idx
            results.append(result)

            clip_scores.append(result["clip_score"])

            # 색상 정확도 수집
            if result["color_accuracy"]:
                color_correct = sum(1 for v in result["color_accuracy"].values() if v["correct"])
                color_total = len(result["color_accuracy"])
                if color_total > 0:
                    color_accuracies.append(color_correct / color_total)

            # 개수 정확도 수집
            count_evals = {k: v for k, v in result["evaluation"].items() if k.startswith("count_")}
            if count_evals:
                count_correct = sum(1 for v in count_evals.values() if v["correct"])
                count_total = len(count_evals)
                if count_total > 0:
                    count_accuracies.append(count_correct / count_total)

        # 평균 점수 계산
        evaluation_summary = {
            'total_samples': len(results),
            'clip_score': {
                'mean': np.mean(clip_scores) if clip_scores else 0.0,
                'std': np.std(clip_scores) if clip_scores else 0.0
            },
            'color_accuracy': {
                'mean': np.mean(color_accuracies) if color_accuracies else 0.0,
                'std': np.std(color_accuracies) if color_accuracies else 0.0,
                'num_evaluated': len(color_accuracies)
            },
            'count_accuracy': {
                'mean': np.mean(count_accuracies) if count_accuracies else 0.0,
                'std': np.std(count_accuracies) if count_accuracies else 0.0,
                'num_evaluated': len(count_accuracies)
            },
            'results': results
        }

        # GenEval 종합 점수 계산 (사용 가능한 메트릭의 평균)
        available_metrics = []
        if clip_scores:
            # CLIP score를 0-1 범위로 정규화
            available_metrics.append(np.mean(clip_scores) / 100.0)
        if color_accuracies:
            available_metrics.append(np.mean(color_accuracies))
        if count_accuracies:
            available_metrics.append(np.mean(count_accuracies))

        geneval_score = np.mean(available_metrics) if available_metrics else 0.0
        evaluation_summary['geneval_score'] = geneval_score

        # 결과 출력
        print("\n" + "="*70)
        print("GenEval Evaluation Results")
        print("="*70)
        print(f"Total Samples: {evaluation_summary['total_samples']}")
        print(f"\nGenEval Score: {geneval_score:.4f}")
        print(f"\nCLIP Score: {evaluation_summary['clip_score']['mean']:.4f} ± {evaluation_summary['clip_score']['std']:.4f}")

        if color_accuracies:
            print(f"Color Accuracy: {evaluation_summary['color_accuracy']['mean']:.4f} ± {evaluation_summary['color_accuracy']['std']:.4f} "
                  f"({evaluation_summary['color_accuracy']['num_evaluated']} samples)")
        else:
            print("Color Accuracy: N/A (no color attributes in prompts)")

        if count_accuracies:
            print(f"Count Accuracy: {evaluation_summary['count_accuracy']['mean']:.4f} ± {evaluation_summary['count_accuracy']['std']:.4f} "
                  f"({evaluation_summary['count_accuracy']['num_evaluated']} samples)")
        else:
            print("Count Accuracy: N/A (no count attributes in prompts)")

        print("="*70)

        # 결과 저장
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_summary, f, indent=2, ensure_ascii=False)

            print(f"\nResults saved to {output_file}")

        return evaluation_summary


def main():
    parser = argparse.ArgumentParser(description='GenEval Score Evaluation')
    parser.add_argument(
        '--img_dir',
        type=str,
        required=True,
        help='Path to directory containing generated images'
    )
    parser.add_argument(
        '--prompt_file',
        type=str,
        required=True,
        help='Path to file containing prompts (txt or json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='geneval_results.json',
        help='Path to output JSON file (default: geneval_results.json)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda or cpu)'
    )

    args = parser.parse_args()

    # 평가 실행
    evaluator = GenEvalEvaluator(device=args.device)
    evaluator.evaluate(
        img_dir=args.img_dir,
        prompt_file=args.prompt_file,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
