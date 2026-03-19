"""
GenEval Score Evaluation Script
이미지 디렉토리와 프롬프트 파일을 받아서 생성된 이미지의 품질을 평가합니다.
"""

import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np


class GenEvalEvaluator:
    """GenEval 점수를 계산하는 평가기"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self._load_models()

    def _load_models(self):
        """평가에 필요한 모델들을 로드합니다"""
        print(f"Loading models on {self.device}...")

        # CLIP 모델 로드
        try:
            from transformers import CLIPProcessor, CLIPModel
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            print("✓ CLIP model loaded")
        except Exception as e:
            print(f"Warning: Failed to load CLIP model: {e}")
            self.clip_model = None

        # BLIP 모델 로드 (이미지 캡셔닝용)
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(self.device)
            print("✓ BLIP model loaded")
        except Exception as e:
            print(f"Warning: Failed to load BLIP model: {e}")
            self.blip_model = None

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

    def calculate_clip_score(self, image_path: str, prompt: str) -> float:
        """CLIP 점수를 계산합니다"""
        if self.clip_model is None:
            return 0.0

        try:
            image = Image.open(image_path).convert('RGB')

            inputs = self.clip_processor(
                text=[prompt],
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                clip_score = logits_per_image.item()

            return clip_score
        except Exception as e:
            print(f"Error calculating CLIP score for {image_path}: {e}")
            return 0.0

    def generate_caption(self, image_path: str) -> str:
        """이미지에서 캡션을 생성합니다"""
        if self.blip_model is None:
            return ""

        try:
            image = Image.open(image_path).convert('RGB')

            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50)
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)

            return caption
        except Exception as e:
            print(f"Error generating caption for {image_path}: {e}")
            return ""

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """두 텍스트 간의 유사도를 계산합니다 (CLIP 사용)"""
        if self.clip_model is None:
            return 0.0

        try:
            inputs = self.clip_processor(
                text=[text1, text2],
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
                # 코사인 유사도 계산
                similarity = torch.nn.functional.cosine_similarity(
                    text_features[0].unsqueeze(0),
                    text_features[1].unsqueeze(0)
                ).item()

            return similarity
        except Exception as e:
            print(f"Error calculating text similarity: {e}")
            return 0.0

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
        text_similarities = []

        for idx, (img_path, prompt) in enumerate(tqdm(zip(image_paths, prompts), total=len(image_paths))):
            # CLIP Score 계산
            clip_score = self.calculate_clip_score(img_path, prompt)

            # 캡션 생성 및 텍스트 유사도 계산
            caption = self.generate_caption(img_path)
            text_similarity = self.calculate_text_similarity(prompt, caption)

            result = {
                'index': idx,
                'image_path': img_path,
                'prompt': prompt,
                'generated_caption': caption,
                'clip_score': clip_score,
                'text_similarity': text_similarity
            }

            results.append(result)
            clip_scores.append(clip_score)
            text_similarities.append(text_similarity)

        # 평균 점수 계산
        evaluation_summary = {
            'total_samples': len(results),
            'avg_clip_score': np.mean(clip_scores),
            'std_clip_score': np.std(clip_scores),
            'avg_text_similarity': np.mean(text_similarities),
            'std_text_similarity': np.std(text_similarities),
            'results': results
        }

        # 결과 출력
        print("\n" + "="*50)
        print("GenEval Evaluation Results")
        print("="*50)
        print(f"Total Samples: {evaluation_summary['total_samples']}")
        print(f"Average CLIP Score: {evaluation_summary['avg_clip_score']:.4f} ± {evaluation_summary['std_clip_score']:.4f}")
        print(f"Average Text Similarity: {evaluation_summary['avg_text_similarity']:.4f} ± {evaluation_summary['std_text_similarity']:.4f}")
        print("="*50)

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
