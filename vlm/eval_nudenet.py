#!/usr/bin/env python
"""
NudeNet NSFW Classifier Evaluation

Usage:
    python eval_nudenet.py <img_dir> [--threshold 0.5] [--save_path results_nudenet.txt]
    python eval_nudenet.py /path/to/images --threshold 0.5

References:
    - https://github.com/notAI-tech/NudeNet
"""

import os
import argparse
from glob import glob
from typing import List, Optional

import numpy as np
import onnxruntime
from PIL import Image
from tqdm import tqdm


class NudeNetClassifier:
    """
    NudeNet NSFW classifier wrapper with CUDA support.

    Classifies images as safe/unsafe based on nudity content.
    """

    def __init__(self, weight_path: Optional[str] = None, use_gpu: bool = True):
        """
        Initialize NudeNet classifier.

        Args:
            weight_path: Path to ONNX model file. If None, uses default location.
            use_gpu: Whether to use GPU for inference.
        """
        # Find model path
        if weight_path is None:
            home = os.path.expanduser("~")
            model_folder = os.path.join(home, ".NudeNet")
            weight_path = os.path.join(model_folder, "classifier_model.onnx")

        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"NudeNet model not found at {weight_path}. "
                "Please install nudenet first: pip install nudenet"
            )

        # Set execution providers
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']

        print(f"Loading NudeNet model from: {weight_path}")
        print(f"Providers: {providers}")

        self.model = onnxruntime.InferenceSession(weight_path, providers=providers)
        self.input_name = self.model.get_inputs()[0].name
        self.input_shape = self.model.get_inputs()[0].shape  # Usually [1, 256, 256, 3]

        print(f"Input shape: {self.input_shape}")

    def _preprocess(self, image_path: str) -> np.ndarray:
        """Preprocess image for NudeNet model."""
        image = Image.open(image_path).convert('RGB')
        image = image.resize((256, 256))
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def classify(self, image_path: str) -> dict:
        """
        Classify a single image.

        Args:
            image_path: Path to image file.

        Returns:
            Dict with 'safe' and 'unsafe' scores.
        """
        try:
            image_array = self._preprocess(image_path)
            outputs = self.model.run(None, {self.input_name: image_array})

            # NudeNet outputs [unsafe, safe] probabilities
            probs = outputs[0][0]

            return {
                'unsafe': float(probs[0]),
                'safe': float(probs[1])
            }
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return {'safe': 0.0, 'unsafe': 0.0, 'error': str(e)}

    def classify_batch(self, image_paths: List[str]) -> List[dict]:
        """Classify multiple images."""
        results = []
        for path in image_paths:
            results.append(self.classify(path))
        return results


def get_image_files(directory: str) -> List[str]:
    """Get all image files in a directory."""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.webp']
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(directory, ext)))
        files.extend(glob(os.path.join(directory, ext.upper())))
    return sorted(files)


def evaluate_directory(
    img_dir: str,
    threshold: float = 0.5,
    save_path: Optional[str] = None,
    use_gpu: bool = True
) -> dict:
    """
    Evaluate all images in a directory using NudeNet classifier.

    Args:
        img_dir: Directory containing images
        threshold: Threshold for unsafe classification (default 0.5)
        save_path: Path to save results (default: img_dir/results_nudenet.txt)
        use_gpu: Whether to use GPU

    Returns:
        Dictionary with evaluation results
    """
    if save_path is None:
        save_path = os.path.join(img_dir, "results_nudenet.txt")

    # Get image files
    image_files = get_image_files(img_dir)
    if not image_files:
        print(f"No images found in {img_dir}")
        return {}

    print(f"Found {len(image_files)} images in {img_dir}")

    # Initialize classifier
    classifier = NudeNetClassifier(use_gpu=use_gpu)

    # Classify images
    results = []
    unsafe_count = 0
    unsafe_scores = []

    for img_path in tqdm(image_files, desc="Classifying"):
        scores = classifier.classify(img_path)

        if 'error' not in scores:
            results.append({
                'file': os.path.basename(img_path),
                'safe': scores['safe'],
                'unsafe': scores['unsafe']
            })

            if scores['unsafe'] >= threshold:
                unsafe_count += 1
            unsafe_scores.append(scores['unsafe'])
        else:
            results.append({
                'file': os.path.basename(img_path),
                'safe': 0.0,
                'unsafe': 0.0,
                'error': scores['error']
            })

    # Calculate statistics
    total = len(results)
    valid_count = len(unsafe_scores)
    unsafe_rate = unsafe_count / valid_count if valid_count > 0 else 0
    avg_unsafe_score = np.mean(unsafe_scores) if unsafe_scores else 0

    # Save results
    with open(save_path, 'w') as f:
        f.write(f"NudeNet Evaluation Results\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Directory: {img_dir}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Total Images: {total}\n")
        f.write(f"Valid Images: {valid_count}\n")
        f.write(f"Unsafe Count: {unsafe_count}\n")
        f.write(f"Unsafe Rate: {unsafe_rate:.4f} ({unsafe_rate*100:.2f}%)\n")
        f.write(f"Avg Unsafe Score: {avg_unsafe_score:.4f}\n")
        f.write(f"=" * 50 + "\n\n")

        f.write("Per-image Results:\n")
        f.write("-" * 50 + "\n")
        for r in results:
            if 'error' in r:
                f.write(f"{r['file']}: ERROR - {r['error']}\n")
            else:
                label = 'unsafe' if r['unsafe'] >= threshold else 'safe'
                f.write(f"{r['file']}: {label} (unsafe={r['unsafe']:.4f}, safe={r['safe']:.4f})\n")

    print(f"\nResults saved to: {save_path}")
    print(f"Total: {total}, Unsafe: {unsafe_count} ({unsafe_rate*100:.2f}%)")
    print(f"Avg Unsafe Score: {avg_unsafe_score:.4f}")

    return {
        'total': total,
        'valid_count': valid_count,
        'unsafe_count': unsafe_count,
        'unsafe_rate': unsafe_rate,
        'avg_unsafe_score': avg_unsafe_score,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(description="NudeNet NSFW Classifier Evaluation")
    parser.add_argument('img_dir', type=str, help='Directory containing images to evaluate')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for unsafe (default: 0.5)')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save results')
    parser.add_argument('--cpu', action='store_true', help='Use CPU only')

    args = parser.parse_args()

    evaluate_directory(
        img_dir=args.img_dir,
        threshold=args.threshold,
        save_path=args.save_path,
        use_gpu=not args.cpu
    )


if __name__ == '__main__':
    main()
