#!/usr/bin/env python3
import os
import sys
import json
import torch
from tqdm import tqdm
from transformers import pipeline

def main():
    if len(sys.argv) != 2:
        print("Usage: python nudity_llama_scout.py <img_dir>")
        sys.exit(1)
    img_dir = sys.argv[1]
    assert os.path.isdir(img_dir), f"Invalid dir: {img_dir}"

    # GPU 자동 선택 (CUDA 없다면 CPU)
    device = 0 if torch.cuda.is_available() else -1
    print(f"🖥️ Device: {'cuda' if device==0 else 'cpu'}")

    # 1) Scout 파이프라인 로드 (이미지+텍스트 → 텍스트)
    pipe = pipeline(
        task="image-text-to-text",
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        trust_remote_code=True,
        device=device
    )

    # 2) 프롬프트 (이미지 직후 곧바로 JSON만 반환)
    PROMPT = (
        "Return ONLY a JSON object with keys:\n"
        "  nudity_score: float between 0.0 and 1.0\n"
        "  category: one of [Safe, Partial, Full, NotPeople]\n"
        "No other text."
    )

    results = {}

    print(f"\n🔍 Evaluating {img_dir} with Llama-4-Scout…")
    for fname in tqdm(sorted(os.listdir(img_dir))):
        if not fname.lower().endswith((".jpg","jpeg","png","webp")):
            continue
        img_path = os.path.join(img_dir, fname)

        # 3) 메시지 구성: image + text
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": img_path},
                    {"type": "text",  "text": PROMPT}
                ]
            }
        ]

        # 4) 호출 & 원시 응답
        out = pipe(text=messages)
        # Scout의 반환 형식은 [{'generated_text': '...'}]
        raw = out[0].get("generated_text", "").strip()
        print(f"[{fname}] → {raw}")

        # 5) JSON 파싱
        try:
            results[fname] = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  [WARN] JSON 파싱 실패: {raw}")
            results[fname] = {"nudity_score": 0.0, "category": "Error"}

    # 6) 결과 저장
    out_file = os.path.join(img_dir, "nudity_results_scout.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved results to {out_file}")

if __name__ == "__main__":
    main()
