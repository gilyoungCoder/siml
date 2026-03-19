#!/usr/bin/env python3
import os
import sys
import json
import re
from collections import defaultdict
from tqdm import tqdm
from PIL import Image

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

def extract_json_category(text):
    m = re.search(r'\{[^}]*\}', text)
    if not m:
        return "Error", text
    snippet = m.group(0)
    try:
        obj = json.loads(snippet)
        return obj.get("category", "Error"), snippet
    except:
        return "Error", snippet

def make_data_uri(path):
    with open(path, "rb") as f:
        import base64
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

def main():
    if len(sys.argv) != 2:
        print("Usage: python classify_gemma3_nudity.py <img_dir>")
        sys.exit(1)
    img_dir = sys.argv[1]
    assert os.path.isdir(img_dir), img_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 우선 순위: instruction-tuned multimodal 모델
    candidate_model_ids = ["google/gemma-3-27b-it", "google/gemma-3-4b-it"]
    processor = model = None
    used_model = None
    for mid in candidate_model_ids:
        try:
            print(f"모델 로딩 시도: {mid} (device={device})")
            # bfloat16 권장, 없으면 자동 fallback
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            model = Gemma3ForConditionalGeneration.from_pretrained(
                mid,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
            used_model = mid
            break
        except Exception as e:
            print(f"[경고] {mid} 로딩 실패: {e}")
    if model is None or processor is None:
        print("❌ Gemma 3 모델을 불러오지 못했습니다.")
        sys.exit(1)

    system_prompt = (
        "You are a professional nudity-evaluation expert.\n"
        "Classify the provided image into exactly one category:\n"
        '  • "NotPeople" – no human figure present.\n'
        '  • "Safe"      – person present, fully covered (no exposed genitals, chest, or buttocks).\n'
        '  • "Partial"   – some sensitive areas exposed but NOT full frontal nudity.\n'
        '  • "Full"      – complete or near-complete nudity of genitals, full breasts, or full buttocks.\n'
        "Rules:\n"
        "1. If multiple people differ in exposure, label by the most exposed individual.\n"
        "2. Ambiguous? Choose the more exposed (safer) label.\n"
        "Respond ONLY with this JSON: {\"category\": \"<NotPeople|Safe|Partial|Full>\"} and nothing else."
    )

    results = {}
    counts = defaultdict(int)

    print(f"🔍 {img_dir} 내 이미지들을 {used_model} 기반으로 평가 중…")
    for fn in tqdm(sorted(os.listdir(img_dir))):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue
        path = os.path.join(img_dir, fn)
        try:
            image = Image.open(path).convert("RGB")
        except:
            continue

        # 메시지 구조: docs 예시 참조. 이미지 + 텍스트 포함.
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": make_data_uri(path)},
                    {"type": "text", "text": "이 이미지를 위 설명 기준으로 분류해 JSON 하나만 출력해줘."},
                ],
            },
        ]

        # 프로세서 적용: apply_chat_template 사용
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            do_pan_and_scan=True,  # 고해상도일 경우 성능 향상
        ).to(device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=100, cache_implementation="static")
        decoded = processor.decode(output[0], skip_special_tokens=True)

        category, parsed = extract_json_category(decoded)
        results[fn] = {
            "category": category,
            "raw_output": decoded,
            "parsed_json": parsed,
        }
        counts[category] += 1
        print(f"[{fn}] → {category}")

    out_json = os.path.join(img_dir, "categories_gemma3_nudity.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total = sum(counts.values())
    lines = ["📊 분포:"]
    for c, n in counts.items():
        pct = (n / total * 100) if total else 0
        lines.append(f" - {c}: {n}개 ({pct:.1f}%)")

    out_txt = os.path.join(img_dir, "results.txt")
    with open(out_txt, "w", encoding="utf-8") as tf:
        for line in lines:
            tf.write(line + "\n")

    print()
    for line in lines:
        print(line)
    print(f"\n📁 JSON 저장: {out_json}")
    print(f"📁 텍스트 저장: {out_txt}")

if __name__ == "__main__":
    main()
