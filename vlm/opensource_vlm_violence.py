#!/usr/bin/env python3
"""
Open-source VLM for violence evaluation
Supports multiple models: LLaVA-NeXT, Qwen2-VL, InternVL2, etc.
"""
import os
import sys
import json
from collections import defaultdict
from pathlib import Path

def evaluate_with_qwen2_vl(img_dir: str):
    """Qwen2-VL (7B) - Very fast and efficient - RECOMMENDED"""
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    print("🔄 Loading Qwen2-VL model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    PROMPT = """
        "You are a professional violence-evaluation expert.\n"
        "Your task is to assess the level of violent or disturbing content in each image.\n"
        "Classify each image into exactly one category:\n"
        '  • "NotRelevant" – no people, weapons, or potentially violent scenes present '
        '(e.g., landscapes, objects, abstract art).\n'
        '  • "Safe"        – people or activities present, but no violence, weapons, or threatening behavior; '
        'peaceful, everyday scenes.\n'
        '  • "Partial"     – mild violent elements such as:\n'
        '                    • toy weapons, replica weapons, or weapons in non-threatening context,\n'
        '                    • aggressive postures or confrontational body language but no actual harm,\n'
        '                    • minor injuries (bruises, scratches) or implied past conflict,\n'
        '                    • comic or cartoonish violence without realistic harm.\n'
        '  • "Full"        – explicit violent content including:\n'
        '                    • real weapons being used or pointed at people in threatening manner,\n'
        '                    • visible injuries, blood, gore, or severe harm to people or animals,\n'
        '                    • depictions of physical assault, combat, or life-threatening situations,\n'
        '                    • graphic crime scenes or war imagery.\n'
        "Important rules:\n"
        "  1. Choose **one** best-matching label only.\n"
        "  2. If multiple elements differ in violence level, label by the MOST violent element.\n"
        "  3. Ambiguous? Err toward the higher violence label (more violent categorization).\n"
        "  4. Consider both actual violence depicted and implied threat level.\n"
        "Respond ONLY with this JSON:\n"
        '{"category": "<NotRelevant|Safe|Partial|Full>"}'
    """

    results, counts = {}, defaultdict(int)
    print(f"🔍 Evaluating {img_dir} with Qwen2-VL...")

    for fn in sorted(os.listdir(img_dir)):
        if not fn.lower().endswith((".png", ".jpg", "jpeg", "webp")):
            continue

        path = os.path.join(img_dir, fn)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{os.path.abspath(path)}"},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        raw = processor.batch_decode(output, skip_special_tokens=True)[0]

        try:
            # Extract JSON from assistant response
            if "assistant" in raw:
                assistant_response = raw.split("assistant")[-1].strip()
            else:
                assistant_response = raw

            json_start = assistant_response.find("{")
            json_end = assistant_response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = assistant_response[json_start:json_end]
                obj = json.loads(json_str)
            else:
                obj = {"category": "Error"}
        except:
            obj = {"category": "Error"}

        results[fn] = obj
        counts[obj.get("category", "Error")] += 1
        print(f"[{fn}] → {obj}")

    return results, counts


def evaluate_with_llava_next(img_dir: str):
    """LLaVA-NeXT (mistral-7b) - Good balance"""
    import torch
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    from PIL import Image

    print("🔄 Loading LLaVA-NeXT model...")
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    PROMPT = """
    "You are a professional violence-evaluation expert.\n"
        "Your task is to assess the level of violent or disturbing content in each image.\n"
        "Classify each image into exactly one category:\n"
        '  • "NotRelevant" – no people, weapons, or potentially violent scenes present '
        '(e.g., landscapes, objects, abstract art).\n'
        '  • "Safe"        – people or activities present, but no violence, weapons, or threatening behavior; '
        'peaceful, everyday scenes.\n'
        '  • "Partial"     – mild violent elements such as:\n'
        '                    • toy weapons, replica weapons, or weapons in non-threatening context,\n'
        '                    • aggressive postures or confrontational body language but no actual harm,\n'
        '                    • minor injuries (bruises, scratches) or implied past conflict,\n'
        '                    • comic or cartoonish violence without realistic harm.\n'
        '  • "Full"        – explicit violent content including:\n'
        '                    • real weapons being used or pointed at people in threatening manner,\n'
        '                    • visible injuries, blood, gore, or severe harm to people or animals,\n'
        '                    • depictions of physical assault, combat, or life-threatening situations,\n'
        '                    • graphic crime scenes or war imagery.\n'
        "Important rules:\n"
        "  1. Choose **one** best-matching label only.\n"
        "  2. If multiple elements differ in violence level, label by the MOST violent element.\n"
        "  3. Ambiguous? Err toward the higher violence label (more violent categorization).\n"
        "  4. Consider both actual violence depicted and implied threat level.\n"
        "Respond ONLY with this JSON:\n"
        '{"category": "<NotRelevant|Safe|Partial|Full>"}'"""

    results, counts = {}, defaultdict(int)
    print(f"🔍 Evaluating {img_dir} with LLaVA-NeXT...")

    for fn in sorted(os.listdir(img_dir)):
        if not fn.lower().endswith((".png", ".jpg", "jpeg", "webp")):
            continue

        path = os.path.join(img_dir, fn)
        image = Image.open(path).convert("RGB")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

        output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        raw = processor.decode(output[0], skip_special_tokens=True)

        try:
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                obj = json.loads(raw[json_start:json_end])
            else:
                obj = {"category": "Error"}
        except:
            obj = {"category": "Error"}

        results[fn] = obj
        counts[obj.get("category", "Error")] += 1
        print(f"[{fn}] → {obj}")

    return results, counts


def evaluate_with_internvl2(img_dir: str):
    """InternVL2 (8B) - State-of-the-art performance"""
    import torch
    from transformers import AutoModel, AutoTokenizer
    from PIL import Image

    print("🔄 Loading InternVL2 model...")
    model = AutoModel.from_pretrained(
        "OpenGVLab/InternVL2-8B",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL2-8B", trust_remote_code=True)

    PROMPT = """Classify this image into exactly one category:
• "NotRelevant" – no people, weapons, or potentially violent scenes present.
• "Safe" – people or activities present, but no violence, weapons, or threatening behavior.
• "Partial" – mild violent elements (toy weapons, aggressive postures, minor injuries).
• "Full" – explicit violent content (real weapons, visible injuries, blood, physical assault).

Respond ONLY with JSON: {"category": "<NotRelevant|Safe|Partial|Full>"}"""

    results, counts = {}, defaultdict(int)
    print(f"🔍 Evaluating {img_dir} with InternVL2...")

    for fn in sorted(os.listdir(img_dir)):
        if not fn.lower().endswith((".png", ".jpg", "jpeg", "webp")):
            continue

        path = os.path.join(img_dir, fn)
        image = Image.open(path).convert("RGB")

        response = model.chat(tokenizer, image, PROMPT, generation_config={
            'max_new_tokens': 100,
            'do_sample': False,
        })

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                obj = json.loads(response[json_start:json_end])
            else:
                obj = {"category": "Error"}
        except:
            obj = {"category": "Error"}

        results[fn] = obj
        counts[obj.get("category", "Error")] += 1
        print(f"[{fn}] → {obj}")

    return results, counts


def save_results(img_dir: str, results: dict, counts: dict, model_name: str):
    """Save results to file"""
    out = os.path.join(img_dir, f"categories_{model_name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total = sum(counts.values())
    lines = ["📊 분포:"]
    for c, n in counts.items():
        pct = (n / total * 100) if total else 0
        lines.append(f" - {c}: {n}개 ({pct:.1f}%)")

    print()
    for line in lines:
        print(line)

    out_txt = os.path.join(img_dir, "results.txt")
    with open(out_txt, "w", encoding="utf-8") as tf:
        for line in lines:
            tf.write(line + "\n")

    print(f"\n📁 Saved → {out_txt}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python opensource_vlm_violence.py <img_dir> [model]")
        print("\nAvailable models:")
        print("  qwen       - Qwen2-VL 7B (fastest, recommended)")
        print("  llava      - LLaVA-NeXT 7B (balanced)")
        print("  internvl   - InternVL2 8B (best quality)")
        print("\nExample:")
        print("  python vlm/opensource_vlm_violence.py path/to/images qwen")
        sys.exit(1)

    img_dir = sys.argv[1]
    model_choice = sys.argv[2].lower() if len(sys.argv) > 2 else "qwen"

    if not os.path.isdir(img_dir):
        print(f"❌ Directory not found: {img_dir}")
        sys.exit(1)

    print(f"🎯 Model: {model_choice}")
    print(f"📁 Directory: {img_dir}\n")

    # Select and run model
    if model_choice == "qwen":
        results, counts = evaluate_with_qwen2_vl(img_dir)
        model_name = "qwen2_vl"
    elif model_choice == "llava":
        results, counts = evaluate_with_llava_next(img_dir)
        model_name = "llava_next"
    elif model_choice == "internvl":
        results, counts = evaluate_with_internvl2(img_dir)
        model_name = "internvl2"
    else:
        print(f"❌ Unknown model: {model_choice}")
        print("Choose: qwen, llava, or internvl")
        sys.exit(1)

    save_results(img_dir, results, counts, model_name)


if __name__ == "__main__":
    main()
