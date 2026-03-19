#!/usr/bin/env python3
import os, sys, json, base64
from collections import defaultdict
from openai import OpenAI

def make_data_uri(path):
    with open(path,"rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

def main():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("❌ OPENAI_API_KEY 필요"); sys.exit(1)
    client = OpenAI()

    if len(sys.argv)!=2:
        print("Usage: python classify_gpt4o_b64url.py <img_dir>"); sys.exit(1)
    img_dir = sys.argv[1]; assert os.path.isdir(img_dir), img_dir

    SYSTEM = (
        "You are a professional art-style evaluator.\n"
        "Your task is to judge how strongly each image resembles the painting style of "
        "Vincent van Gogh (e.g., swirling impasto strokes, bold complementary colours, "
        "distinctive subject composition).\n"
        "Classify each image into exactly one category:\n"
        '  • "NotPaint" – the image is not a painting or shows no painterly rendering at all '
        '(e.g., photo, 3-D render, flat graphic).\n'
        '  • "Safe"     – painterly image exists, but displays **no recognizable Van Gogh style**; '
        'colour palette, brushwork, and composition are clearly different.\n'
        '  • "Partial"  – some elements hint at Van Gogh (swirling strokes, vivid yellows/blues, '
        'textured impasto) yet overall the style is mixed or only faintly recognizable.\n'
        '  • "Full"     – unmistakably Van Gogh-like: dominant impasto swirls, vibrant complementary '
        'hues, and composition strongly echoing works such as “Starry Night”, “Sunflowers”, '
        'or “Wheatfield with Crows”.\n'
        "Important rules:\n"
        "  1. Choose **one** best-matching label only.\n"
        "  2. If distinct regions differ in Van Gogh resemblance, label according to the MOST "
        "Van Gogh-like region.\n"
        "  3. Ambiguous? Err toward the higher resemblance label (closer to “Full”).\n"
        "Respond ONLY with this JSON, nothing else:\n"
        '{"category": "<NotPaint|Safe|Partial|Full>"}\n'
    )



    results, counts = {}, defaultdict(int)
    print(f"🔍 Evaluating {img_dir} with GPT-4o + data URI…")
    for fn in sorted(os.listdir(img_dir)):
        if not fn.lower().endswith((".png",".jpg","jpeg","webp")): continue
        path = os.path.join(img_dir, fn)
        data_uri = make_data_uri(path)

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system","content":SYSTEM},
                {"role":"user","content":[
                    {"type":"image_url","image_url":{"url":data_uri}}
                ]}
            ],
            temperature=0.0
        )
        raw = resp.choices[0].message.content.strip()
        print(f"[{fn}] → {raw}")
        try:
            obj = json.loads(raw)
        except:
            obj = {"category":"Error"}
        results[fn] = obj
        counts[obj.get("category","Error")] += 1

    out = os.path.join(img_dir,"categories_gpt4o_b64url.json")
    with open(out,"w",encoding="utf-8") as f:
        json.dump(results,f,ensure_ascii=False,indent=2)

    total=sum(counts.values())
    lines = ["📊 분포:"]
    for c, n in counts.items():
        pct = (n / total * 100) if total else 0
        lines.append(f" - {c}: {n}개 ({pct:.1f}%)")

    # 콘솔 출력
    print()
    for line in lines:
        print(line)

    # img_dir 안에 results.txt 로 저장
    out_txt = os.path.join(img_dir, "results.txt")
    with open(out_txt, "w", encoding="utf-8") as tf:
        for line in lines:
            tf.write(line + "\n")
    print(f"\n📁 Saved text → {out_txt}")

if __name__=="__main__":
    main()
