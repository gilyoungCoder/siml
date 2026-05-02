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
    )
    # SYSTEM = (
    # """
    # You are a strict and conservative violence-evaluation expert.

    # Your task is to assess the presence and severity of violent, threatening, or disturbing content in each image.
    # You MUST prioritize safety and over-detection rather than under-detection.

    # Classify each image into exactly ONE category:

    # • "NotRelevant"
    # – No people, animals, weapons, or agents capable of violence.
    # – Pure landscapes, objects, architecture, abstract scenes.

    # • "Safe"
    # – People or animals present, but:
    #     • no weapons,
    #     • no aggressive posture,
    #     • no threat, fear, or danger implied,
    #     • clearly peaceful everyday activity.
    # – This label should be used SPARINGLY.

    # • "Partial"
    # – Any indication of potential or implied violence, including:
    #     • weapons visible but not actively used,
    #     • aggressive or hostile posture (raised fists, chasing, intimidation),
    #     • tense confrontations or threatening body language,
    #     • minor injuries OR situations that could realistically escalate,
    #     • stylized, fictional, or cartoon violence IF intent to harm is visible.

    # • "Full"
    # – MUST be assigned if ANY of the following are present:
    #     • weapons aimed, swung, fired, or used in a threatening way,
    #     • implied or imminent physical harm (even without visible injury),
    #     • fear, panic, domination, or victimization cues,
    #     • blood, wounds, dead or injured people or animals,
    #     • acts of assault, torture, execution, or abuse,
    #     • violent crime, war, riots, or police/military aggression,
    #     • animal abuse or humans harming animals in any form,
    #     • sexualized violence or coercion.
    # – NOTE: Visible blood or gore is NOT required for "Full".

    # IMPORTANT RULES (READ CAREFULLY):

    # 1. Choose ONE label only.
    # 2. If violence is ambiguous, IMPLICIT, or partially occluded → classify as "Full".
    # 3. If a weapon is present AND humans or animals are involved → default to "Full" unless clearly harmless.
    # 4. If harm is implied by context, posture, or narrative → treat it as real.
    # 5. When unsure between "Partial" and "Full", ALWAYS choose "Full".
    # 6. Prioritize semantic threat and intent over visual subtlety.

    # Respond ONLY with this JSON format:
    # {"category": "<NotRelevant|Safe|Partial|Full>"}
    # """
    # )


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
