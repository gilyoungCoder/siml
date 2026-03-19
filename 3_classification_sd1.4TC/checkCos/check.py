#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
from typing import List, Optional, Dict, Tuple

import torch
from transformers import CLIPModel, CLIPTokenizer


def load_clip(model_name: str = "openai/clip-vit-large-patch14", device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    return tokenizer, model, device


def parse_lines(words_arg: Optional[str], file_arg: Optional[str]) -> List[str]:
    items: List[str] = []
    if words_arg:
        raw = words_arg.replace(";", ",")
        items.extend([w.strip() for w in raw.split(",") if w.strip()])
    if file_arg:
        with open(file_arg, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if t:
                    items.append(t)
    # 중복 제거(순서 보존)
    seen = set()
    uniq = []
    for w in items:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq


def apply_template(s: str, template: str) -> str:
    return template.format(s) if "{}" in template else f"{template} {s}".strip()


def find_subseq(long: List[int], short: List[int]) -> Optional[Tuple[int, int]]:
    """long에서 short 서브시퀀스(연속) 최초 매치를 찾아 (start, end) 반환. 없으면 None."""
    n, m = len(long), len(short)
    if m == 0 or m > n:
        return None
    for i in range(n - m + 1):
        if long[i:i+m] == short:
            return i, i + m  # [start, end)
    return None


@torch.no_grad()
def embed_concept_only(
    tokenizer: CLIPTokenizer,
    model: CLIPModel,
    concepts: List[str],
    template: str,
    device: str,
) -> torch.Tensor:
    """
    각 개념 c에 대해 full = template.format(c)를 토크나이즈 → text_model last_hidden_state 획득.
    그 안에서 c를 add_special_tokens=False로 토크나이즈한 토큰 시퀀스를 full의 input_ids에서 서브시퀀스로 찾아
    해당 구간 토큰만 평균 풀링 → text_projection → L2 정규화하여 반환.

    매칭 실패 시, c 단독(add_special_tokens=False) 문장으로 한 번 더 인코딩하여 그 토큰만 평균 풀링.
    """
    # 1) 템플릿 문장 배치 인코딩
    full_texts = [apply_template(c, template) for c in concepts]
    full_inputs = tokenizer(full_texts, padding=True, truncation=True, return_tensors="pt").to(device)
    out = model.text_model(input_ids=full_inputs.input_ids, attention_mask=full_inputs.attention_mask)
    H_full = out.last_hidden_state            # (B, L, H)
    input_ids_full = full_inputs.input_ids    # (B, L)

    # 2) 각 개념 토큰화(add_special_tokens=False) 준비
    concept_token_ids_list: List[List[int]] = []
    for c in concepts:
        # 특수토큰 없이 개념 토큰만
        c_ids = tokenizer(c, add_special_tokens=False)["input_ids"]
        concept_token_ids_list.append(c_ids)

    pooled_list = []
    for i, (c_ids) in enumerate(concept_token_ids_list):
        ids_row = input_ids_full[i].tolist()
        # BOS/EOS는 ids_row에 포함되어 있을 수 있으나, c_ids는 특수토큰 없음 → 그대로 서브시퀀스 탐색
        pos = find_subseq(ids_row, c_ids)

        if pos is not None:
            s, e = pos
            # s:e 구간 토큰만 평균
            h_span = H_full[i, s:e, :]                        # (m, H)
            pooled = h_span.mean(dim=0)                       # (H,)
        else:
            # fallback: 개념 단독 문장으로 인코딩 후 평균 풀링
            solo = tokenizer(c, add_special_tokens=False, return_tensors="pt").to(device)
            out_solo = model.text_model(input_ids=solo.input_ids, attention_mask=solo.attention_mask)
            H_solo = out_solo.last_hidden_state              # (1, l, H)
            # attention_mask가 1인 위치 평균
            mask = solo.attention_mask.float()[0].unsqueeze(-1)  # (l,1)
            pooled = (H_solo[0] * mask).sum(dim=0) / mask.sum().clamp_min(1.0)

        # text_projection → L2 정규화
        z = model.text_projection(pooled)                     # (D,)
        z = z / z.norm().clamp_min(1e-8)
        pooled_list.append(z)

    embs = torch.stack(pooled_list, dim=0)                    # (B, D)
    return embs


def print_pairwise(terms: List[str], sims: torch.Tensor):
    n = len(terms)
    width = max(max(len(t) for t in terms), 6)
    header = " " * (width + 2) + "  ".join([t[:10].ljust(10) for t in terms])
    print(header)
    print("-" * len(header))
    for i in range(n):
        row = [f"{s:>10.4f}" for s in sims[i].tolist()]
        print(terms[i].ljust(width + 2) + "  ".join(row))


def print_one_vs_many(target: str, queries: List[str], scores: torch.Tensor):
    pairs = list(zip(queries, scores.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)
    width = max([len(q) for q, _ in pairs] + [len("query")])
    print(f"[CLIP cosine similarity] concept-only  target = '{target}'\n")
    print(f"{'rank':>4}  {'query'.ljust(width)}  cos")
    print("-" * (10 + width + 6))
    for i, (q, s) in enumerate(pairs, start=1):
        print(f"{i:>4}  {q.ljust(width)}  {s: .4f}")


def main():
    ap = argparse.ArgumentParser(description="Cosine similarity using ONLY the concept tokens within the template (CLIP).")
    ap.add_argument("--model", type=str, default="openai/clip-vit-large-patch14")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--template", type=str, default="a {} person", help="예: 'a {} person'")
    ap.add_argument("--words", type=str, default=None, help="쉼표/세미콜론 구분")
    ap.add_argument("--file", type=str, default=None, help="줄 단위 텍스트 파일")
    ap.add_argument("--target", type=str, default="nude", help="타깃 개념(기본: nude).")
    ap.add_argument("--pairwise", action="store_true", help="모든 항목 상호 코사인 행렬 출력.")
    args = ap.parse_args()

    concepts = parse_lines(args.words, args.file)
    if not concepts:
        print("비교할 개념들을 쉼표로 입력 후 Enter 하십시오.")
        line = sys.stdin.readline()
        concepts = parse_lines(line, None)
    if not concepts:
        print("[ERR] 입력이 없습니다. --file 또는 --words를 사용하십시오.", file=sys.stderr)
        sys.exit(1)

    # target 보장
    if args.target not in concepts:
        concepts = [args.target] + concepts
    else:
        idx = concepts.index(args.target)
        concepts = [concepts[idx]] + concepts[:idx] + concepts[idx+1:]

    tokenizer, model, device = load_clip(args.model, args.device)
    embs = embed_concept_only(tokenizer, model, concepts, args.template, device)  # (N, D)

    if args.pairwise:
        sims = embs @ embs.T
        print_pairwise(concepts, sims)
    else:
        target_vec = embs[0:1]
        q_vecs = embs[1:]
        scores = (q_vecs @ target_vec.T).squeeze(-1)
        print_one_vs_many(concepts[0], concepts[1:], scores)


if __name__ == "__main__":
    main()
