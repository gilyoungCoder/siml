#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
from typing import List, Optional

import torch
from transformers import CLIPModel, CLIPTokenizer


def load_clip(model_name: str = "openai/clip-vit-large-patch14", device: Optional[str] = None):
    """
    CLIP 모델과 토크나이저를 로드합니다.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


@torch.no_grad()
def get_text_features(tokenizer, model, texts: List[str], device: str) -> torch.Tensor:
    """
    텍스트 리스트를 CLIP 텍스트 임베딩으로 변환합니다.
    출력은 L2 정규화된 임베딩(batch, dim)입니다.
    """
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    # HF의 CLIPModel.get_text_features(normalize=True)가 L2 정규화까지 수행합니다.
    feats = model.get_text_features(**inputs)  # (B, D)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def parse_word_list(words_arg: Optional[str], file_arg: Optional[str]) -> List[str]:
    """
    --words 또는 --file로부터 단어(문구) 리스트를 파싱합니다.
    """
    items: List[str] = []
    if words_arg:
        # 쉼표/세미콜론/라인브레이크 모두 구분자로 처리
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


def pretty_print(target: str, queries: List[str], scores: torch.Tensor, topk: Optional[int] = None):
    """
    콘솔 로그로 깔끔하게 출력합니다.
    """
    pairs = list(zip(queries, scores.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)
    if topk is not None:
        pairs = pairs[:topk]

    width = max([len(q) for q, _ in pairs] + [len("query")])
    print(f"[CLIP cosine similarity] target = '{target}'\n")
    print(f"{'rank':>4}  {'query'.ljust(width)}  cos")
    print("-" * (10 + width + 6))
    for i, (q, s) in enumerate(pairs, start=1):
        print(f"{i:>4}  {q.ljust(width)}  {s: .4f}")


def main():
    parser = argparse.ArgumentParser(description="Compute cosine similarity to a target text with CLIP text embeddings.")
    parser.add_argument(
        "--model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Hugging Face CLIP model name (e.g., openai/clip-vit-base-patch32).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="nude",
        help="기준이 되는 텍스트(기본값: 'nude').",
    )
    parser.add_argument(
        "--words",
        type=str,
        default=None,
        help="쉼표 또는 세미콜론으로 구분된 단어/문구 리스트. 예) \"fully clothed, bikini, space\"",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="단어/문구 목록이 들어있는 텍스트 파일 경로(줄 단위).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="상위 K개만 표시하고 싶다면 지정합니다.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="강제로 'cuda' 또는 'cpu' 지정 가능. 기본은 자동 판별.",
    )

    args = parser.parse_args()

    # 단어 리스트 수집
    queries = parse_word_list(args.words, args.file)

    # 인터랙티브 입력(둘 다 비었을 때)
    if not queries:
        print("비교할 단어/문구를 한 줄에 쉼표로 구분하여 입력한 뒤 Enter 하십시오.")
        line = sys.stdin.readline()
        queries = parse_word_list(line, None)

    if not queries:
        print("입력된 단어가 없습니다. --words 또는 --file을 사용하십시오.")
        sys.exit(1)

    # 모델 로드 및 임베딩 계산
    tokenizer, model, device = load_clip(args.model, args.device)
    texts = [args.target] + queries
    feats = get_text_features(tokenizer, model, texts, device)  # (1 + N, D)
    target_feat = feats[0:1]  # (1, D)
    query_feats = feats[1:]   # (N, D)

    # 코사인 유사도: 이미 L2 정규화되어 있으므로 내적이 곧 코사인
    scores = torch.matmul(query_feats, target_feat.T).squeeze(-1)  # (N,)

    # 출력
    pretty_print(args.target, queries, scores, topk=args.topk)


if __name__ == "__main__":
    main()
