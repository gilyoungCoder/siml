import pandas as pd
import os

# CSV 파일 불러오기
df = pd.read_csv("i2p_benchmark.csv")

# prompt와 categories 열만 사용
df = df[["prompt", "categories"]]

# categories 안에 여러 개의 카테고리가 있을 수 있으므로 , 로 분리
for idx, row in df.iterrows():
    categories = [c.strip() for c in str(row["categories"]).split(",")]
    for cat in categories:
        filename = f"{cat}.txt"
        # 프롬프트를 파일에 추가 (append)
        with open(filename, "a", encoding="utf-8") as f:
            f.write(row["prompt"] + "\n")

print("카테고리별 txt 파일 생성 완료!")
