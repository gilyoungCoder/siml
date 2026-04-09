import pandas as pd
from huggingface_hub import hf_hub_download

for name in ["safety", "alignment", "bias", "quality"]:
    path = hf_hub_download("MJ-Bench/MJ-Bench", f"data/{name}.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    non_empty = (df["info"].str.strip() != "").sum()
    unique = df["info"].nunique()
    print(f"{name}: {len(df)} rows, info non-empty={non_empty}, unique={unique}")
    if non_empty > 0:
        vals = df[df["info"].str.strip() != ""]["info"].head(10).tolist()
        print(f"  Sample: {vals}")
    print()
