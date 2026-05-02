import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download("MJ-Bench/MJ-Bench", "data/safety.parquet", repo_type="dataset")
df = pd.read_parquet(path)
print("Rows:", len(df))
print("Columns:", list(df.columns))
print("info nunique:", df["info"].nunique())
print()
print("info value_counts:")
print(df["info"].value_counts())
print()
print("First 10 info values (repr):")
for i in range(10):
    print(f"  [{i}] {repr(df.iloc[i]['info'])}")
print()

# Check caption for NSFW keywords
nsfw_kw = ["nude", "naked", "nsfw", "sexual", "porn", "breast", "bikini"]
toxicity_kw = ["blood", "gore", "crime", "murder", "weapon", "scary", "monster"]

for kw_name, kws in [("NSFW", nsfw_kw), ("Toxicity", toxicity_kw)]:
    count = 0
    for cap in df["caption"]:
        if any(k in str(cap).lower() for k in kws):
            count += 1
    print(f"Captions matching {kw_name} keywords: {count}/{len(df)}")

# Try to infer subcategory from caption content
print()
print("Sample captions:")
for i in range(20):
    print(f"  [{i}] {df.iloc[i]['caption'][:80]}")
