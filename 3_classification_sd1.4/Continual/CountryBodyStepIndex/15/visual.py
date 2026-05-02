import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) CSV 읽기 (경로만 바꿔주세요)
csv_path = "./fai_log.csv"  # <- 여기만 수정
df = pd.read_csv(csv_path)

# 혹시 '...' 같은 깨진 라인이 있으면 걸러내기
df = df[pd.to_numeric(df['FAI'], errors='coerce').notna()]
df = df[pd.to_numeric(df['step'], errors='coerce').notna()]
df = df[pd.to_numeric(df['prompt_idx'], errors='coerce').notna()]
df[['prompt_idx','step','timestep']] = df[['prompt_idx','step','timestep']].astype(int)
df['FAI'] = df['FAI'].astype(float)

# 2) 초반/후반 평균 및 delta
early = df[df['step'].between(0,5)].groupby('prompt_idx')['FAI'].mean().rename('early_mean')
late  = df[df['step'].between(45,50)].groupby('prompt_idx')['FAI'].mean().rename('late_mean')
summ  = pd.concat([early, late], axis=1)
summ['delta'] = summ['early_mean'] - summ['late_mean']
summ = summ.sort_values('delta', ascending=False)

print("=== Per-prompt early/late means and delta (desc) ===")
print(summ.head(15))
print("\nOverall delta (mean ± std): "
      f"{summ['delta'].mean():.3f} ± {summ['delta'].std():.3f}  (n={len(summ)})")

# 3) step별 전 프롬프트 평균/표준편차
step_stats = df.groupby('step')['FAI'].agg(['mean','std','count'])
print("\n=== Step-wise mean/std ===")
print(step_stats.head())

# 4) 시각화
plt.figure(figsize=(7,4))
plt.plot(step_stats.index, step_stats['mean'], label='mean FAI across prompts')
plt.fill_between(step_stats.index,
                 step_stats['mean']-step_stats['std'],
                 step_stats['mean']+step_stats['std'],
                 alpha=0.2, label='±1 std')
plt.xlabel('step'); plt.ylabel('FAI'); plt.title('Global FAI vs step')
plt.legend(); plt.tight_layout(); plt.show()

# 프롬프트 몇 개 샘플 궤적(상위 delta 3개 + 하위 delta 3개)
top3 = summ.head(3).index.tolist()
bot3 = summ.tail(3).index.tolist()
pick = top3 + bot3

plt.figure(figsize=(7,4))
for pid in pick:
    sub = df[df['prompt_idx']==pid].sort_values('step')
    plt.plot(sub['step'], sub['FAI'], label=f'prompt {pid}')
plt.xlabel('step'); plt.ylabel('FAI'); plt.title('Sample FAI trajectories')
plt.legend(ncol=2, fontsize=8); plt.tight_layout(); plt.show()
