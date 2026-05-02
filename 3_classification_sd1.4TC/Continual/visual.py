# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 비교할 파일들: {범례라벨: 파일경로}
FILES = {
    "Classifier guidance": "/mnt/home/yhgil99/unlearning/3_classification_sd1.4/Continual/CountryNudeBody/single_m/fai_log.csv",
    "No Guidance": "/mnt/home/yhgil99/unlearning/3_classification_sd1.4/Continual/CountryNudeBody/no_guidance_mm/fai_log.csv",
    "Benign Human": "/mnt/home/yhgil99/unlearning/3_classification_sd1.4/Continual/CountryNudeBody/coco_m/fai_log.csv",
    "Cross-Attention Suppression Soft delete": "./CountryNudeBody/soft_delete15,0.5cg10,5/fai_log.csv",
    "Cross-Attention Suppression Dual(Hard+Soft)": "./CountryNudeBody/dual_soft_hard_tau0.20_tauH0.45+cg/fai_log.csv"
}

def load_step_mean(csv_path: str) -> pd.Series:
    """CSV를 읽어 step별 FAI 평균(프롬프트 전체 평균)을 반환."""
    df = pd.read_csv(csv_path)
    # 필요한 컬럼 체크
    for col in ["step", "FAI"]:
        if col not in df.columns:
            raise ValueError(f"{csv_path}: '{col}' 컬럼이 필요합니다. 실제 컬럼: {list(df.columns)}")
    # step별로 FAI 평균
    s = df.groupby("step", as_index=True)["FAI"].mean().sort_index()
    return s

# 모든 파일을 읽어 step 축으로 결합(outer join)
summary = pd.DataFrame()
for label, path in FILES.items():
    s = load_step_mean(path)
    summary[label] = s

# 결과를 CSV로도 저장(옵션)
out_csv = Path("fai_step_means_summary.csv")
summary.to_csv(out_csv, encoding="utf-8-sig")
print(f"[저장] step별 평균 요약: {out_csv.resolve()}")
# 그래프
plt.figure(figsize=(10, 6))
for label in summary.columns:
    plt.plot(summary.index, summary[label], marker="o", linewidth=2, label=label)
plt.title("FAI Mean per Diffusion Step")
plt.xlabel("Step")
plt.ylabel("FAI (mean across prompts)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# 항상 파일로 저장
fig_dir = Path("figs"); fig_dir.mkdir(exist_ok=True)
png_path = fig_dir / "fai_various_soft.png"

plt.savefig(png_path, dpi=200)
print(f"[저장] 그래프 PNG: {png_path.resolve()}")

# GUI 가능하면 창도 띄우기
try:
    if plt.get_backend().lower() not in ("agg", "svg", "pgf"):
        plt.show()  # 가능한 환경에서만 창 표시
except Exception as e:
    print(f"plt.show() 생략: {e}")