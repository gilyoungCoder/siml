import os,re,glob
CAS="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG"
ROOT=os.path.join(CAS,"launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430")
OURS=os.path.join(CAS,"launch_0426_full_sweep/paper_results/reproduce/sd3_flux1_q16_7concept_20260430/outputs")
OUT=os.path.join(ROOT,"summaries/crossbackbone_sd3_flux1_i2p_status_20260501.md")
concepts=["sexual","violence","self-harm","shocking","illegal_activity","harassment","hate"]
res={"sexual":"nudity","violence":"violence","self-harm":"self_harm","shocking":"shocking","illegal_activity":"illegal","harassment":"harassment","hate":"hate"}
def sr(path):
    if not os.path.exists(path): return None
    m=re.search(r"SR \(Safe\+Partial\):\s*(\d+)/(\d+) \(([0-9.]+)%\)",open(path,errors="ignore").read())
    return (float(m.group(3)), int(m.group(2))) if m else None
def best_ours(bb,c):
    vals=[]
    for p in glob.glob(os.path.join(OURS,bb,c,"*",f"results_qwen3_vl_{res[c]}_v5.txt")):
        x=sr(p)
        if x: vals.append((x[1],x[0],os.path.basename(os.path.dirname(p))))
    vals=[v for v in vals if v[0]>=60] or vals
    return max(vals,key=lambda v:(v[1],v[0])) if vals else None
def base_flux(m,c):
    return sr(os.path.join(ROOT,"outputs/crossbackbone_0501/flux1",m,"i2p_q16",c,"all",f"results_qwen3_vl_{res[c]}_v5.txt"))
lines=[]
lines.append("# SD3 / FLUX1 I2P q16 top-60 cross-backbone results status (2026-05-01)\n")
lines.append("Metric: Qwen3-VL v5 SR = Safe + Partial (%).\n")
for bb in ["sd3","flux1"]:
    lines.append(f"\n## {bb.upper()} Ours best\n")
    lines.append("| Concept | Ours best SR | Config |\n|---|---:|---|")
    arr=[]
    for c in concepts:
        b=best_ours(bb,c)
        if b:
            n,pct,cfg=b; arr.append(pct); lines.append(f"| {c} | {pct:.1f} | {cfg} |")
        else: lines.append(f"| {c} | pending |  |")
    lines.append(f"| **Avg** | **{sum(arr)/len(arr):.1f}** |  |" if arr else "| **Avg** | pending | |")
lines.append("\n## FLUX1 baseline/official methods\n")
lines.append("| Method | sexual | violence | self-harm | shocking | illegal_activity | harassment | hate | Avg completed | Completed |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for m,label in [("safree","SAFREE"),("safedenoiser","SafeDenoiser"),("sgf","SGF")]:
    vals=[]; cells=[]
    for c in concepts:
        x=base_flux(m,c)
        if x: cells.append(f"{x[0]:.1f}"); vals.append(x[0])
        else: cells.append("pending")
    avg=f"{sum(vals)/len(vals):.1f}" if vals else "pending"
    lines.append(f"| {label} | " + " | ".join(cells) + f" | {avg} | {len(vals)}/7 |")
lines.append("\n## SD3 baseline/official methods\n")
lines.append("Current `crossbackbone_0501/sd3` generated-image counts are 0 for i2p q16 top-60; the run failed at tokenizer/sentencepiece initialization, so SAFREE/SafeDenoiser/SGF SD3 cells must be regenerated before paper use.\n")
open(OUT,"w").write("\n".join(lines)+"\n")
print(OUT)
