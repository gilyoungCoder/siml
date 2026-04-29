#!/usr/bin/env python3
"""NFE full v2: 11 step values × 4 concepts × 3 methods (EBSG, SAFREE, Baseline SD1.4)."""
import json, os, sys, subprocess
from pathlib import Path

REPO = "/mnt/home3/yhgil99/unlearning"
PR = f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/paper_results"
OUT_BASE = f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/outputs/phase_nfe_full"
PYTHON_EBSG = "/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
PYTHON_SAFREE = "/mnt/home3/yhgil99/.conda/envs/safree/bin/python3.10"
PYTHON_BASELINE = "/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
BASELINE_SCRIPT = f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/scripts/baseline_runner.py"

CONCEPTS = ["violence", "shocking", "self-harm", "sexual"]
STEPS = [1, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50]
METHODS = ["ebsg", "safree", "baseline"]

S_CAT = {"violence":"violence","shocking":"shocking","self-harm":"self-harm","sexual":"nudity"}

CELLS = []
for method in METHODS:
    for concept in CONCEPTS:
        for steps in STEPS:
            CELLS.append((method, concept, steps))


def run_ebsg(concept, steps, outdir, gpu, log):
    args_path = f"{PR}/single/{concept}/args.json"
    if not os.path.isfile(args_path):
        with open(log,"a") as f: f.write(f"[err-noargs] {concept}\n")
        return 1
    a = json.load(open(args_path))
    cmd = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", PYTHON_EBSG, "-m", "safegen.generate_family",
           "--prompts", a["prompts"], "--outdir", outdir,
           "--family_guidance", "--family_config", a["family_config"],
           "--probe_mode", a["probe_mode"], "--how_mode", a["how_mode"],
           "--cas_threshold", str(a["cas_threshold"]),
           "--safety_scale", str(a["safety_scale"]),
           "--attn_threshold", str(a["attn_threshold"]),
           "--img_attn_threshold", str(a["img_attn_threshold"]),
           "--n_img_tokens", str(a.get("n_img_tokens",4)),
           "--steps", str(steps), "--seed", str(a.get("seed",42)),
           "--cfg_scale", str(a.get("cfg_scale",7.5)),
           "--target_concepts", *a["target_concepts"],
           "--anchor_concepts", *a["anchor_concepts"]]
    with open(log,"a") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              cwd=f"{REPO}/SafeGen").returncode

def run_safree(concept, steps, outdir, gpu, log):
    a = json.load(open(f"{PR}/single/{concept}/args.json"))
    cmd = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", PYTHON_SAFREE,
           f"{REPO}/SAFREE/gen_safree_single.py",
           "--txt", a["prompts"], "--save-dir", outdir,
           "--category", S_CAT[concept],
           "--re_attn_t=-1,1001",
           "--linear_per_prompt_seed",
           "--num_inference_steps", str(steps),
           "--safree", "-svf", "-lra"]
    with open(log,"a") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                            cwd=f"{REPO}/SAFREE").returncode
    gendir = f"{outdir}/generated"
    if os.path.isdir(gendir):
        for fn in os.listdir(gendir):
            if fn.endswith(".png"): os.rename(f"{gendir}/{fn}", f"{outdir}/{fn}")
        try: os.rmdir(gendir)
        except: pass
    return rc

def run_baseline(concept, steps, outdir, gpu, log):
    a = json.load(open(f"{PR}/single/{concept}/args.json"))
    cmd = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", PYTHON_BASELINE,
           BASELINE_SCRIPT,
           "--prompts", a["prompts"], "--outdir", outdir,
           "--steps", str(steps), "--seed", str(a.get("seed",42)),
           "--cfg_scale", str(a.get("cfg_scale",7.5)),
           "--device", "cuda:0"]   # CUDA_VISIBLE_DEVICES sets which is cuda:0
    with open(log,"a") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              cwd=f"{REPO}/SafeGen").returncode


def main():
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    slot = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nslots = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    LOGDIR = f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/logs"
    log = f"{LOGDIR}/nfe_full_v2_g{gpu}_s{slot}.log"
    Path(OUT_BASE).mkdir(parents=True, exist_ok=True); Path(LOGDIR).mkdir(parents=True, exist_ok=True)
    with open(log,"a") as f: f.write(f"[start] GPU={gpu} slot={slot} nslots={nslots}\n")

    for i, (method, concept, steps) in enumerate(CELLS):
        if (i % nslots) != slot: continue
        cell = f"{method}_{concept}_steps{steps}"
        outdir = f"{OUT_BASE}/{cell}"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        existing = len(list(Path(outdir).glob("*.png")))
        if existing >= 60:
            with open(log,"a") as f: f.write(f"[skip-done] {cell} ({existing}/60)\n")
            continue
        with open(log,"a") as f: f.write(f"[run] {cell}\n")
        try:
            if method == "ebsg":     rc = run_ebsg(concept, steps, outdir, gpu, log)
            elif method == "safree": rc = run_safree(concept, steps, outdir, gpu, log)
            else:                    rc = run_baseline(concept, steps, outdir, gpu, log)
            final = len(list(Path(outdir).glob("*.png")))
            with open(log,"a") as f: f.write(f"[done] {cell} rc={rc} imgs={final}\n")
        except Exception as e:
            with open(log,"a") as f: f.write(f"[exc] {cell} {e}\n")

    with open(log,"a") as f: f.write(f"[end] GPU={gpu} slot={slot}\n")


if __name__ == "__main__":
    main()
