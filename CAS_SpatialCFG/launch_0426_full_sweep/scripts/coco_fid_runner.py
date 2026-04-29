#!/usr/bin/env python3
"""COCO FID generation runner: split coco_10k.txt across N slots, generate per method.
Args: $1=GPU $2=SLOT $3=NSLOTS $4=METHOD (ebsg|safree|baseline)
"""
import json, os, sys, subprocess
from pathlib import Path

REPO = "/mnt/home3/yhgil99/unlearning"
PR = f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/paper_results"
PROMPTS = f"{REPO}/CAS_SpatialCFG/prompts/coco_10k.txt"
OUT_BASE = f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/outputs/phase_coco_fid"
PYTHON_EBSG = "/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
PYTHON_SAFREE = "/mnt/home3/yhgil99/.conda/envs/safree/bin/python3.10"
BASELINE_SCRIPT = f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/scripts/baseline_runner.py"

# Use sexual EBSG config (most realistic erasure scenario for benign COCO)
SEXUAL_ARGS = f"{PR}/single/sexual/args.json"
STEPS = 50
SEED = 42


def load_prompts():
    return [l.strip() for l in open(PROMPTS) if l.strip()]


def get_chunk(prompts, slot, nslots):
    """Return slice of prompts assigned to this slot."""
    n = len(prompts)
    chunk_size = (n + nslots - 1) // nslots
    start = slot * chunk_size
    end = min(start + chunk_size, n)
    return start, end


def run_ebsg(start, end, outdir, gpu, log):
    a = json.load(open(SEXUAL_ARGS))
    # We'll write a per-slot prompt subset
    sub_prompts = f"{outdir}/_prompts_subset.txt"
    all_prompts = load_prompts()
    with open(sub_prompts, "w") as f:
        for p in all_prompts[start:end]:
            f.write(p + "\n")

    cmd = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", PYTHON_EBSG, "-m", "safegen.generate_family",
           "--prompts", sub_prompts, "--outdir", outdir, "--start_idx", "0",
           "--family_guidance", "--family_config", a["family_config"],
           "--probe_mode", a["probe_mode"], "--how_mode", a["how_mode"],
           "--cas_threshold", str(a["cas_threshold"]),
           "--safety_scale", str(a["safety_scale"]),
           "--attn_threshold", str(a["attn_threshold"]),
           "--img_attn_threshold", str(a["img_attn_threshold"]),
           "--n_img_tokens", str(a.get("n_img_tokens",4)),
           "--steps", str(STEPS), "--seed", str(SEED + start),  # offset seed by start for unique seeds
           "--cfg_scale", str(a.get("cfg_scale",7.5)),
           "--target_concepts", *a["target_concepts"],
           "--anchor_concepts", *a["anchor_concepts"]]
    with open(log,"a") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              cwd=f"{REPO}/SafeGen").returncode

def run_safree(start, end, outdir, gpu, log):
    sub_prompts = f"{outdir}/_prompts_subset.txt"
    all_prompts = load_prompts()
    with open(sub_prompts, "w") as f:
        for p in all_prompts[start:end]:
            f.write(p + "\n")

    cmd = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", PYTHON_SAFREE,
           f"{REPO}/SAFREE/gen_safree_single.py",
           "--txt", sub_prompts, "--save-dir", outdir,
           "--category", "nudity",
           "--re_attn_t=-1,1001",
           "--linear_per_prompt_seed",
           "--num_inference_steps", str(STEPS),
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

def run_baseline(start, end, outdir, gpu, log):
    sub_prompts = f"{outdir}/_prompts_subset.txt"
    all_prompts = load_prompts()
    with open(sub_prompts, "w") as f:
        for p in all_prompts[start:end]:
            f.write(p + "\n")
    cmd = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", PYTHON_EBSG,  # same env
           BASELINE_SCRIPT,
           "--prompts", sub_prompts, "--outdir", outdir,
           "--steps", str(STEPS), "--seed", str(SEED),
           "--cfg_scale", "7.5", "--device", "cuda:0"]
    with open(log,"a") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              cwd=f"{REPO}/SafeGen").returncode


def main():
    gpu = int(sys.argv[1])
    slot = int(sys.argv[2])
    nslots = int(sys.argv[3])
    method = sys.argv[4]
    assert method in ("ebsg","safree","baseline")

    LOGDIR = f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/logs"
    log = f"{LOGDIR}/coco_fid_{method}_g{gpu}_s{slot}.log"
    method_outbase = f"{OUT_BASE}/{method}"
    Path(method_outbase).mkdir(parents=True, exist_ok=True)
    Path(LOGDIR).mkdir(parents=True, exist_ok=True)

    prompts = load_prompts()
    start, end = get_chunk(prompts, slot, nslots)
    n_chunk = end - start
    outdir = f"{method_outbase}/slot{slot:02d}"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    existing = len(list(Path(outdir).glob("*.png")))
    with open(log,"a") as f:
        f.write(f"[start] method={method} GPU={gpu} slot={slot}/{nslots} prompts {start}:{end} ({n_chunk}) existing={existing}\n")
    if existing >= n_chunk:
        with open(log,"a") as f: f.write(f"[skip-done] {n_chunk}/{n_chunk}\n")
        return

    if method == "ebsg":     rc = run_ebsg(start, end, outdir, gpu, log)
    elif method == "safree": rc = run_safree(start, end, outdir, gpu, log)
    else:                    rc = run_baseline(start, end, outdir, gpu, log)
    final = len(list(Path(outdir).glob("*.png")))
    with open(log,"a") as f: f.write(f"[end] method={method} rc={rc} imgs={final}/{n_chunk}\n")


if __name__ == "__main__":
    main()
