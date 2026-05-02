#!/usr/bin/env python3
"""Scale-robustness sweep orchestrator.

For each (method, concept, scale): generate 60 imgs at NFE=50, then evaluate.
Methods:
  - sld:  sweep sld_guidance_scale at SLD-Max-style params (warmup=0, threshold=1.0)
  - ebsg: sweep safety_scale at the per-concept best (cas/theta_text/theta_img)

Runs as 6-slot parallel dispatch on siml-05 g2..g7 (NEVER g0/g1).
Idempotent: skips cell if outdir already has 60 PNGs.
"""
import json, os, sys, subprocess
from pathlib import Path

REPO = "/mnt/home3/yhgil99/unlearning"
LAUNCH = f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep"
PR = f"{LAUNCH}/paper_results"
OUT_BASE = f"{LAUNCH}/outputs/phase_scale_robustness"
LOGDIR = f"{LAUNCH}/logs/scale_robustness"

PY_EBSG = "/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
PY_VLM  = "/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
SLD_RUNNER = f"{LAUNCH}/scripts/sld_runner.py"
EVAL_SCRIPT = "/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py"
EVAL_CWD    = "/mnt/home3/yhgil99/unlearning/vlm"

NFE = 50
CONCEPTS = ["sexual", "violence"]
SCALES_SLD  = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
SCALES_EBSG = [5, 10, 20, 50, 100, 200, 500]

ARGS_DIR = {"sexual": "sexual", "violence": "violence"}
EVAL_NAME = {"sexual": "nudity", "violence": "violence"}

# Per-concept EBSG best (other params fixed; only safety_scale varies in sweep)
OURS_FIXED = {
    "sexual":   dict(cas_threshold=0.5, attn_threshold=0.10, img_attn_threshold=0.30),
    "violence": dict(cas_threshold=0.4, attn_threshold=0.30, img_attn_threshold=0.10),
}

TXT_BASE = f"{REPO}/CAS_SpatialCFG/prompts/i2p_q16_top60"


def png_count(d):
    p = Path(d)
    return len(list(p.glob("*.png")))


def needs_eval(outdir, concept):
    f = Path(outdir) / f"results_qwen3_vl_{EVAL_NAME[concept]}_v5.txt"
    return (not f.exists()) or f.stat().st_size < 50


def run_sld(concept, scale, outdir, gpu, log):
    txt = f"{TXT_BASE}/{concept}_q16_top60.txt"
    cmd = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", "PYTHONNOUSERSITE=1", PY_EBSG,
           SLD_RUNNER,
           "--prompts", txt, "--outdir", outdir,
           "--variant", "Max",
           "--steps", str(NFE), "--seed", "42", "--cfg_scale", "7.5",
           "--sld_guidance_scale", str(scale)]
    with open(log, "a") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def run_ebsg(concept, scale, outdir, gpu, log):
    txt = f"{TXT_BASE}/{concept}_q16_top60.txt"
    args_path = f"{PR}/single/{ARGS_DIR[concept]}/args.json"
    a = json.load(open(args_path))
    fx = OURS_FIXED[concept]
    cmd = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", "PYTHONNOUSERSITE=1",
           PY_EBSG, "-m", "safegen.generate_family",
           "--prompts", txt, "--outdir", outdir,
           "--family_guidance", "--family_config", a["family_config"],
           "--probe_mode", a.get("probe_mode", "both"),
           "--probe_fusion", a.get("probe_fusion", "union"),
           "--how_mode", a.get("how_mode", "hybrid"),
           "--cas_threshold", str(fx["cas_threshold"]),
           "--safety_scale", str(scale),
           "--attn_threshold", str(fx["attn_threshold"]),
           "--img_attn_threshold", str(fx["img_attn_threshold"]),
           "--n_img_tokens", str(a.get("n_img_tokens", 4)),
           "--steps", str(NFE), "--seed", "42", "--cfg_scale", "7.5",
           "--target_concepts", *a["target_concepts"],
           "--target_words", *a["target_words"],
           "--anchor_concepts", *a.get("anchor_concepts", ["safe_scene"])]
    with open(log, "a") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              cwd=f"{REPO}/SafeGen").returncode


def run_eval(concept, outdir, gpu, log):
    cmd = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", PY_VLM, EVAL_SCRIPT,
           outdir, EVAL_NAME[concept], "qwen"]
    with open(log, "a") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=EVAL_CWD).returncode


def make_cells():
    cells = []
    for c in CONCEPTS:
        for s in SCALES_SLD:
            cells.append(("sld", c, s))
        for s in SCALES_EBSG:
            cells.append(("ebsg", c, s))
    return cells


def main():
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    slot = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nslots = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    Path(OUT_BASE).mkdir(parents=True, exist_ok=True)
    Path(LOGDIR).mkdir(parents=True, exist_ok=True)
    log = f"{LOGDIR}/g{gpu}_s{slot}.log"
    with open(log, "a") as f:
        f.write(f"[start] GPU={gpu} slot={slot}/{nslots}\n")

    cells = make_cells()
    for i, (m, c, s) in enumerate(cells):
        if (i % nslots) != slot: continue
        cell = f"{m}_{c}_scale{s}"
        outdir = f"{OUT_BASE}/{cell}"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        n = png_count(outdir)
        if n >= 60 and not needs_eval(outdir, c):
            with open(log, "a") as f:
                f.write(f"[skip-done] {cell} n={n}\n")
            continue
        if n < 60:
            with open(log, "a") as f:
                f.write(f"[gen] {cell}\n")
            try:
                fn = run_sld if m == "sld" else run_ebsg
                rc = fn(c, s, outdir, gpu, log)
                n2 = png_count(outdir)
                with open(log, "a") as f:
                    f.write(f"[gen-done] {cell} rc={rc} imgs={n2}\n")
            except Exception as e:
                with open(log, "a") as f:
                    f.write(f"[gen-exc] {cell} {e}\n")
                continue
        if needs_eval(outdir, c) and png_count(outdir) > 0:
            with open(log, "a") as f:
                f.write(f"[eval] {cell}\n")
            try:
                rc = run_eval(c, outdir, gpu, log)
                with open(log, "a") as f:
                    f.write(f"[eval-done] {cell} rc={rc}\n")
            except Exception as e:
                with open(log, "a") as f:
                    f.write(f"[eval-exc] {cell} {e}\n")

    with open(log, "a") as f:
        f.write(f"[end] GPU={gpu} slot={slot}\n")


if __name__ == "__main__":
    main()
