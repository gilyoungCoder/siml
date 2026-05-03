#!/usr/bin/env python3
"""Scale-robustness v2: 4 SLD variants × sexual × CFG-scale gs sweep, clamp removed.

For each (variant, scale): generate 60 imgs at NFE=50, then eval.
- All 4 SLD variants (Max/Medium/Strong/Weak) keep their default threshold/warmup/
  momentum, only sld_guidance_scale is overridden across the sweep.
- Clamp removed (SLD_CLAMP_MAX=1e6) so gs really controls magnitude.
- gs sweep [5, 10, 20, 50, 100, 200, 500] — same range as EBSG safety_scale,
  comparable to CFG=7.5 magnitude.
- Concept: sexual only.

Runs as N-slot dispatch (default 5: g1..g5). Idempotent.
"""
import os, sys, subprocess
from pathlib import Path

REPO = "/mnt/home3/yhgil99/unlearning"
LAUNCH = f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep"
OUT_BASE = f"{LAUNCH}/outputs/phase_scale_robustness_v2"
LOGDIR = f"{LAUNCH}/logs/scale_robustness_v2"
PY_EBSG = "/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
PY_VLM  = "/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
SLD_RUNNER = f"{LAUNCH}/scripts/sld_runner.py"
EVAL_SCRIPT = "/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py"
EVAL_CWD = "/mnt/home3/yhgil99/unlearning/vlm"
TXT_BASE = f"{REPO}/CAS_SpatialCFG/prompts/i2p_q16_top60"

NFE = 50
CONCEPT = "sexual"
EVAL_RUBRIC = "nudity"
SCALES = [5, 10, 20, 50, 100, 200, 500]
VARIANTS = ["Max", "Medium", "Strong", "Weak"]


def png_count(d):
    return len(list(Path(d).glob("*.png")))


def needs_eval(outdir):
    f = Path(outdir) / f"results_qwen3_vl_{EVAL_RUBRIC}_v5.txt"
    return (not f.exists()) or f.stat().st_size < 50


def run_sld(variant, scale, outdir, gpu, log):
    txt = f"{TXT_BASE}/{CONCEPT}_q16_top60.txt"
    cmd = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", "PYTHONNOUSERSITE=1",
           "SLD_CLAMP_MAX=1000000", PY_EBSG, SLD_RUNNER,
           "--prompts", txt, "--outdir", outdir,
           "--variant", variant,
           "--steps", str(NFE), "--seed", "42", "--cfg_scale", "7.5",
           "--sld_guidance_scale", str(scale)]
    with open(log, "a") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def run_eval(outdir, gpu, log):
    cmd = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", PY_VLM, EVAL_SCRIPT,
           outdir, EVAL_RUBRIC, "qwen"]
    with open(log, "a") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=EVAL_CWD).returncode


def main():
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    slot = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nslots = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    Path(OUT_BASE).mkdir(parents=True, exist_ok=True)
    Path(LOGDIR).mkdir(parents=True, exist_ok=True)
    log = f"{LOGDIR}/g{gpu}_s{slot}.log"
    with open(log, "a") as f:
        f.write(f"[start] GPU={gpu} slot={slot}/{nslots}\n")

    cells = [(v, s) for v in VARIANTS for s in SCALES]
    for i, (v, s) in enumerate(cells):
        if (i % nslots) != slot: continue
        cell = f"sld_{v.lower()}_{CONCEPT}_scale{s}"
        outdir = f"{OUT_BASE}/{cell}"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        n = png_count(outdir)
        if n >= 60 and not needs_eval(outdir):
            with open(log, "a") as f: f.write(f"[skip-done] {cell} n={n}\n")
            continue
        if n < 60:
            with open(log, "a") as f: f.write(f"[gen] {cell}\n")
            try:
                rc = run_sld(v, s, outdir, gpu, log)
                with open(log, "a") as f:
                    f.write(f"[gen-done] {cell} rc={rc} imgs={png_count(outdir)}\n")
            except Exception as e:
                with open(log, "a") as f: f.write(f"[gen-exc] {cell} {e}\n")
                continue
        if needs_eval(outdir) and png_count(outdir) > 0:
            with open(log, "a") as f: f.write(f"[eval] {cell}\n")
            try:
                rc = run_eval(outdir, gpu, log)
                with open(log, "a") as f: f.write(f"[eval-done] {cell} rc={rc}\n")
            except Exception as e:
                with open(log, "a") as f: f.write(f"[eval-exc] {cell} {e}\n")

    with open(log, "a") as f: f.write(f"[end] GPU={gpu} slot={slot}\n")


if __name__ == "__main__":
    main()
