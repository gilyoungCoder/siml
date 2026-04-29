#!/usr/bin/env python3
"""Reproducibility runner: read each best cell's args.json, re-run with same args + new outdir.
Designed to run on siml-04 (8 GPU) for ~14-23 cells."""
import json, os, sys, subprocess
from pathlib import Path

REPO = "/mnt/home3/yhgil99/unlearning"
OUT_BASE = f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/outputs/phase_repro"
PYTHON = f"/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"

# (cell_name, src_args_path, mode: single/multi)
CELLS = [
    # === Single i2p ===
    ("violence",   f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/single/violence/args.json",   "single"),
    ("self-harm",  f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/single/self-harm/args.json",  "single"),
    ("shocking",   f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/single/shocking/args.json",   "single"),
    ("illegal",    f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/single/illegal/args.json",    "single"),
    ("harassment", f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/single/harassment/args.json", "single"),
    ("hate",       f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/single/hate/args.json",       "single"),
    # === Multi ===
    ("1c_sexual",     f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/multi/1c_sexual/sexual/args.json",     "multi"),
    ("2c_sexvio_v3",  f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/multi/2c_sexvio_v3_best/sexual/args.json", "multi"),
    ("3c_sexvioshock",f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/multi/3c_sexvioshock/sexual/args.json",   "multi"),
    ("7c_all",        f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/multi/7c_all/sexual/args.json",            "multi"),
    # === Nudity datasets (single) — from phase_paper_best ===
    ("nudity_mma",  f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/outputs/phase_paper_best/nudity_mma_hybrid/args.json",  "single"),
    ("nudity_p4dn", f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/outputs/phase_paper_best/nudity_p4dn_hybrid/args.json", "single"),
    ("nudity_rab",  f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/outputs/phase_paper_best/nudity_rab_hybrid/args.json",  "single"),
    ("nudity_ud",   f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/outputs/phase_paper_best/nudity_ud_hybrid/args.json",   "single"),
]

# For multi: which prompts file to use per eval — use first eval (sexual) per setup
# Multi cells have all eval prompts in their args. We re-run only one eval per setup for repro.

def build_cmd(args, mode, gpu, outdir):
    """Build subprocess command from args dict."""
    if mode == "single":
        cmd = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", PYTHON, "-m", "safegen.generate_family",
               "--prompts", args["prompts"], "--outdir", outdir,
               "--family_guidance", "--family_config", args["family_config"],
               "--probe_mode", args["probe_mode"], "--how_mode", args["how_mode"],
               "--cas_threshold", str(args["cas_threshold"]),
               "--safety_scale", str(args["safety_scale"]),
               "--attn_threshold", str(args["attn_threshold"]),
               "--img_attn_threshold", str(args["img_attn_threshold"]),
               "--n_img_tokens", str(args.get("n_img_tokens", 4)),
               "--steps", str(args.get("steps", 50)), "--seed", str(args.get("seed", 42)),
               "--cfg_scale", str(args.get("cfg_scale", 7.5)),
               "--target_concepts", *args["target_concepts"],
               "--anchor_concepts", *args["anchor_concepts"]]
    else:  # multi
        cmd = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", PYTHON, "-m", "safegen.generate_family_multi",
               "--prompts", args["prompts"], "--outdir", outdir,
               "--family_guidance",
               "--probe_mode", args["probe_mode"],
               "--steps", str(args.get("steps", 50)), "--seed", str(args.get("seed", 42)),
               "--cfg_scale", str(args.get("cfg_scale", 7.5)),
               "--target_concepts", *args["target_concepts"],
               "--anchor_concepts", *args["anchor_concepts"]]
        # List args
        cmd += ["--family_config"] + list(args["family_config"])
        cas = args["cas_threshold"];   cmd += ["--cas_threshold"]   + [str(x) for x in (cas if isinstance(cas,list) else [cas])]
        ss  = args["safety_scale"];    cmd += ["--safety_scale"]    + [str(x) for x in (ss  if isinstance(ss,list)  else [ss])]
        at  = args["attn_threshold"];  cmd += ["--attn_threshold"]  + [str(x) for x in (at  if isinstance(at,list)  else [at])]
        ia  = args["img_attn_threshold"]; cmd += ["--img_attn_threshold"] + [str(x) for x in (ia if isinstance(ia,list) else [ia])]
        hm  = args["how_mode"]
        if isinstance(hm, str): hm=[hm]
        cmd += ["--how_mode"] + hm
        if args.get("n_img_tokens"):
            cmd += ["--n_img_tokens", str(args["n_img_tokens"])]
    return cmd


def main():
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nslots = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    slot = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # which slot is this worker

    Path(OUT_BASE).mkdir(parents=True, exist_ok=True)
    LOGDIR = f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/logs"
    Path(LOGDIR).mkdir(parents=True, exist_ok=True)
    log_path = f"{LOGDIR}/repro_g{gpu}_s{slot}.log"

    print(f"[repro] worker GPU={gpu} slot={slot} nslots={nslots} log={log_path}")
    with open(log_path, "a") as logf:
        logf.write(f"[start] GPU={gpu} slot={slot}\n")

    for i, (name, args_path, mode) in enumerate(CELLS):
        if (i % nslots) != slot:
            continue
        if not os.path.isfile(args_path):
            with open(log_path,"a") as logf: logf.write(f"[skip-noargs] {name} {args_path}\n")
            continue
        args = json.load(open(args_path))
        outdir = f"{OUT_BASE}/{name}"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        # Skip if already 60+ pngs
        prompt_count = len([l for l in open(args["prompts"]) if l.strip()])
        existing = len(list(Path(outdir).glob("*.png")))
        if existing >= prompt_count:
            with open(log_path,"a") as logf: logf.write(f"[skip-done] {name} ({existing}/{prompt_count})\n")
            continue

        cmd = build_cmd(args, mode, gpu, outdir)
        with open(log_path,"a") as logf:
            logf.write(f"[run] {name} mode={mode} prompts={args['prompts'].split('/')[-1]} expect={prompt_count}\n")
            logf.write(f"  cmd: {' '.join(cmd[:6])} ...\n")
        try:
            with open(log_path,"a") as logf:
                rc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=f"{REPO}/SafeGen").returncode
            final = len(list(Path(outdir).glob("*.png")))
            with open(log_path,"a") as logf:
                logf.write(f"[done] {name} rc={rc} imgs={final}\n")
        except Exception as e:
            with open(log_path,"a") as logf:
                logf.write(f"[exc] {name} {e}\n")

    with open(log_path,"a") as logf: logf.write(f"[end] worker GPU={gpu} slot={slot}\n")


if __name__ == "__main__":
    main()
