#!/usr/bin/env python3
"""NFE ablation: re-run best configs with steps ∈ {1, 5, 10, 25}.
Tests whether EBSG safety holds at lower denoising step counts (Giung's hypothesis).
Sequential on siml-09 GPU 0."""
import json, os, sys, subprocess
from pathlib import Path

REPO = "/mnt/home3/yhgil99/unlearning"
OUT_BASE = f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/outputs/phase_nfe_ablation"
PYTHON = f"/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"

# Pick a few representative best configs
CONCEPTS = [
    ("violence",   f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/single/violence/args.json",   "single"),
    ("shocking",   f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/single/shocking/args.json",   "single"),
    ("self-harm",  f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/single/self-harm/args.json",  "single"),
]
STEPS_LIST = [1, 5, 10, 25]   # 50 already in main results

def build_cmd(args, mode, gpu, outdir, steps_override):
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
               "--steps", str(steps_override),  # OVERRIDDEN
               "--seed", str(args.get("seed", 42)),
               "--cfg_scale", str(args.get("cfg_scale", 7.5)),
               "--target_concepts", *args["target_concepts"],
               "--anchor_concepts", *args["anchor_concepts"]]
        return cmd
    raise NotImplementedError("multi NFE ablation not needed")


def main():
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    Path(OUT_BASE).mkdir(parents=True, exist_ok=True)
    LOGDIR = f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/logs"
    log_path = f"{LOGDIR}/nfe_ablation_g{gpu}.log"

    with open(log_path, "a") as logf: logf.write(f"[start] NFE ablation GPU={gpu}\n")
    print(f"[NFE] worker GPU={gpu} log={log_path}")

    for concept, args_path, mode in CONCEPTS:
        if not os.path.isfile(args_path):
            print(f"  skip {concept}: no args at {args_path}"); continue
        args = json.load(open(args_path))
        prompt_count = len([l for l in open(args["prompts"]) if l.strip()])

        for steps in STEPS_LIST:
            cell = f"{concept}_steps{steps}"
            outdir = f"{OUT_BASE}/{cell}"
            Path(outdir).mkdir(parents=True, exist_ok=True)
            existing = len(list(Path(outdir).glob("*.png")))
            if existing >= prompt_count:
                with open(log_path,"a") as logf: logf.write(f"[skip-done] {cell} ({existing}/{prompt_count})\n")
                continue
            cmd = build_cmd(args, mode, gpu, outdir, steps)
            with open(log_path,"a") as logf: logf.write(f"[run] {cell} steps={steps}\n")
            try:
                with open(log_path,"a") as logf:
                    rc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=f"{REPO}/SafeGen").returncode
                final = len(list(Path(outdir).glob("*.png")))
                with open(log_path,"a") as logf: logf.write(f"[done] {cell} rc={rc} imgs={final}\n")
            except Exception as e:
                with open(log_path,"a") as logf: logf.write(f"[exc] {cell} {e}\n")

    with open(log_path,"a") as logf: logf.write(f"[end] NFE ablation GPU={gpu}\n")


if __name__ == "__main__":
    main()
