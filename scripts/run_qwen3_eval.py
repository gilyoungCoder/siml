#!/usr/bin/env python3
"""
Run Qwen3-VL evaluation on all v17/v18/v19 folders missing qwen3_vl results.
Distributes across GPUs via subprocess.

Usage:
    python scripts/run_qwen3_eval.py --gpus 0,1,2,3,4,5,6,7
    python scripts/run_qwen3_eval.py --gpus 0,1,2,3,4,5,6,7 --dry-run
    python scripts/run_qwen3_eval.py --status
"""
import os
import sys
import glob
import subprocess
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VLM_DIR = REPO_ROOT / "vlm"
VLM_PYTHON = "/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
OUTPUT_BASE = REPO_ROOT / "CAS_SpatialCFG" / "outputs"
VERSIONS = ["v14", "v15", "v17", "v18", "v19"]


def collect_folders():
    folders = []
    for ver in VERSIONS:
        ver_dir = OUTPUT_BASE / ver
        if not ver_dir.exists():
            continue
        for d in sorted(ver_dir.iterdir()):
            if not d.is_dir():
                continue
            if (d / "categories_qwen3_vl_nudity.json").exists():
                continue
            pngs = list(d.glob("*.png"))
            if len(pngs) > 0:
                folders.append(str(d))
    return folders


def check_status():
    for ver in VERSIONS:
        ver_dir = OUTPUT_BASE / ver
        if not ver_dir.exists():
            continue
        total = 0
        done = 0
        for d in ver_dir.iterdir():
            if not d.is_dir():
                continue
            if not list(d.glob("*.png")):
                continue
            total += 1
            if (d / "categories_qwen3_vl_nudity.json").exists():
                done += 1
        print(f"  {ver}: {done}/{total} evaluated")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.status:
        print("=== Qwen3-VL Eval Status ===")
        check_status()
        return

    gpus = args.gpus.split(",")
    num_gpus = len(gpus)

    folders = collect_folders()
    print(f"Total folders to evaluate: {len(folders)}")

    if not folders:
        print("Nothing to evaluate!")
        return

    # Distribute round-robin
    gpu_jobs = {g: [] for g in gpus}
    for i, folder in enumerate(folders):
        g = gpus[i % num_gpus]
        gpu_jobs[g].append(folder)

    for g in gpus:
        print(f"  GPU {g}: {len(gpu_jobs[g])} folders")

    if args.dry_run:
        print("\n[DRY RUN] Would evaluate:")
        for g in gpus:
            for f in gpu_jobs[g][:3]:
                print(f"  GPU {g}: {f}")
            if len(gpu_jobs[g]) > 3:
                print(f"  GPU {g}: ... and {len(gpu_jobs[g])-3} more")
        return

    # Launch per-GPU processes
    procs = []
    log_dir = REPO_ROOT / "scripts" / "logs" / "qwen3_eval"
    log_dir.mkdir(parents=True, exist_ok=True)

    for g in gpus:
        if not gpu_jobs[g]:
            continue

        # Write per-GPU script
        script = log_dir / f"gpu_{g}.sh"
        with open(script, "w") as f:
            f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
            for folder in gpu_jobs[g]:
                f.write(f'echo "[$(date +%H:%M:%S)] EVAL: {folder}"\n')
                f.write(f'cd {VLM_DIR} && CUDA_VISIBLE_DEVICES={g} '
                        f'{VLM_PYTHON} opensource_vlm_i2p_all.py "{folder}" nudity qwen 2>&1 | tail -3\n')
                f.write(f'echo "[$(date +%H:%M:%S)] DONE: {folder}"\n\n')
            f.write(f'echo "GPU {g} ALL COMPLETE ({len(gpu_jobs[g])} folders)"\n')

        os.chmod(script, 0o755)
        log_file = open(log_dir / f"gpu_{g}.log", "w")
        p = subprocess.Popen(["bash", str(script)], stdout=log_file, stderr=subprocess.STDOUT)
        procs.append((g, p, log_file))
        print(f"  GPU {g}: PID {p.pid} -> {log_dir}/gpu_{g}.log")

    print(f"\nAll {len(procs)} GPU processes launched!")
    print(f"Monitor: tail -f {log_dir}/gpu_*.log")
    print(f"Status:  python {__file__} --status")

    # Wait for all
    for g, p, lf in procs:
        p.wait()
        lf.close()
        print(f"GPU {g}: finished (exit code {p.returncode})")

    print("\nAll done!")
    check_status()


if __name__ == "__main__":
    main()
