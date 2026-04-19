#!/usr/bin/env python3
"""
build_job_manifests.py — Generate per-worker job CSV manifests for launch_0420.

Each row: phase,backbone,dataset,config_id,gen_cmd,eval_concept,output_dir,log_path

Run from REPO root:
    python3 scripts/launch_0420/build_job_manifests.py
"""

import csv
import os
import sys
from itertools import product
from pathlib import Path

REPO = "/mnt/home3/yhgil99/unlearning"
PYTHON_GEN = "/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
PYTHON_VLM = "/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
OUT_ROOT = f"{REPO}/CAS_SpatialCFG/outputs/launch_0420"
LOG_ROOT = f"{REPO}/logs/launch_0420"
MANIFEST_DIR = f"{REPO}/scripts/launch_0420/manifests"

# ─── Dataset definitions ────────────────────────────────────────────────────
DATASETS = [
    # name,           prompt_file,                                                            concept,      eval_concept, n_prompts
    ("mja_sexual",    "CAS_SpatialCFG/prompts/mja_sexual.txt",                               "sexual",     "nudity",     100),
    ("mja_violent",   "CAS_SpatialCFG/prompts/mja_violent.txt",                              "violent",    "violence",   100),
    ("mja_disturbing","CAS_SpatialCFG/prompts/mja_disturbing.txt",                           "disturbing", "disturbing", 100),
    ("mja_illegal",   "CAS_SpatialCFG/prompts/mja_illegal.txt",                              "illegal",    "illegal",    100),
    ("rab",           "CAS_SpatialCFG/prompts/nudity-ring-a-bell.csv",                       "sexual",     "nudity",     79),
]

# ─── GPU pools ───────────────────────────────────────────────────────────────
SD14_WORKERS  = [("siml-01", g) for g in range(8)]           # GPU 0-7
SD3_WORKERS   = [("siml-06", g) for g in [4, 5, 6, 7]]       # GPU 4-7
FLUX1_WORKERS = [("siml-08", 4), ("siml-08", 5), ("siml-09", 0)]

# ─── Sweep config ────────────────────────────────────────────────────────────
HOW_MODES = ["anchor_inpaint", "hybrid"]

def get_ours_sweep(backbone):
    """Return list of (safety_scale, attn_threshold, how_mode) tuples."""
    configs = []
    if backbone in ("sd14", "sd3"):
        ss_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        for ds_name, _, _, _, _ in DATASETS:
            if ds_name == "rab":
                thr_list = [0.05, 0.1, 0.2]
            else:
                thr_list = [0.1, 0.2]
            for ss, thr, how in product(ss_list, thr_list, HOW_MODES):
                configs.append((ds_name, ss, thr, how))
    elif backbone == "flux1":
        ss_list = [1.5, 2.0, 2.5, 3.0]
        for ds_name, _, _, _, _ in DATASETS:
            if ds_name == "rab":
                thr_list = [0.05, 0.1, 0.2]
            else:
                thr_list = [0.1]   # FLUX1 MJA: attn_threshold 0.1 only
            for ss, thr, how in product(ss_list, thr_list, HOW_MODES):
                configs.append((ds_name, ss, thr, how))
    return configs


def family_config_path(concept):
    return f"CAS_SpatialCFG/exemplars/concepts_v2/{concept}/clip_grouped.pt"


def make_baseline_cmd(backbone, ds_name, prompt_file, outdir, gpu):
    """Build baseline generation command."""
    device = f"cuda:{gpu}"
    if backbone == "sd14":
        return (
            f"cd {REPO} && CUDA_VISIBLE_DEVICES={gpu} {PYTHON_GEN} "
            f"CAS_SpatialCFG/generate_baseline.py "
            f"--prompts {prompt_file} --outdir {outdir} --steps 50"
        )
    elif backbone == "sd3":
        return (
            f"cd {REPO} && CUDA_VISIBLE_DEVICES={gpu} {PYTHON_GEN} "
            f"scripts/sd3/generate_sd3_baseline.py "
            f"--prompts {prompt_file} --outdir {outdir} "
            f"--device {device} --no_cpu_offload"
        )
    elif backbone == "flux1":
        extra = ""
        # siml-09 H100: no cpu_offload needed; siml-08: keep default (cpu_offload on)
        # generate_flux1_v1.py default height/width is 512, need to pass 1024
        return (
            f"cd {REPO} && CUDA_VISIBLE_DEVICES={gpu} {PYTHON_GEN} "
            f"CAS_SpatialCFG/generate_flux1_v1.py "
            f"--prompts {prompt_file} --outdir {outdir} "
            f"--no_safety --height 1024 --width 1024 --device {device}"
        )


def make_safree_cmd(backbone, ds_name, prompt_file, concept, outdir, gpu):
    """Build SAFREE generation command."""
    device = f"cuda:{gpu}"
    if backbone == "sd14":
        # SAFREE uses --data, --save-dir, --category, --safree, --svf flags
        return (
            f"cd {REPO} && CUDA_VISIBLE_DEVICES={gpu} {PYTHON_GEN} "
            f"SAFREE/generate_safree.py "
            f"--data {prompt_file} --save-dir {outdir} "
            f"--category {concept} --safree --svf --device {device}"
        )
    elif backbone == "sd3":
        return (
            f"cd {REPO} && CUDA_VISIBLE_DEVICES={gpu} {PYTHON_GEN} "
            f"scripts/sd3/generate_sd3_safree.py "
            f"--prompts {prompt_file} --outdir {outdir} "
            f"--concept {concept} --device {device} --no_cpu_offload"
        )
    elif backbone == "flux1":
        return (
            f"cd {REPO} && CUDA_VISIBLE_DEVICES={gpu} {PYTHON_GEN} "
            f"CAS_SpatialCFG/generate_flux1_safree.py "
            f"--prompts {prompt_file} --outdir {outdir} "
            f"--device {device}"
        )


def make_ours_cmd(backbone, ds_name, prompt_file, concept, outdir,
                  ss, thr, how, host, gpu):
    """Build 'ours' generation command."""
    device = f"cuda:{gpu}"
    fc = family_config_path(concept)
    common = (
        f"--probe_mode both --cas_threshold 0.6 "
        f"--safety_scale {ss} --attn_threshold {thr} "
        f"--how_mode {how} --family_guidance --family_config {fc}"
    )
    if backbone == "sd14":
        return (
            f"cd {REPO} && CUDA_VISIBLE_DEVICES={gpu} {PYTHON_GEN} "
            f"CAS_SpatialCFG/generate_v27.py "
            f"--prompts {prompt_file} --outdir {outdir} {common}"
        )
    elif backbone == "sd3":
        return (
            f"cd {REPO} && CUDA_VISIBLE_DEVICES={gpu} {PYTHON_GEN} "
            f"scripts/sd3/generate_sd3_safegen.py "
            f"--prompts {prompt_file} --outdir {outdir} "
            f"--device {device} --no_cpu_offload {common}"
        )
    elif backbone == "flux1":
        # FLUX1 has no img_attn_threshold arg
        extra = ""
        if host == "siml-09":
            extra = " --no_cpu_offload"  # H100 has enough VRAM
        return (
            f"cd {REPO} && CUDA_VISIBLE_DEVICES={gpu} {PYTHON_GEN} "
            f"CAS_SpatialCFG/generate_flux1_v1.py "
            f"--prompts {prompt_file} --outdir {outdir} "
            f"--height 1024 --width 1024 --device {device}{extra} {common}"
        )


def ours_config_id(ss, thr, how):
    ss_str = f"{ss:.1f}".replace(".", "")  # 0.5 → "05", 1.0 → "10"
    thr_str = f"{thr:.2f}".lstrip("0") or "0"  # 0.05 → ".05", 0.1 → ".1"
    return f"cas0.6_ss{ss}_thr{thr}_{how}_both"


def build_jobs_for_backbone(backbone, workers):
    """Build all jobs for a backbone, returning list of (host, gpu, job_row)."""
    ds_map = {ds[0]: ds for ds in DATASETS}
    jobs_by_worker = {(h, g): [] for h, g in workers}
    all_jobs = []  # (phase_order, ds_name, config_id, gen_cmd, eval_concept, output_dir)

    # ── Phase 1: baselines (5 datasets) ──
    for ds_name, prompt_file, concept, eval_concept, n_prompts in DATASETS:
        outdir = f"{OUT_ROOT}/baseline_{backbone}/{ds_name}"
        host, gpu = workers[len(all_jobs) % len(workers)]
        gen_cmd = make_baseline_cmd(backbone, ds_name, prompt_file, outdir, gpu)
        all_jobs.append(("baseline", ds_name, "baseline", gen_cmd, eval_concept, outdir, host, gpu))

    # ── Phase 2: SAFREE (5 datasets) ──
    for ds_name, prompt_file, concept, eval_concept, n_prompts in DATASETS:
        outdir = f"{OUT_ROOT}/safree_{backbone}/{ds_name}"
        host, gpu = workers[len(all_jobs) % len(workers)]
        gen_cmd = make_safree_cmd(backbone, ds_name, prompt_file, concept, outdir, gpu)
        all_jobs.append(("safree", ds_name, "safree", gen_cmd, eval_concept, outdir, host, gpu))

    # ── Phase 3: Ours sweep ──
    sweep = get_ours_sweep(backbone)
    for ds_name, ss, thr, how in sweep:
        _, prompt_file, concept, eval_concept, n_prompts = ds_map[ds_name]
        config_id = ours_config_id(ss, thr, how)
        outdir = f"{OUT_ROOT}/ours_{backbone}/{ds_name}/{config_id}"
        host, gpu = workers[len(all_jobs) % len(workers)]
        gen_cmd = make_ours_cmd(backbone, ds_name, prompt_file, concept, outdir,
                                ss, thr, how, host, gpu)
        all_jobs.append(("ours", ds_name, config_id, gen_cmd, eval_concept, outdir, host, gpu))

    return all_jobs


def assign_to_workers(all_jobs, workers):
    """Round-robin assign jobs to workers."""
    worker_jobs = {(h, g): [] for h, g in workers}
    for i, job in enumerate(all_jobs):
        phase, ds_name, config_id, gen_cmd, eval_concept, outdir, _, _ = job
        h, g = workers[i % len(workers)]
        worker_jobs[(h, g)].append((phase, ds_name, config_id, gen_cmd, eval_concept, outdir))
    return worker_jobs


def write_manifests(backbone, workers):
    all_jobs = build_jobs_for_backbone(backbone, workers)
    worker_jobs = assign_to_workers(all_jobs, workers)

    for (host, gpu), jobs in worker_jobs.items():
        fname = f"{MANIFEST_DIR}/worker_{host}_g{gpu}.csv"
        with open(fname, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["phase", "backbone", "dataset", "config_id",
                             "gen_cmd", "eval_concept", "output_dir"])
            for phase, ds_name, config_id, gen_cmd, eval_concept, outdir in jobs:
                writer.writerow([phase, backbone, ds_name, config_id,
                                 gen_cmd, eval_concept, outdir])
        print(f"  {fname}: {len(jobs)} jobs")
    return sum(len(v) for v in worker_jobs.values())


def main():
    os.makedirs(MANIFEST_DIR, exist_ok=True)

    print("=== Building SD1.4 manifests (siml-01 GPU 0-7) ===")
    n14 = write_manifests("sd14", SD14_WORKERS)

    print("=== Building SD3 manifests (siml-06 GPU 4-7) ===")
    n3 = write_manifests("sd3", SD3_WORKERS)

    print("=== Building FLUX1 manifests (siml-08:4,5 + siml-09:0) ===")
    nf = write_manifests("flux1", FLUX1_WORKERS)

    total = n14 + n3 + nf
    print(f"\nTotal jobs: {total} ({n14} SD1.4, {n3} SD3, {nf} FLUX1)")
    print(f"Expected ~342 gen jobs. Each gen job has 2 eval calls = ~{total*2} eval calls.")


if __name__ == "__main__":
    main()
