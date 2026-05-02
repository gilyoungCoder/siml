#!/usr/bin/env python3
"""
build_job_manifests.py — Generate per-worker job CSV manifests for launch_0420.

Each row: phase,backbone,dataset,config_id,gen_cmd,eval_concept,output_dir

Run from REPO root:
    python3 scripts/launch_0420/build_job_manifests.py

Deviations from original spec:
  - SD1.4 ours: uses SafeGen/safegen/generate_family.py (python -m safegen.generate_family
    from SafeGen dir) — generate_v27.py does NOT support --family_guidance
  - SD3 ours (safegen): OOMs on siml-06 A6000 49GB; routed to siml-09 GPU 0
  - SD3 baseline/safree: stay on siml-06 GPU 4-7 (work fine)
  - siml-09 GPU 0 runs: SD3 ours (132 jobs) + FLUX1 (66 jobs) sequentially
"""

import csv
import os
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
    # name,            prompt_file,                                               concept,      eval_concept, n_prompts
    ("mja_sexual",    "CAS_SpatialCFG/prompts/mja_sexual.txt",                   "sexual",     "nudity",     100),
    ("mja_violent",   "CAS_SpatialCFG/prompts/mja_violent.txt",                  "violent",    "violence",   100),
    ("mja_disturbing","CAS_SpatialCFG/prompts/mja_disturbing.txt",               "disturbing", "disturbing", 100),
    ("mja_illegal",   "CAS_SpatialCFG/prompts/mja_illegal.txt",                  "illegal",    "illegal",    100),
    ("rab",           "CAS_SpatialCFG/prompts/nudity-ring-a-bell.csv",           "sexual",     "nudity",     79),
]

# ─── GPU pools ───────────────────────────────────────────────────────────────
SD14_WORKERS      = [("siml-01", g) for g in range(8)]         # GPU 0-7
SD3_BL_WORKERS    = [("siml-06", g) for g in [4, 5, 6, 7]]    # baseline+safree only
SD3_OURS_WORKERS  = [("siml-09", 0)]                            # ours only (OOM on A6000)
FLUX1_WORKERS     = [("siml-08", 4), ("siml-08", 5), ("siml-09", 0)]

# ─── Sweep config ────────────────────────────────────────────────────────────
HOW_MODES = ["anchor_inpaint", "hybrid"]


def get_ours_sweep(backbone):
    """Return list of (ds_name, safety_scale, attn_threshold, how_mode) tuples."""
    configs = []
    if backbone in ("sd14", "sd3"):
        ss_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        for ds_name, _, _, _, _ in DATASETS:
            thr_list = [0.05, 0.1, 0.2] if ds_name == "rab" else [0.1, 0.2]
            for ss, thr, how in product(ss_list, thr_list, HOW_MODES):
                configs.append((ds_name, ss, thr, how))
    elif backbone == "flux1":
        ss_list = [1.5, 2.0, 2.5, 3.0]
        for ds_name, _, _, _, _ in DATASETS:
            thr_list = [0.05, 0.1, 0.2] if ds_name == "rab" else [0.1]
            for ss, thr, how in product(ss_list, thr_list, HOW_MODES):
                configs.append((ds_name, ss, thr, how))
    return configs


def family_config_path(concept):
    return f"{REPO}/CAS_SpatialCFG/exemplars/concepts_v2/{concept}/clip_grouped.pt"


def make_baseline_cmd(backbone, prompt_file, outdir, gpu):
    prompt_abs = f"{REPO}/{prompt_file}"
    # With CUDA_VISIBLE_DEVICES=<gpu>, device inside script is always cuda:0
    if backbone == "sd14":
        return (
            f"cd {REPO} && CUDA_VISIBLE_DEVICES={gpu} {PYTHON_GEN} "
            f"CAS_SpatialCFG/generate_baseline.py "
            f"--prompts {prompt_abs} --outdir {outdir} --steps 50"
        )
    elif backbone == "sd3":
        return (
            f"cd {REPO} && CUDA_VISIBLE_DEVICES={gpu} {PYTHON_GEN} "
            f"scripts/sd3/generate_sd3_baseline.py "
            f"--prompts {prompt_abs} --outdir {outdir} "
            f"--device cuda:0 --no_cpu_offload"
        )
    elif backbone == "flux1":
        return (
            f"cd {REPO} && CUDA_VISIBLE_DEVICES={gpu} {PYTHON_GEN} "
            f"CAS_SpatialCFG/generate_flux1_v1.py "
            f"--prompts {prompt_abs} --outdir {outdir} "
            f"--no_safety --height 1024 --width 1024 --device cuda:0"
        )


def make_safree_cmd(backbone, prompt_file, concept, outdir, gpu):
    prompt_abs = f"{REPO}/{prompt_file}"
    if backbone == "sd14":
        # SAFREE SD1.4 uses --data, --save-dir, --category, --safree, --svf
        return (
            f"cd {REPO} && CUDA_VISIBLE_DEVICES={gpu} {PYTHON_GEN} "
            f"SAFREE/generate_safree.py "
            f"--data {prompt_abs} --save-dir {outdir} "
            f"--category {concept} --safree --svf --device cuda:0"
        )
    elif backbone == "sd3":
        return (
            f"cd {REPO} && CUDA_VISIBLE_DEVICES={gpu} {PYTHON_GEN} "
            f"scripts/sd3/generate_sd3_safree.py "
            f"--prompts {prompt_abs} --outdir {outdir} "
            f"--concept {concept} --device cuda:0 --no_cpu_offload"
        )
    elif backbone == "flux1":
        return (
            f"cd {REPO} && CUDA_VISIBLE_DEVICES={gpu} {PYTHON_GEN} "
            f"CAS_SpatialCFG/generate_flux1_safree.py "
            f"--prompts {prompt_abs} --outdir {outdir} --device cuda:0"
        )


def make_ours_cmd(backbone, prompt_file, concept, outdir, ss, thr, how, host, gpu):
    prompt_abs = f"{REPO}/{prompt_file}"
    fc = family_config_path(concept)
    # SD1.4 and SD3 support img_attn_threshold; FLUX1 does not
    common_with_img = (
        f"--probe_mode both --cas_threshold 0.6 "
        f"--safety_scale {ss} --attn_threshold {thr} --img_attn_threshold {thr} "
        f"--how_mode {how} --family_guidance --family_config {fc}"
    )
    common_flux1 = (
        f"--probe_mode both --cas_threshold 0.6 "
        f"--safety_scale {ss} --attn_threshold {thr} "
        f"--how_mode {how} --family_guidance --family_config {fc}"
    )
    if backbone == "sd14":
        # Run as module from SafeGen dir (needs safegen package on sys.path)
        return (
            f"cd {REPO}/SafeGen && CUDA_VISIBLE_DEVICES={gpu} {PYTHON_GEN} "
            f"-m safegen.generate_family "
            f"--prompts {prompt_abs} --outdir {outdir} {common_with_img}"
        )
    elif backbone == "sd3":
        # SD3 safegen: no --device arg; uses CUDA_VISIBLE_DEVICES -> cuda internally
        return (
            f"cd {REPO} && CUDA_VISIBLE_DEVICES={gpu} {PYTHON_GEN} "
            f"scripts/sd3/generate_sd3_safegen.py "
            f"--prompts {prompt_abs} --outdir {outdir} "
            f"--no_cpu_offload {common_with_img}"
        )
    elif backbone == "flux1":
        extra = " --no_cpu_offload" if host == "siml-09" else ""
        return (
            f"cd {REPO} && CUDA_VISIBLE_DEVICES={gpu} {PYTHON_GEN} "
            f"CAS_SpatialCFG/generate_flux1_v1.py "
            f"--prompts {prompt_abs} --outdir {outdir} "
            f"--height 1024 --width 1024 --device cuda:0{extra} {common_flux1}"
        )


def ours_config_id(ss, thr, how):
    return f"cas0.6_ss{ss}_thr{thr}_{how}_both"


def build_sd14_jobs():
    """SD1.4: all phases on siml-01 GPU 0-7."""
    ds_map = {ds[0]: ds for ds in DATASETS}
    all_jobs = []

    # Phase 1: baseline
    for ds_name, prompt_file, concept, eval_concept, _ in DATASETS:
        outdir = f"{OUT_ROOT}/baseline_sd14/{ds_name}"
        all_jobs.append(("baseline", "sd14", ds_name, "baseline",
                         None, eval_concept, outdir))  # gen_cmd filled below

    # Phase 2: safree
    for ds_name, prompt_file, concept, eval_concept, _ in DATASETS:
        outdir = f"{OUT_ROOT}/safree_sd14/{ds_name}"
        all_jobs.append(("safree", "sd14", ds_name, "safree",
                         None, eval_concept, outdir))

    # Phase 3: ours sweep
    for ds_name, ss, thr, how in get_ours_sweep("sd14"):
        _, prompt_file, concept, eval_concept, _ = ds_map[ds_name]
        config_id = ours_config_id(ss, thr, how)
        outdir = f"{OUT_ROOT}/ours_sd14/{ds_name}/{config_id}"
        all_jobs.append(("ours", "sd14", ds_name, config_id,
                         None, eval_concept, outdir))

    # Assign to workers round-robin and fill gen_cmd
    workers = SD14_WORKERS
    worker_jobs = {w: [] for w in workers}
    for i, (phase, backbone, ds_name, config_id, _, eval_concept, outdir) in enumerate(all_jobs):
        host, gpu = workers[i % len(workers)]
        _, prompt_file, concept, _, _ = ds_map[ds_name]
        if phase == "baseline":
            gen_cmd = make_baseline_cmd("sd14", prompt_file, outdir, gpu)
        elif phase == "safree":
            gen_cmd = make_safree_cmd("sd14", prompt_file, concept, outdir, gpu)
        else:
            ss = float(config_id.split("_ss")[1].split("_")[0])
            thr = float(config_id.split("_thr")[1].split("_")[0])
            how = config_id.split("_thr")[1].split("_", 2)[1]
            gen_cmd = make_ours_cmd("sd14", prompt_file, concept, outdir,
                                    ss, thr, how, host, gpu)
        worker_jobs[(host, gpu)].append(
            (phase, backbone, ds_name, config_id, gen_cmd, eval_concept, outdir))
    return worker_jobs


def build_sd3_jobs():
    """SD3: baseline+safree on siml-06 GPU 4-7; ours on siml-09 GPU 0."""
    ds_map = {ds[0]: ds for ds in DATASETS}
    bl_safree_jobs = []
    ours_jobs = []

    # Phase 1: baseline
    for ds_name, prompt_file, concept, eval_concept, _ in DATASETS:
        outdir = f"{OUT_ROOT}/baseline_sd3/{ds_name}"
        bl_safree_jobs.append(("baseline", "sd3", ds_name, "baseline",
                               eval_concept, outdir, prompt_file, concept))

    # Phase 2: safree
    for ds_name, prompt_file, concept, eval_concept, _ in DATASETS:
        outdir = f"{OUT_ROOT}/safree_sd3/{ds_name}"
        bl_safree_jobs.append(("safree", "sd3", ds_name, "safree",
                               eval_concept, outdir, prompt_file, concept))

    # Phase 3: ours sweep
    for ds_name, ss, thr, how in get_ours_sweep("sd3"):
        _, prompt_file, concept, eval_concept, _ = ds_map[ds_name]
        config_id = ours_config_id(ss, thr, how)
        outdir = f"{OUT_ROOT}/ours_sd3/{ds_name}/{config_id}"
        ours_jobs.append(("ours", "sd3", ds_name, config_id,
                          eval_concept, outdir, prompt_file, concept, ss, thr, how))

    # Assign baseline/safree round-robin to siml-06 GPU 4-7
    bl_workers = SD3_BL_WORKERS
    worker_jobs = {w: [] for w in bl_workers}
    # Also add siml-09 entry for ours
    worker_jobs[("siml-09", 0)] = worker_jobs.get(("siml-09", 0), [])

    for i, job in enumerate(bl_safree_jobs):
        phase, backbone, ds_name, config_id, eval_concept, outdir, prompt_file, concept = job
        host, gpu = bl_workers[i % len(bl_workers)]
        if phase == "baseline":
            gen_cmd = make_baseline_cmd("sd3", prompt_file, outdir, gpu)
        else:
            gen_cmd = make_safree_cmd("sd3", prompt_file, concept, outdir, gpu)
        worker_jobs[(host, gpu)].append(
            (phase, backbone, ds_name, config_id, gen_cmd, eval_concept, outdir))

    # Assign ours round-robin to siml-09 GPU 0 (only 1 worker)
    ours_workers = SD3_OURS_WORKERS
    for i, job in enumerate(ours_jobs):
        phase, backbone, ds_name, config_id, eval_concept, outdir, prompt_file, concept, ss, thr, how = job
        host, gpu = ours_workers[i % len(ours_workers)]
        gen_cmd = make_ours_cmd("sd3", prompt_file, concept, outdir,
                                ss, thr, how, host, gpu)
        worker_jobs[(host, gpu)].append(
            (phase, backbone, ds_name, config_id, gen_cmd, eval_concept, outdir))

    return worker_jobs


def build_flux1_jobs():
    """FLUX1: all phases on siml-08:4,5 + siml-09:0."""
    ds_map = {ds[0]: ds for ds in DATASETS}
    all_jobs = []

    # Phase 1: baseline
    for ds_name, prompt_file, concept, eval_concept, _ in DATASETS:
        outdir = f"{OUT_ROOT}/baseline_flux1/{ds_name}"
        all_jobs.append(("baseline", "flux1", ds_name, "baseline",
                         eval_concept, outdir, prompt_file, concept))

    # Phase 2: safree
    for ds_name, prompt_file, concept, eval_concept, _ in DATASETS:
        outdir = f"{OUT_ROOT}/safree_flux1/{ds_name}"
        all_jobs.append(("safree", "flux1", ds_name, "safree",
                         eval_concept, outdir, prompt_file, concept))

    # Phase 3: ours sweep
    for ds_name, ss, thr, how in get_ours_sweep("flux1"):
        _, prompt_file, concept, eval_concept, _ = ds_map[ds_name]
        config_id = ours_config_id(ss, thr, how)
        outdir = f"{OUT_ROOT}/ours_flux1/{ds_name}/{config_id}"
        all_jobs.append(("ours", "flux1", ds_name, config_id,
                         eval_concept, outdir, prompt_file, concept, ss, thr, how))

    # Assign round-robin to FLUX1 workers
    workers = FLUX1_WORKERS
    worker_jobs = {w: [] for w in workers}
    for i, job in enumerate(all_jobs):
        host, gpu = workers[i % len(workers)]
        phase = job[0]
        backbone, ds_name, config_id, eval_concept, outdir, prompt_file, concept = job[1:8]
        if phase == "baseline":
            gen_cmd = make_baseline_cmd("flux1", prompt_file, outdir, gpu)
        elif phase == "safree":
            gen_cmd = make_safree_cmd("flux1", prompt_file, concept, outdir, gpu)
        else:
            ss, thr, how = job[8], job[9], job[10]
            gen_cmd = make_ours_cmd("flux1", prompt_file, concept, outdir,
                                    ss, thr, how, host, gpu)
        worker_jobs[(host, gpu)].append(
            (phase, backbone, ds_name, config_id, gen_cmd, eval_concept, outdir))
    return worker_jobs


def write_worker_manifests(worker_jobs, label):
    total = 0
    for (host, gpu), jobs in sorted(worker_jobs.items()):
        if not jobs:
            continue
        fname = f"{MANIFEST_DIR}/worker_{host}_g{gpu}.csv"
        # Append mode for siml-09 which gets both FLUX1 and SD3 ours
        mode = "a" if os.path.exists(fname) else "w"
        with open(fname, mode, newline="") as f:
            writer = csv.writer(f)
            if mode == "w":
                writer.writerow(["phase", "backbone", "dataset", "config_id",
                                 "gen_cmd", "eval_concept", "output_dir"])
            for row in jobs:
                writer.writerow(list(row))
        n = len(jobs)
        print(f"  {fname}: {n} jobs ({'appended' if mode == 'a' else 'created'})")
        total += n
    return total


def main():
    os.makedirs(MANIFEST_DIR, exist_ok=True)

    # Clear existing manifests
    import glob
    for f in glob.glob(f"{MANIFEST_DIR}/*.csv"):
        os.remove(f)

    print("=== Building SD1.4 manifests (siml-01 GPU 0-7) ===")
    n14 = write_worker_manifests(build_sd14_jobs(), "sd14")

    print("=== Building SD3 manifests ===")
    print("  baseline+safree → siml-06 GPU 4-7")
    print("  ours (safegen)  → siml-09 GPU 0 [OOM on A6000 49GB]")
    n3 = write_worker_manifests(build_sd3_jobs(), "sd3")

    print("=== Building FLUX1 manifests (siml-08:4,5 + siml-09:0) ===")
    nf = write_worker_manifests(build_flux1_jobs(), "flux1")

    total = n14 + n3 + nf
    print(f"\nTotal jobs: {total} ({n14} SD1.4, {n3} SD3, {nf} FLUX1)")
    print("Worker manifest summary:")
    import glob
    for f in sorted(glob.glob(f"{MANIFEST_DIR}/*.csv")):
        with open(f) as fh:
            n = sum(1 for _ in fh) - 1  # subtract header
        print(f"  {os.path.basename(f)}: {n} jobs")


if __name__ == "__main__":
    main()
