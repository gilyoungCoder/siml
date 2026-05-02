#!/usr/bin/env python3
"""Eval dispatcher (siml-09 side): scans NFE + scale-robustness output trees
for cells that have generated images but no Qwen v5 result file, then runs eval.

Each worker is assigned `cells where (i % nworkers) == worker_id` to balance load.
Idempotent: skips cells whose result file already exists with size > 50 bytes.

Usage: python eval_dispatcher.py <GPU> <WORKER_ID> <NWORKERS>
"""
import os, re, subprocess, sys
from pathlib import Path

LAUNCH = Path("/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep")
PHASE_NFE = LAUNCH / "outputs/phase_nfe_walltime_v3"
PHASE_SCALE = LAUNCH / "outputs/phase_scale_robustness"
LOGDIR = LAUNCH / "logs/eval_dispatcher"

PY_VLM = "/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
EVAL_SCRIPT = "/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py"
EVAL_CWD = "/mnt/home3/yhgil99/unlearning/vlm"

# concept -> Qwen rubric token (filename token)
EVAL_NAME = {
    "sexual": "nudity", "violence": "violence", "self-harm": "self_harm",
    "shocking": "shocking", "illegal_activity": "illegal",
    "harassment": "harassment", "hate": "hate",
}

# Cell name patterns:
#   NFE tree:   <method>_<concept>_steps<NFE>
#   Scale tree: <method>_<concept>_scale<X>
NFE_PAT   = re.compile(r"^(?P<method>[a-z_]+?)_(?P<concept>sexual|violence|self-harm|shocking|illegal_activity|harassment|hate)_steps(?P<nfe>\d+)$")
SCALE_PAT = re.compile(r"^(?P<method>sld|ebsg)_(?P<concept>sexual|violence|self-harm|shocking|illegal_activity|harassment|hate)_scale(?P<scale>\d+)$")


def cell_concept(name):
    m = NFE_PAT.match(name) or SCALE_PAT.match(name)
    return m.group("concept") if m else None


def png_count(d):
    n = len(list(d.glob("*.png")))
    n += len(list((d / "all").glob("*.png")))
    n += len(list((d / "safe").glob("*.png")))
    n += len(list((d / "unsafe").glob("*.png")))
    n += len(list((d / "generated").glob("*.png")))
    return n


def needs_eval(cell_dir, concept):
    rub = EVAL_NAME[concept]
    f = cell_dir / f"results_qwen3_vl_{rub}_v5.txt"
    if f.exists() and f.stat().st_size > 50:
        return False
    # Also check inside `all/` (SD/SGF style)
    f2 = cell_dir / "all" / f"results_qwen3_vl_{rub}_v5.txt"
    if f2.exists() and f2.stat().st_size > 50:
        return False
    return True


def collect_cells():
    cells = []
    for tree in (PHASE_NFE, PHASE_SCALE):
        if not tree.exists(): continue
        for d in sorted(tree.iterdir()):
            if not d.is_dir(): continue
            c = cell_concept(d.name)
            if c is None: continue
            if png_count(d) < 1: continue
            if not needs_eval(d, c): continue
            cells.append((d, c))
    return cells


def run_eval(cell_dir, concept, gpu, log):
    target = cell_dir
    inner = cell_dir / "all"
    if inner.exists() and any(inner.glob("*.png")):
        target = inner
    cmd = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", PY_VLM, EVAL_SCRIPT,
           str(target), EVAL_NAME[concept], "qwen"]
    with open(log, "a") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=EVAL_CWD).returncode
    # Mirror result file up to cell_dir if eval wrote into <cell_dir>/all/
    if target != cell_dir:
        for fn in (f"results_qwen3_vl_{EVAL_NAME[concept]}_v5.txt",
                   f"categories_qwen3_vl_{EVAL_NAME[concept]}_v5.json"):
            src = target / fn; dst = cell_dir / fn
            if src.exists() and not dst.exists():
                try: dst.write_bytes(src.read_bytes())
                except OSError: pass
    return rc


def main():
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    wid = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nworkers = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    LOGDIR.mkdir(parents=True, exist_ok=True)
    log = LOGDIR / f"g{gpu}_w{wid}.log"
    with open(log, "a") as f:
        f.write(f"[start] gpu={gpu} worker={wid}/{nworkers}\n")

    # Re-scan periodically since generation is still running (siml-05)
    seen = set()
    while True:
        cells = collect_cells()
        # Round-robin slice
        my_cells = [(d, c) for i, (d, c) in enumerate(cells) if (i % nworkers) == wid]
        # Filter unseen
        new_cells = [(d, c) for (d, c) in my_cells if str(d) not in seen]
        if not new_cells:
            with open(log, "a") as f:
                f.write(f"[idle] {len(cells)} pending across all workers; sleeping 60s\n")
            import time; time.sleep(60)
            # Stop if nothing pending after waiting
            if len(cells) == 0:
                # Wait one more round to see if gen finishes producing more
                time.sleep(60)
                if not collect_cells():
                    break
            continue
        for cell_dir, concept in new_cells:
            with open(log, "a") as f:
                f.write(f"[eval] {cell_dir.name} concept={concept}\n")
            try:
                rc = run_eval(cell_dir, concept, gpu, log)
                with open(log, "a") as f:
                    f.write(f"[eval-done] {cell_dir.name} rc={rc}\n")
            except Exception as e:
                with open(log, "a") as f:
                    f.write(f"[eval-exc] {cell_dir.name} {e}\n")
            seen.add(str(cell_dir))

    with open(log, "a") as f:
        f.write(f"[end] gpu={gpu} worker={wid}\n")


if __name__ == "__main__":
    main()
