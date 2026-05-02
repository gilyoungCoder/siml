#!/usr/bin/env python3
"""NFE wall-clock sweep orchestrator (5 method x 7 concept x 8 NFE = 280 cells).

Per-concept best Ours config (override safety_scale / cas_threshold / theta_text / theta_img).
SD/SGF use concept-specific YAML configs already on NFS.
SAFREE uses concept category for negative-prompt-space.
Baseline is plain SD1.4 with seed/cfg/steps only.

Outputs to: launch_0426_full_sweep/outputs/phase_nfe_walltime_v3/<method>_<concept>_steps<N>/
Idempotent: skips cell if outdir already has 60 PNGs.
"""
import json, os, sys, subprocess
from pathlib import Path

REPO = "/mnt/home3/yhgil99/unlearning"
LAUNCH = f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep"
PR = f"{LAUNCH}/paper_results"
OUT_BASE = f"{LAUNCH}/outputs/phase_nfe_walltime_v3"
LOGDIR = f"{LAUNCH}/logs/nfe_walltime_v3"

# Envs
PY_EBSG     = "/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
PY_SAFREE   = "/mnt/home3/yhgil99/.conda/envs/safree/bin/python3.10"
PY_BASELINE = "/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
PY_SFGD     = "/mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10"
PY_VLM      = "/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
BASELINE_SCRIPT = f"{LAUNCH}/scripts/baseline_runner.py"
EVAL_SCRIPT = "/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py"
EVAL_CWD    = "/mnt/home3/yhgil99/unlearning/vlm"

# Sweep grid
CONCEPTS = ["sexual", "violence", "self-harm", "shocking", "illegal_activity", "harassment", "hate"]
STEPS = [5, 10, 15, 20, 25, 30, 40, 50]
METHODS = ["baseline", "safree", "safedenoiser", "sgf", "sld_max", "sld_medium", "ebsg"]
SLD_RUNNER = f"{LAUNCH}/scripts/sld_runner.py"

# concept -> args.json subdir name (note illegal_activity vs illegal)
ARGS_DIR = {
    "sexual": "sexual", "violence": "violence", "self-harm": "self-harm",
    "shocking": "shocking", "illegal_activity": "illegal",
    "harassment": "harassment", "hate": "hate",
}
# concept -> SAFREE category arg
SAFREE_CAT = {
    "sexual": "nudity", "violence": "violence", "self-harm": "self-harm",
    "shocking": "shocking", "illegal_activity": "illegal_activity",
    "harassment": "harassment", "hate": "hate",
}
# concept -> Qwen evaluator name (filename token)
EVAL_NAME = {
    "sexual": "nudity", "violence": "violence", "self-harm": "self_harm",
    "shocking": "shocking", "illegal_activity": "illegal",
    "harassment": "harassment", "hate": "hate",
}

# Per-concept Ours best config override (from user 2026-05-02 spec)
OURS_OVERRIDE = {
    "sexual":           dict(safety_scale=20.0,  cas_threshold=0.5, attn_threshold=0.10, img_attn_threshold=0.30),
    "violence":         dict(safety_scale=20.0,  cas_threshold=0.4, attn_threshold=0.30, img_attn_threshold=0.10),
    "self-harm":        dict(safety_scale=7.0,   cas_threshold=0.5, attn_threshold=0.10, img_attn_threshold=0.10),
    "shocking":         dict(safety_scale=27.5,  cas_threshold=0.6, attn_threshold=0.15, img_attn_threshold=0.10),
    "illegal_activity": dict(safety_scale=25.0,  cas_threshold=0.6, attn_threshold=0.10, img_attn_threshold=0.50),
    "harassment":       dict(safety_scale=31.25, cas_threshold=0.5, attn_threshold=0.10, img_attn_threshold=0.50),
    "hate":             dict(safety_scale=28.0,  cas_threshold=0.6, attn_threshold=0.25, img_attn_threshold=0.0375),
}

# Concept-specific NEGSPACE for SD/SGF (lifted from bench_timing_isolated.sh + extended)
NEGSPACE = {
    "sexual":           "nudity|naked|nude|topless|breasts|genitals|pornography|explicit nudity|sexual content|fully exposed",
    "violence":         "violence|blood|gore|weapon attack|brutal action|dismemberment|physical assault|injury|weapon threat|shooting|stabbing",
    "self-harm":        "self-harm|self injury|cutting|wrist|suicide|hanging|overdose|self mutilation",
    "shocking":         "shocking|grotesque|disturbing|disfigured|horror|nightmarish|monstrous|gory",
    "illegal_activity": "illegal activity|drugs|cocaine|heroin|narcotics|crime scene|contraband",
    "harassment":       "harassment|bullying|mockery|intimidation|abuse|stalking|insulting gestures",
    "hate":             "hate|hate symbol|swastika|kkk|racial slur|white supremacy|hateful gathering",
}

CSV_BASE = f"{PR}/reproduce/sd14_q16_repro_ours_baselines_20260430/prompts/i2p_q16_csv"
TXT_BASE = f"{REPO}/CAS_SpatialCFG/prompts/i2p_q16_top60"
CFG_BASE = f"{PR}/reproduce/sd14_q16_repro_ours_baselines_20260430/configs/concept_specific_official"


def prompt_paths(concept):
    """Return (txt_path, csv_path). Falls back to CSV-derived txt if needed."""
    txt = f"{TXT_BASE}/{concept}_q16_top60.txt"
    csv = f"{CSV_BASE}/{concept}_q16_top60.csv"
    return txt, csv


def png_count(d):
    p = Path(d)
    n = len(list(p.glob("*.png")))
    n += len(list(p.glob("all/*.png")))
    n += len(list(p.glob("generated/*.png")))
    return n


def needs_eval(outdir, concept):
    f = Path(outdir) / f"results_qwen3_vl_{EVAL_NAME[concept]}_v5.txt"
    if not f.exists():
        return True
    return f.stat().st_size < 50


def run_baseline(concept, steps, outdir, gpu, log):
    txt, _ = prompt_paths(concept)
    cmd = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", PY_BASELINE, BASELINE_SCRIPT,
           "--prompts", txt, "--outdir", outdir,
           "--steps", str(steps), "--seed", "42", "--cfg_scale", "7.5",
           "--device", "cuda:0"]
    with open(log, "a") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              cwd=f"{REPO}/SafeGen").returncode


def run_safree(concept, steps, outdir, gpu, log):
    txt, _ = prompt_paths(concept)
    cmd = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", PY_SAFREE,
           f"{REPO}/SAFREE/gen_safree_single.py",
           "--txt", txt, "--save-dir", outdir,
           "--category", SAFREE_CAT[concept],
           "--re_attn_t=-1,1001", "--linear_per_prompt_seed",
           "--num_inference_steps", str(steps),
           "--safree", "-svf", "-lra"]
    with open(log, "a") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                            cwd=f"{REPO}/SAFREE").returncode
    gendir = f"{outdir}/generated"
    if os.path.isdir(gendir):
        for fn in os.listdir(gendir):
            if fn.endswith(".png"):
                os.rename(f"{gendir}/{fn}", f"{outdir}/{fn}")
        try: os.rmdir(gendir)
        except OSError: pass
    return rc


def run_safedenoiser(concept, steps, outdir, gpu, log):
    _, csv = prompt_paths(concept)
    yaml = f"{CFG_BASE}/safedenoiser_i2p_{concept}.yaml"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for sub in ("all", "safe", "unsafe"):
        Path(f"{outdir}/{sub}").mkdir(parents=True, exist_ok=True)
    cwd = f"{LAUNCH}/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/code/official_repos/Safe_Denoiser"
    cmd = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", "PYTHONNOUSERSITE=1",
           f"I2P_NEGATIVE_PROMPT_SPACE={NEGSPACE[concept]}", PY_SFGD,
           "run_copro_i2p_concept_np.py",
           "--nudenet-path=pretrained/classifier_model.onnx", "--nudity_thr=0.6",
           f"--num_inference_steps={steps}",
           "--config=configs/base/vanilla/safree_neg_prompt_config.json",
           "--safe_level=MEDIUM", f"--data={csv}", "--category=all",
           f"--task_config={yaml}",
           f"--save-dir={outdir}",
           "--erase_id=safree_neg_prompt_rep_threshold_time",
           "--guidance_scale=7.5", "--seed=42", "--valid_case_numbers=0,100000"]
    with open(log, "a") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=cwd).returncode


def run_sgf(concept, steps, outdir, gpu, log):
    _, csv = prompt_paths(concept)
    yaml = f"{CFG_BASE}/sgf_i2p_{concept}.yaml"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for sub in ("all", "safe", "unsafe"):
        Path(f"{outdir}/{sub}").mkdir(parents=True, exist_ok=True)
    cwd = f"{LAUNCH}/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/code/official_repos/SGF/nudity_sdv1"
    cmd = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", "PYTHONNOUSERSITE=1",
           f"I2P_NEGATIVE_PROMPT_SPACE={NEGSPACE[concept]}", PY_SFGD,
           "generate_unsafe_sgf_i2p_concept_np.py",
           "--nudenet-path=pretrained/classifier_model.onnx", "--nudity_thr=0.6",
           f"--num_inference_steps={steps}",
           "--config=configs/base/vanilla/safree_neg_prompt_config.json",
           "--safe_level=MEDIUM", f"--data={csv}", "--category=all",
           f"--task_config={yaml}",
           f"--save-dir={outdir}",
           "--erase_id=safree_neg_prompt_rep_time",
           "--guidance_scale=7.5", "--seed=42", "--valid_case_numbers=0,100000"]
    with open(log, "a") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=cwd).returncode


def run_sld(variant, concept, steps, outdir, gpu, log):
    """SLD via local sld_runner.py. variant in {'Max','Medium','Strong','Weak'}."""
    txt, _ = prompt_paths(concept)
    cmd = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", "PYTHONNOUSERSITE=1", PY_EBSG,
           SLD_RUNNER,
           "--prompts", txt, "--outdir", outdir,
           "--variant", variant,
           "--steps", str(steps), "--seed", "42", "--cfg_scale", "7.5"]
    with open(log, "a") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def run_sld_max(concept, steps, outdir, gpu, log):
    return run_sld("Max", concept, steps, outdir, gpu, log)


def run_sld_medium(concept, steps, outdir, gpu, log):
    return run_sld("Medium", concept, steps, outdir, gpu, log)


def run_ebsg(concept, steps, outdir, gpu, log):
    txt, _ = prompt_paths(concept)
    args_path = f"{PR}/single/{ARGS_DIR[concept]}/args.json"
    a = json.load(open(args_path))
    ovr = OURS_OVERRIDE[concept]
    a.update(ovr)
    cmd = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", "PYTHONNOUSERSITE=1", PY_EBSG, "-m", "safegen.generate_family",
           "--prompts", txt, "--outdir", outdir,
           "--family_guidance", "--family_config", a["family_config"],
           "--probe_mode", a.get("probe_mode", "both"),
           "--probe_fusion", a.get("probe_fusion", "union"),
           "--how_mode", a.get("how_mode", "hybrid"),
           "--cas_threshold", str(a["cas_threshold"]),
           "--safety_scale", str(a["safety_scale"]),
           "--attn_threshold", str(a["attn_threshold"]),
           "--img_attn_threshold", str(a["img_attn_threshold"]),
           "--n_img_tokens", str(a.get("n_img_tokens", 4)),
           "--steps", str(steps), "--seed", "42", "--cfg_scale", "7.5",
           "--target_concepts", *a["target_concepts"],
           "--target_words", *a["target_words"],
           "--anchor_concepts", *a.get("anchor_concepts", ["safe_scene"])]
    with open(log, "a") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              cwd=f"{REPO}/SafeGen").returncode


def run_eval(concept, outdir, gpu, log):
    """Run Qwen3-VL v5 eval if result file missing.
    Calls opensource_vlm_i2p_all_v5.py <dir> <rubric> qwen which writes
    results_qwen3_vl_<rubric>_v5.txt + categories_qwen3_vl_<rubric>_v5.json into outdir.
    For SD/SGF whose images live in <outdir>/all/, point eval there instead."""
    target = outdir
    if (Path(outdir) / "all").exists() and any(Path(outdir, "all").glob("*.png")):
        target = f"{outdir}/all"
    cmd = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", PY_VLM, EVAL_SCRIPT,
           target, EVAL_NAME[concept], "qwen"]
    with open(log, "a") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=EVAL_CWD).returncode
    # If eval wrote into <outdir>/all/, mirror result file up to <outdir>/ for needs_eval check
    if target != outdir:
        for fn in (f"results_qwen3_vl_{EVAL_NAME[concept]}_v5.txt",
                   f"categories_qwen3_vl_{EVAL_NAME[concept]}_v5.json"):
            src = Path(target) / fn
            dst = Path(outdir) / fn
            if src.exists() and not dst.exists():
                try: dst.write_bytes(src.read_bytes())
                except OSError: pass
    return rc


METHOD_FN = {
    "baseline": run_baseline, "safree": run_safree,
    "safedenoiser": run_safedenoiser, "sgf": run_sgf,
    "sld_max": run_sld_max, "sld_medium": run_sld_medium,
    "ebsg": run_ebsg,
}


def main():
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    slot = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nslots = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    do_eval = (sys.argv[4] == "1") if len(sys.argv) > 4 else True

    Path(OUT_BASE).mkdir(parents=True, exist_ok=True)
    Path(LOGDIR).mkdir(parents=True, exist_ok=True)
    log = f"{LOGDIR}/g{gpu}_s{slot}.log"
    with open(log, "a") as f:
        f.write(f"[start] GPU={gpu} slot={slot}/{nslots} eval={do_eval}\n")

    cells = [(m, c, s) for m in METHODS for c in CONCEPTS for s in STEPS]
    for i, (m, c, s) in enumerate(cells):
        if (i % nslots) != slot:
            continue
        cell = f"{m}_{c}_steps{s}"
        outdir = f"{OUT_BASE}/{cell}"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        n = png_count(outdir)
        if n >= 60 and (not do_eval or not needs_eval(outdir, c)):
            with open(log, "a") as f:
                f.write(f"[skip-done] {cell} n={n}\n")
            continue
        if n < 60:
            with open(log, "a") as f:
                f.write(f"[gen] {cell}\n")
            try:
                rc = METHOD_FN[m](c, s, outdir, gpu, log)
                n2 = png_count(outdir)
                with open(log, "a") as f:
                    f.write(f"[gen-done] {cell} rc={rc} imgs={n2}\n")
            except Exception as e:
                with open(log, "a") as f:
                    f.write(f"[gen-exc] {cell} {e}\n")
                continue
        if do_eval and needs_eval(outdir, c) and png_count(outdir) > 0:
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
