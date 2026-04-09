from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_alignment_prompts_preserves_idx_and_prompt_variants(tmp_path):
    module = load_module("eval_vqascore_alignment", "vlm/eval_vqascore_alignment.py")
    csv_path = tmp_path / "alignment.csv"
    csv_path.write_text(
        "idx,prompt,anchor_prompt,erased_prompt\n"
        '3,"unsafe prompt","safe anchor","erased prompt"\n',
        encoding="utf-8",
    )

    rows = module.load_alignment_prompts(str(csv_path))

    assert rows == [
        {
            "idx": 3,
            "prompt": "unsafe prompt",
            "anchor_prompt": "safe anchor",
            "erased_prompt": "erased prompt",
        }
    ]


def test_load_alignment_prompts_falls_back_to_detected_prompt_column(tmp_path):
    module = load_module("eval_vqascore_alignment", "vlm/eval_vqascore_alignment.py")
    csv_path = tmp_path / "alignment_adv.csv"
    csv_path.write_text(
        "idx,adv_prompt,anchor_prompt,erased_prompt\n"
        '0,"mma prompt","safer mma prompt","trimmed prompt"\n',
        encoding="utf-8",
    )

    rows = module.load_alignment_prompts(str(csv_path))

    assert rows[0]["prompt"] == "mma prompt"
    assert rows[0]["anchor_prompt"] == "safer mma prompt"
    assert rows[0]["erased_prompt"] == "trimmed prompt"


def test_load_simple_prompts_prefers_known_csv_prompt_columns(tmp_path):
    module = load_module("eval_vqascore_alignment", "vlm/eval_vqascore_alignment.py")
    csv_path = tmp_path / "simple.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "sensitive prompt"])
        writer.writeheader()
        writer.writerow({"id": "0", "sensitive prompt": "ring-a-bell prompt"})

    rows = module.load_simple_prompts(str(csv_path))

    assert rows == [
        {
            "idx": 0,
            "prompt": "ring-a-bell prompt",
            "anchor_prompt": "",
            "erased_prompt": "",
        }
    ]


def test_load_simple_prompts_reads_txt_lines_in_order(tmp_path):
    module = load_module("eval_vqascore_alignment", "vlm/eval_vqascore_alignment.py")
    txt_path = tmp_path / "simple.txt"
    txt_path.write_text("prompt one\n\nprompt two\n", encoding="utf-8")

    rows = module.load_simple_prompts(str(txt_path))

    assert [row["prompt"] for row in rows] == ["prompt one", "prompt two"]
    assert [row["idx"] for row in rows] == [0, 1]

