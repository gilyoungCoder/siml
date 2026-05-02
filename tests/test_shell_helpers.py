from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_shell(script: str) -> str:
    completed = subprocess.run(
        ["bash", "-lc", script],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def test_shell_helpers_parse_canonical_qwen_and_nudenet_files(tmp_path):
    (tmp_path / "results_qwen_nudity.txt").write_text(
        "Concept: NUDITY\n"
        "Model: qwen\n"
        "Total images: 10\n"
        "Distribution:\n"
        "  - NotRel: 1 (10.0%)\n"
        "  - Safe: 4 (40.0%)\n"
        "  - Partial: 2 (20.0%)\n"
        "  - Full: 3 (30.0%)\n"
        "\n"
        "SR (Safe+Partial): 6/10 (60.0%)\n",
        encoding="utf-8",
    )
    (tmp_path / "results_nudenet.txt").write_text(
        "Total Images: 10\nUnsafe Count: 3\nUnsafe Rate: 0.30 (30.0%)\n",
        encoding="utf-8",
    )

    stdout = run_shell(
        f'''
        source "{REPO_ROOT / "scripts/lib/repo_env.sh"}"
        dir="{tmp_path}"
        printf "%s\\n%s\\n%s\\n%s\\n" \
          "$(unlearning_qwen_count "$dir" NotRel)" \
          "$(unlearning_qwen_count "$dir" Safe)" \
          "$(unlearning_qwen_percent_value "$dir" SR)" \
          "$(unlearning_nudenet_percent "$dir")"
        '''
    )

    assert stdout.splitlines() == ["1", "4", "60.0", "30.0%"]


def test_shell_helpers_fallback_to_legacy_qwen_filename(tmp_path):
    (tmp_path / "results_qwen3_vl_nudity.txt").write_text(
        "Concept: NUDITY\n"
        "Model: qwen3_vl\n"
        "Total images: 5\n"
        "Distribution:\n"
        "  - NotRel: 1 (20.0%)\n"
        "  - Safe: 2 (40.0%)\n"
        "  - Partial: 1 (20.0%)\n"
        "  - Full: 1 (20.0%)\n"
        "\n"
        "SR (Safe+Partial): 3/5 (60.0%)\n",
        encoding="utf-8",
    )

    stdout = run_shell(
        f'''
        source "{REPO_ROOT / "scripts/lib/repo_env.sh"}"
        dir="{tmp_path}"
        printf "%s\\n%s\\n" \
          "$(basename "$(unlearning_find_qwen_result_txt "$dir")")" \
          "$(unlearning_qwen_percent_value "$dir" Full)"
        '''
    )

    assert stdout.splitlines() == ["results_qwen3_vl_nudity.txt", "20.0"]


def test_shell_helpers_fallback_to_categories_json_when_txt_missing(tmp_path):
    (tmp_path / "categories_qwen_nudity.json").write_text(
        json.dumps(
            {
                "a.png": {"category": "NotRel"},
                "b.png": {"category": "Safe"},
                "c.png": {"category": "Partial"},
                "d.png": {"category": "Full"},
            }
        ),
        encoding="utf-8",
    )

    stdout = run_shell(
        f'''
        source "{REPO_ROOT / "scripts/lib/repo_env.sh"}"
        dir="{tmp_path}"
        printf "%s\\n%s\\n" \
          "$(unlearning_qwen_count "$dir" NotRel)" \
          "$(unlearning_qwen_percent_value "$dir" SR)"
        '''
    )

    assert stdout.splitlines() == ["1", "50.0"]

