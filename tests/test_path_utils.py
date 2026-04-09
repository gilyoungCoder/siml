from __future__ import annotations

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


def test_repo_root_defaults_to_current_repository(monkeypatch):
    monkeypatch.delenv("UNLEARNING_REPO_ROOT", raising=False)
    module = load_module("path_utils_default", "vlm/path_utils.py")

    assert module.get_repo_root() == REPO_ROOT


def test_repo_root_can_be_overridden(monkeypatch, tmp_path):
    monkeypatch.setenv("UNLEARNING_REPO_ROOT", str(tmp_path / "repo"))
    module = load_module("path_utils_repo_override", "vlm/path_utils.py")

    assert module.get_repo_root() == tmp_path / "repo"


def test_user_home_defaults_to_parent_of_repo_root(monkeypatch):
    monkeypatch.delenv("UNLEARNING_REPO_ROOT", raising=False)
    monkeypatch.delenv("UNLEARNING_USER_HOME", raising=False)
    module = load_module("path_utils_user_home_default", "vlm/path_utils.py")

    assert module.get_user_home() == REPO_ROOT.parent


def test_guided2_and_conda_paths_respect_user_home_override(monkeypatch, tmp_path):
    monkeypatch.setenv("UNLEARNING_USER_HOME", str(tmp_path / "user"))
    module = load_module("path_utils_user_home_override", "vlm/path_utils.py")

    assert module.get_guided2_root() == tmp_path / "user" / "guided2-safe-diffusion"
    assert module.get_conda_python("sdd_copy") == tmp_path / "user" / ".conda" / "envs" / "sdd_copy" / "bin" / "python3.10"
    assert module.get_conda_python("vlm") == tmp_path / "user" / ".conda" / "envs" / "vlm" / "bin" / "python3.10"


def test_output_root_helpers_are_repo_relative(monkeypatch):
    monkeypatch.delenv("UNLEARNING_REPO_ROOT", raising=False)
    module = load_module("path_utils_output_roots", "vlm/path_utils.py")

    assert module.get_scg_outputs_root() == REPO_ROOT / "SoftDelete+CG" / "scg_outputs"
    assert module.get_outputs_root() == REPO_ROOT / "outputs"
