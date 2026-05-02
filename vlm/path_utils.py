from __future__ import annotations

import os
from pathlib import Path


def get_repo_root() -> Path:
    override = os.environ.get("UNLEARNING_REPO_ROOT")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1]


def get_user_home() -> Path:
    override = os.environ.get("UNLEARNING_USER_HOME")
    if override:
        return Path(override)
    return get_repo_root().parent


def get_guided2_root() -> Path:
    override = os.environ.get("UNLEARNING_GUIDED2_ROOT")
    if override:
        return Path(override)
    return get_user_home() / "guided2-safe-diffusion"


def get_conda_python(env_name: str) -> Path:
    env_var = f"UNLEARNING_{env_name.upper()}_PYTHON"
    override = os.environ.get(env_var)
    if override:
        return Path(override)
    return get_user_home() / ".conda" / "envs" / env_name / "bin" / "python3.10"


def get_scg_outputs_root() -> Path:
    return get_repo_root() / "SoftDelete+CG" / "scg_outputs"


def get_outputs_root() -> Path:
    return get_repo_root() / "outputs"
