# Runtime Configuration

Active scripts should resolve paths through the repository root and a small set
of environment variables rather than hardcoded machine-specific absolute paths.

## Environment variables

- `UNLEARNING_REPO_ROOT`
  - optional override for repository root
- `UNLEARNING_USER_HOME`
  - optional override for the user home that contains sibling repos and conda envs
- `UNLEARNING_SDD_COPY_PYTHON`
  - optional override for the `sdd_copy` interpreter
- `UNLEARNING_VLM_PYTHON`
  - optional override for the `vlm` interpreter
- `UNLEARNING_GUIDED2_ROOT`
  - optional override for the sibling `guided2-safe-diffusion` checkout

## Default resolution

If unset:

- repo root is inferred from the current checkout
- user home defaults to the parent directory of the repo root
- `guided2-safe-diffusion` defaults to `${UNLEARNING_USER_HOME}/guided2-safe-diffusion`
- conda interpreters default to:
  - `${UNLEARNING_USER_HOME}/.conda/envs/sdd_copy/bin/python3.10`
  - `${UNLEARNING_USER_HOME}/.conda/envs/vlm/bin/python3.10`

## Shell bootstrap

Active shell entrypoints should source:

- `scripts/lib/repo_env.sh`

This exports the shared path variables and keeps scripts relocatable across
machines with similar directory layouts.

