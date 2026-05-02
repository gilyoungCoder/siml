# Output Storage Guide

This repository contains many generated images, evaluation dumps, logs, and
 intermediate research artifacts. The source tree should not be the default
 long-term storage location for new bulk outputs.

## Default rule

Keep in git:

- source code
- small docs
- lightweight configs
- compact summaries that are hard to regenerate

Do not rely on git for:

- generated image grids
- bulk prompt outputs
- per-run logs
- temporary evaluation caches
- local coordination/state directories

## Already treated as generated artifacts

Examples include:

- `**/outputs/`
- `**/generated/`
- `**/samples/`
- `**/scg_outputs/`
- `*.log`
- `results_*.txt`
- `categories_*.json`

## Local state to keep out of source control

- `.omx/`
- `.omc/`
- `.pytest_cache/`

## Research communication artifacts

Meeting packs, temporary HTML presentations, and similar export-heavy material
should be treated as generated artifacts unless they are explicitly curated for
publication or archival reference.

## Practical recommendation

For new experiment runs:

1. write outputs to ignored artifact directories
2. keep only compact final summaries in the repo
3. document any non-obvious output locations in the relevant method README or
   in `docs/`

