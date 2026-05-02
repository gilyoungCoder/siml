#!/usr/bin/env bash
set -euo pipefail

# GPU 지정 (원하면 바꾸세요)
export CUDA_VISIBLE_DEVICES=6

python check.py --file words.txt --target "nude" --template "a {} person"