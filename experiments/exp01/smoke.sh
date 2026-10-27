#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
"$SMIMIC_PY" experiments/exp01/validate_data.py
"$SMIMIC_PY" -m supermimic.train --out runs/exp01_smoke --num-envs 64 --buffer-steps 128 --wm-batch 64 --wm-batches 1 --pi-batch 64 --iterations 10 --log-every 5 --ckpt-every 10 --eval-every 10 --eval-envs 40
