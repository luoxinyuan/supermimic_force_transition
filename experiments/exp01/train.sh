#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
OUT=runs/exp01_lafan40_seed0
mkdir -p "$OUT"
exec 9>"$OUT/launch.lock"
flock -n 9 || { echo 'Experiment already running'; exit 1; }
RESUME=()
if [[ -f "$OUT/last.pt" ]]; then RESUME=(--resume "$OUT/last.pt"); fi
exec "$SMIMIC_PY" -m supermimic.train --out "$OUT" --seed 0 --wall-hours 3 --eval-every 4000 --ckpt-every 4000 "${RESUME[@]}" "$@"
