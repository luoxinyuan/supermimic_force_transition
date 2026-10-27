#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
OUT=runs/exp02_teacher_seed0
mkdir -p "$OUT"
exec 9>"$OUT/launch.lock"
flock -n 9 || { echo 'Teacher already running'; exit 1; }
"$SMIMIC_PY" - <<'PY'
import json
from pathlib import Path
for f,key in [('force_validation.json','passed'),('zero_force_validation.json','passed'),('wm_validation.json','force_conditioning_passed')]:
    assert json.loads((Path('reports/exp02')/f).read_text())[key],f
PY
RESUME=runs/exp02_wm_warmup/last.pt
if [[ -f "$OUT/last.pt" ]]; then RESUME="$OUT/last.pt"; fi
exec "$SMIMIC_PY" -m supermimic.force_transition.train --out "$OUT" --resume "$RESUME" \
  --num-envs 256 --wm-batch 256 --pi-batch 256 --iterations 100000 --wm-only-updates 0 \
  --eval-every 1000 --ckpt-every 1000 --wall-hours 2 "$@"
