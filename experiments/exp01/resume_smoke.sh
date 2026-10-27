#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
"$SMIMIC_PY" -m supermimic.train --out runs/exp01_resume_smoke --resume runs/exp01_smoke/last.pt --num-envs 64 --buffer-steps 128 --wm-batch 64 --wm-batches 1 --pi-batch 64 --iterations 15 --log-every 5 --ckpt-every 15 --eval-every 15 --eval-envs 40
"$SMIMIC_PY" experiments/exp01/check_run.py runs/exp01_resume_smoke
"$SMIMIC_PY" - <<'PY'
import torch
old = torch.load('runs/exp01_smoke/last.pt', map_location='cpu', weights_only=False)
new = torch.load('runs/exp01_resume_smoke/last.pt', map_location='cpu', weights_only=False)
assert new['it'] == 15 and new['updates'] > old['updates']
assert new['env_steps'] == old['env_steps'] + (64 + 5) * 64
for name in ('wm', 'policy', 'critic'):
    assert any(not torch.equal(v, new[name][k]) for k, v in old[name].items())
print('PASS: checkpoint resumed, replay refilled, all three networks updated.')
PY
