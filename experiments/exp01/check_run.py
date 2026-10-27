"""Validate a completed training smoke checkpoint and its real-simulator evaluation."""
import argparse
import json
import math
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument('run', type=Path)
a = p.parse_args()
ck = torch.load(a.run / 'last.pt', map_location='cpu', weights_only=False)
assert ck['updates'] > 0 and ck['it'] > 0
for name in ('wm', 'policy', 'critic'):
    assert all(torch.isfinite(v).all() for v in ck[name].values())
for name in ('opt_wm', 'opt_pi', 'opt_v'):
    states = ck[name]['state']
    assert states and all(float(s['step']) > 0 for s in states.values())
records = [json.loads(line) for line in (a.run / 'log.jsonl').read_text().splitlines()]
for r in records:
    assert all(math.isfinite(v) for v in r.values() if isinstance(v, (float, int))), r
assert records[-1]['pi_gnorm'] > 0
if (a.run / 'eval.jsonl').exists():
    ev = json.loads((a.run / 'eval.jsonl').read_text().splitlines()[-1])
    assert ev['num_clips'] == 40
    assert len(ev['per_clip_norm_return']) == 40
    assert all(math.isfinite(v) for v in ev['per_clip_norm_return'])
assert (a.run / 'DONE').exists()
print(json.dumps({'passed': True, 'it': ck['it'], 'updates': ck['updates'], 'all_three_optimizers_updated': True}))
