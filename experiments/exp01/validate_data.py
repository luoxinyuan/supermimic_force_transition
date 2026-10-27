"""Check the actual converted 40-clip library against upstream metadata."""
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[2]
p = root / 'data/processed/lafan1_50hz.npz'
d = np.load(p)
assert len(d['names']) == 40
assert d['qpos'].shape == (441120, 36), d['qpos'].shape
assert d['qvel'].shape == (441120, 35)
assert float(d['fps']) == 50
assert np.isfinite(d['qpos']).all() and np.isfinite(d['qvel']).all()
assert np.all(d['lengths'] > 1)
assert np.array_equal(d['starts'], np.r_[0, np.cumsum(d['lengths'])[:-1]])
assert d['lengths'].sum() == len(d['qpos'])
assert len(set(d['names'].tolist())) == 40
qerr = np.abs(np.linalg.norm(d['qpos'][:, 3:7], axis=1) - 1).max()
assert qerr < 1e-5
for name in d['names']:
    raw = np.loadtxt(root / 'data/raw/lafan1_g1/g1' / str(name), delimiter=',')
    assert raw.ndim == 2 and raw.shape[1] == 36 and np.isfinite(raw).all()
report = dict(clips=40, frames=len(d['qpos']), fps=50, hours=len(d['qpos'])/50/3600,
              quaternion_norm_max_error=float(qerr), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
              clip_names=d['names'].tolist())
(root / 'reports/exp01_data.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
