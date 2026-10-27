# Experiment 3: sensorless latent student

From the project root, `source experiments/exp02/env.sh`.

The student consumes only the existing 10-frame proprioceptive observation (930 values) and a causal GRU hidden state (128 values). No true force, applied-force schedule, reward target, reference pose geometry, world-model context, or root position enters its forward function. Reset the hidden state to zero only on episode/robot reset; preserve it when force stops.

The 256-unit input MLP and 128-unit GRU predict the **total** 32-dimensional teacher control latent. This includes the nominal posture latent, because retaining the teacher reference encoder would leak privileged root geometry. The frozen teacher decoder and observation normalization are reused. This is behavioral latent estimation, not a calibrated force sensor.

Training uses latent MSE plus action MSE through the frozen decoder, 50-step truncated BPTT, batches of 16 sequences, 500 steps per episode, 64 environments per collection, two initial poses, 50% zero force, 15–30 N vertical torso forces, random onset/duration. Teacher labels are queried at the same state and past executed-action history. Terminated samples are masked. Data aggregation preserves earlier collections.

Initial run:

```bash
.venv/bin/python experiments/exp03/run.py --rounds 4 --epochs 12 --lr 3e-4 --action-weight .1
```

Corrective run starts from the stable-but-nonresponsive first DAgger checkpoint and gradually increases student action weight from 0.2 to 1.0:

```bash
.venv/bin/python experiments/exp03/run.py --resume runs/exp03_student/round1.pt --out runs/exp03_refine --rounds 9 --epochs 16
.venv/bin/python experiments/exp03/evaluate.py --checkpoint runs/exp03_refine/last.pt
.venv/bin/python experiments/exp03/audit.py --checkpoint runs/exp03_refine/last.pt
.venv/bin/python experiments/exp03/plot.py
```

Evaluation uses a held-out schedule seed and reports no-force endpoint retention, force-conditioned endpoint change, termination count, and same-state teacher/student action and latent discrepancy. A positive difference caused by a no-force collapse does **not** pass. Require both directions, >4.5 cm signed height change, <2.5 cm no-force endpoint drift, and zero terminations. The final held-out test is authoritative; training diagnostics alone are not acceptance.

The exported `deploy.pt` has only observation and hidden-state inputs. Export validates interface and numerical equivalence, not behavior or real-robot readiness. Current tests use clean observations, no domain randomization, and simulation; they do not establish hardware deployment suitability.
