# Experiment 1: original SuperMimic / LAFAN1-40

Project root: `/home/xl521/supermimic_force_transition`.
No external force modifications. G1 29 DoF, mjlab / MuJoCo Warp, seed 0.
`env.sh` selects host GPU 1 by default; override with `EXP01_GPU=5` if needed.

## Rebuild and verify

```bash
cd /home/xl521/supermimic_force_transition
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r experiments/exp01/requirements.lock.txt
source experiments/exp01/env.sh
python3 experiments/exp01/download_lafan.py
mkdir -p data/processed
"$SMIMIC_PY" code/scripts/prepare_motions.py --format lafan --fps-in 30 \
  --input data/raw/lafan1_g1/g1 --output data/processed/lafan1_50hz.npz
"$SMIMIC_PY" experiments/exp01/validate_data.py
"$SMIMIC_PY" experiments/exp01/validate_sim.py
bash experiments/exp01/smoke.sh
"$SMIMIC_PY" experiments/exp01/check_run.py runs/exp01_smoke
```

Smoke uses all 40 clips, the original network sizes, H=32, k=8, critic and compilation;
only environment/batch counts, world-model updates per iteration and duration are reduced.
A 10-iteration untrained policy is expected to fail clips; this tests execution, not tracking quality.

## Full baseline

```bash
bash experiments/exp01/train.sh
# Optional extension to the upstream 12-hour run (automatically resumes):
bash experiments/exp01/train.sh --wall-hours 12
```

The initial experiment has a 3-hour **trainer-accounted time** budget. Warmup, environment
setup and evaluation add wall-clock time; compilation inside an update is included in
trainer time. Cold-cache timing is therefore not directly comparable to upstream timings. All algorithm defaults remain upstream defaults: 4096 envs,
1000-step replay, WM 10×2048 windows of k=8, 4096 imagined rollouts × H=32, UTD=0.4.
Evaluation/checkpoint every 4000 iterations, all 40 clips from frame 0, friction 1,
no observation/reset noise or domain randomization in benchmark mode.

```bash
tail -f runs/exp01_lafan40_seed0/console.log
cat runs/exp01_lafan40_seed0/eval.jsonl
source experiments/exp01/env.sh
"$SMIMIC_PY" -m supermimic.evaluate --run runs/exp01_lafan40_seed0 --num-envs 40 --out eval_standalone.jsonl
```

Standalone evaluator reads `ckpt_*.pt`, while the trainer evaluates final `last.pt` in memory.
The standalone evaluator uses FP32 MLP inference; the trainer follows upstream BF16.
Do not merge their metrics without labeling the evaluation mode.

Acceptance levels: (1) finite simulator and FK checks, (2) WM/BPTT/critic update and
checkpoint/resume tests, (3) meaningful all-clip tracking benchmark after training.
Upstream reports 0.833 at 1.34 training hours and 0.946 / 0 failures at 12 h on an A6000;
these are reference results, **not results of this run**. Inspect local `eval.jsonl`.

The raw manifest pins HF revision and hashes. The environment lock, architecture map,
original specification and validation outputs are retained under `experiments/`, `notes/`,
and `reports/`. Data, checkpoints, virtualenv and caches are ignored by git.

## Resume repair

Upstream checkpoints omit replay and simulator/RNG state. Upstream `Trainer.run` only
prewarmed at iteration zero, so resuming would sample an empty buffer. The localized
repair refills replay before updates on resume and clears the stale `DONE` marker while
running. Default fresh training still collects exactly 64 warmup steps. Resume retains
networks, normalizers, optimizers, iteration and sampling statistics; it is not bitwise
continuation because replay/simulator/RNG state are not checkpointed.

```bash
bash experiments/exp01/resume_smoke.sh
```
