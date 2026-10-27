# Experiment 2 — Force-aware Teacher

This opt-in package is `code/supermimic/force_transition/`. Original tracking is unchanged.
Baseline initialization is **ckpt_0036000.pt**, normalized return 0.90974, 0/40 failures;
`last.pt` is not the best baseline (0.89115, 1/40 failures).

## Causal interface

- Contacts: up to 3 slots `[global_body_id, local_point_xyz, world_force_xyz, mask]`.
- Force is constant during the 20 ms control interval. Moment is recomputed at each
  5 ms substep about the body's COM: `(world_point - xipos) cross F`.
- Replay stores pre-reset `(s_t, a_t, s_next, physical_context, current_contacts, schedule,
  reward_target, force_memory, delayed_force_queue, elapsed_step)` with episode identity.
- World model input: original state features, executed action, physical context, 24 scaled
  contact features. Every autoregressive step reads that transition's own force.
- Teacher: 930 proprioception history values, delayed 24 contact features, 3 causal
  low-pass force-history values, and **both fixed endpoints** (2×730) in a fixed order.
  The reward target, schedule and future force never enter its input.
- The force history filter is driven only by delayed actual force, not destination labels.
  Delay is configurable and >= 1. A future Student will require memory capable of matching
  this history; this experiment makes no sensorless/deployment claim.

## Policy and optimization

The original motion encoder/decoder are frozen as a stable nominal controller. The
nominal latent mixes the two **static** endpoint latents using observed knee posture.
A trainable 32-dimensional force-latent residual changes the latent fed through the
original frozen decoder. With no current/past force the residual is identically zero.
The policy never receives a selected destination; its residual must learn the relation
between interaction forces and motions through rewards.

This is a deliberately constrained Teacher for the first experiment, not a newly trained
universal decoder. It establishes whether learned force conditioning can drive the
existing controller between stable states. BPTT gradients pass through the frozen decoder
and learned WM; the original critic fitting, optimizer loop, replay observation history,
normalizers, checkpointing and learning-rate schedule are reused.

The initial static behavior-cloning approach was rejected after closed-loop failure.
Its artifacts are retained in `runs/exp02_smoke/` for audit; use `exp02_smoke_v2` instead.

## Endpoints and rewards

`poses.json` defines two 29-DoF poses, designed with symmetric legs, level feet and COM
alignment. The reproduced tracker held both for 10 s in all 64 validation environments.
The nominal achieved heights are about 0.783 m / 0.706 m: this is a **shallow squat**.
No intermediate motion trajectory is used for transition rewards.

Reward terms: destination pose and height, reduction of pose distance, upright posture,
balance, foot-height/contact surrogate, foot-slip penalty, joint velocity, action change,
effort and joint limits. Reward-only mode latches when |vertical force| > 3 N. At zero
force it retains the mode, including the initial squat mode. The Teacher does not read it.
Foot-height and velocity are differentiable contact surrogates, not contact-force sensors.

Initial curriculum: fixed torso point, vertical 5–30 N, smooth 0.5 s ramps, randomized
onset/duration, 25% nominal episodes, random standing/squatting starts and optional reverse
pulse. Domain randomization is disabled initially; simulation dt/PD remain original.

## Commands

```bash
cd /home/xl521/supermimic_force_transition
source experiments/exp02/env.sh   # host GPU 1; EXP02_GPU overrides
"$SMIMIC_PY" experiments/exp02/prepare_poses.py
"$SMIMIC_PY" experiments/exp02/validate_poses.py
"$SMIMIC_PY" experiments/exp02/test_force.py
"$SMIMIC_PY" experiments/exp02/test_zero_force.py
# Run the small end-to-end test before checkpoint/gradient verification:
"$SMIMIC_PY" -m supermimic.force_transition.train --out runs/exp02_smoke_v2 \
  --num-envs 64 --buffer-steps 256 --wm-batch 64 --wm-batches 1 --pi-batch 64 \
  --iterations 10 --log-every 5 --eval-every 10 --ckpt-every 10 \
  --wm-only-updates 0 --sim-compile false --compile-wm false
"$SMIMIC_PY" experiments/exp02/test_model.py

# World-model warmup before Teacher optimization:
"$SMIMIC_PY" -m supermimic.force_transition.train --out runs/exp02_wm_warmup \
  --num-envs 256 --wm-batch 256 --pi-batch 256 --iterations 1000 \
  --wm-only-updates 100000 --eval-every 0 --ckpt-every 500 --wall-hours 1
"$SMIMIC_PY" -m supermimic.force_transition.evaluate_wm \
  --checkpoint runs/exp02_wm_warmup/last.pt --out reports/exp02/wm_validation.json

# After validating force-aware prediction, train Teacher:
"$SMIMIC_PY" -m supermimic.force_transition.train --out runs/exp02_teacher_seed0 \
  --resume runs/exp02_wm_warmup/last.pt --num-envs 256 --wm-batch 256 --pi-batch 256 \
  --iterations 100000 --wm-only-updates 0 --eval-every 1000 --ckpt-every 1000 --wall-hours 2
"$SMIMIC_PY" -m supermimic.force_transition.evaluate \
  --checkpoint runs/exp02_teacher_seed0/last.pt --out reports/exp02/teacher_evaluation
```

Resume retains the original cumulative iteration and training-time counters. The 2-hour
budget above includes the preceding WM warmup trainer time. Replay is recollected.

## Acceptance

1. Correct point-force wrench, direction/magnitude, per-env isolation, replay time alignment,
   target/schedule input isolation, and force delay (including memory) tests.
2. Fresh held-out simulator trajectories, 1/4/8/16-step WM error; forced one-step MSE must
   improve >=5% over the **same model with its force input blinded**. Includes force signs
   at both endpoints not restricted to training's direction/start correlation.
3. ONE checkpoint tested from each starting pose with no force, a 25 N pulse, a pulse with
   Teacher force/history blinded, and latent signal with physical force suppressed.
4. At least 4.5 cm sustained height change in the expected direction, knee change >0.25 rad,
   action difference, no falls, destination RMSE <0.18 rad, and positive active-motion
   advantage in both counterfactuals. Evaluate over 8–10 s, after force release at 4.5 s.

Four replicas per deterministic case are numerical repeats, **not independent statistical
trials**. Later robustness evaluation must vary seeds, magnitudes, directions and contact
locations. Small smoke tests do not establish successful force-guided transitions.

The endpoint geometry uses simulator root pose to construct relative descriptors. This is a
privileged Teacher implementation; a future proprioception-only Student must estimate or
replace this geometry input. The current Teacher is not a deployment policy.

## Accepted checkpoint and observed limits

`runs/exp02_teacher_seed0/ckpt_0002000.pt` passed the paired real-simulator core test
and continuous cycles at 15/25/30 N. Training was stopped after that milestone; see
`reports/EXPERIMENT_02.md`, `reports/exp02/acceptance.json`, and `teacher_cycles.mp4`.
10 N rising is unreliable; sustained downward force has squat-depth overshoot.
This is a mechanism demonstration in the stated setup, not broad robustness certification.

```bash
"$SMIMIC_PY" experiments/exp02/evaluate_cycles.py
"$SMIMIC_PY" experiments/exp02/render_cycles.py
```

The final calibrated checkpoint is `runs/exp02_wm_refine/last.pt`; its policy is bitwise
identical to the accepted Teacher, with additional WM-only calibration on transition
rollouts. On fresh held-out transitions the one-step force-aware MSE is 0.6031× the
force-blind ablation. See the experiment report for exact refinement commands.
