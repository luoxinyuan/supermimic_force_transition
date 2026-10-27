# Full BONES tracking with a CEER-style public interface

This is a new SuperMimic policy trained from scratch on the full retained flat-ground BONES library. It does not load the old CEER checkpoint and is not binary-compatible with its 257-dimensional observation or motor wrapper. The reused part is the **public root/hand command semantics**; the learned world model, BPTT actor optimization, critic, replay and tracking rewards remain SuperMimic.

## Interface

At 50 Hz, `policy(proprio_history, command)` returns 29 normalized joint-target actions.

- Proprioception: 10 oldest-to-newest frames × 93 = 930. Each frame contains q−q0+encoder_bias (29), qd (29), body gyro (3), projected gravity (3), previous action (29).
- Command: root height (1), desired world velocity transformed into current body frame, XY only (2), desired heading in current yaw frame (2), reserved CEER force condition fixed to 5 (1), left/right hand positions in reference root frame (6), left/right hand axis-angle rotations in reference root frame (6). Total 18.
- Hand points reproduce the fixed wrist_yaw → hand_mimic offsets read from feet_reach's G1 USD (approximately +0.116 m along each wrist's X axis). FK uses SuperMimic's MuJoCo robot, independently checked against native MuJoCo.
- Training uses the **current** reference frame, matching the actual CEER command implementation. No future full-joint reference, phase, clip ID or reference XY position is passed to the actor.
- The full reference and velocities still supply the original whole-body tracking reward. There is no external force during this tracking run. The constant reserved force field is not a trained force-control capability.
- Original SuperMimic policy: command normalizer → MLP (2048,1024,512,256) → tanh latent 32; concatenate normalized proprioception → decoder (2048,1024,512,256) → 29 actions. Both encoder and decoder train; neither is frozen.

The old CEER deployment policy also estimates privileged state/joint references internally. Those estimators and its old weights are **not** transplanted: the new command encoder/decoder learn a sparse-command-to-whole-body mapping directly with SuperMimic BPTT.

Sparse hands/root commands do not uniquely specify arbitrary full-body motion, especially legs, root roll/pitch and acrobatics. This experiment therefore changes the conditioning task and cannot inherit the original paper's all-motion zero-failure claim. Training rewards remain dense whole-body rewards to learn a motion prior for the unspecified degrees of freedom.

## Data

Pinned BONES revision: `2f59b2077b9da34dd4e43618e705c7cb962c9a66`. Download hashes: `reports/full_tracking/download.json`.

All 142,220 CSVs convert from 120 Hz to 50 Hz: 51,903,579 frames, 288.353 h. Flat-ground filtering uses the repository's existing keyword and feasibility criteria; `filter_batched.py` batches FK without changing thresholds. The keyword list retains 129,115 motions, different from the paper's 129,667 before feasibility checks. The final library contains **128,904 motions / 47,032,428 frames / 261.291 hours**; feasibility rejects 126 elevated-support, 46 penetration and 39 floating-foot clips. Exact final counts/reasons and retained filenames are saved in `reports/ceer_full/filter.json` and `filter_clips.txt`; do not call this an exact paper split.

`prepare.py` converts, validates finite arrays/counts, filters and builds a root-local hand-command cache. `--resume-raw` reuses the already converted raw library. The full filtered library is resident on **each GPU**, not sharded; do not pass `clip-prefixes` with the full cache.

## Checks

- `test_interface.py`: 128 frames against native MuJoCo FK; common-world-yaw/XY-translation invariance.
- `smoke.py`: real three-rank simulation, world-model fitting and H=32 policy updates; finite/changed parameters, exact rank synchronization, LR synchronization, actor reference-joint/XY isolation, sim/imagination command agreement.
- Full-batch compiled smoke: 3 ranks, 4096 environments per GPU, WM batch 2048 × 10, policy batch 4096, H=32. Training reports finite-loss checks and memory per rank.
- `export.py`: export takes only proprioception + 18D command; validates numerical equivalence for batches 1 and 4. Export correctness is not policy behavioral acceptance.

## Run

From the repository root:

```bash
bash experiments/ceer_full/train.sh
# Resume the same experiment (same data/interface):
bash experiments/ceer_full/train.sh --resume runs/ceer_full_seed0/last.pt
```

Defaults: physical GPUs 2,3,4; 4096 environments per GPU; one synchronized policy; 320,000 iterations or 48 cumulative training hours, whichever first; checkpoint every 4000 iterations; fixed 256-clip diagnostic evaluation every 8000 iterations. Those 256 clips are monitoring only, not a held-out test or a full-library result. Checkpoint cfg records CEER conditioning. `train.py` aborts on nonfinite training statistics.

The launcher automatically evaluates `last.pt` on every retained clip after training and exports the low-level actor. Full-library results are saved as `evaluation_full_final.json`; the 256-clip monitoring scores are not substituted for this result. To repeat evaluation manually with the same CEER interface:

```bash
source env.sh
CUDA_VISIBLE_DEVICES=2 .venv/bin/python experiments/ceer_full/evaluate_final.py --checkpoint runs/ceer_full_seed0/last.pt
.venv/bin/python experiments/ceer_full/export.py --checkpoint runs/ceer_full_seed0/last.pt
```

The final exported `ceer_low_level.pt` requires the **SuperMimic** joint order, q0/action scales, PD parameters and observation convention. It cannot be substituted into the old feet_reach CEER motor wrapper without an explicit adapter.
