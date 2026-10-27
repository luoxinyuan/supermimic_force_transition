# Architecture and experiment scope

Upstream: https://github.com/dhmbb2/supermimic_vibe, commit `072f0910dadc9d95c720e625b546120e9ce917c9`.
The current request is experiment 1 only: original G1 29-DoF tracking on all 40 LAFAN1 clips. Force-guidance, teacher and student are future milestones; no force features are enabled here.

| Component | Actual implementation | Future extension point |
|---|---|---|
| Dataset | `code/scripts/prepare_motions.py`, `motion_lib.py` | Separate pose-pair library |
| Simulation | `sim_env.py:TrackingSimEnv.step/read_state/write_state` | Optional batched wrench module before each physics transition |
| State | `task.py:state_to_flat/flat_to_state`: 71 numbers | Keep nominal state, store force metadata separately |
| Observations | `Task.obs_clean`, `TrackingSimEnv.policy_inputs`: 10×93 history, 10×73 future reference | Separate causal teacher/student inputs; no destination-switch input |
| Actions | 29 joint targets `q0 + scale*a - encoder_bias`; 5 ms physics × 4 | Preserve actuator mapping |
| Replay | `replay.py:ReplayBuffer.add/sample_windows/sample_starts` | Force-conditioned subclass with aligned force/contact schedule |
| World model | `networks.py:WorldModel`, normalized 70-D state delta | Force encoder in model inputs |
| Reward/termination | `task.py:Task.reward/analytics_fast` | Separate transition task; preserve tracking reward |
| Collection/WM updates | `train.py:Trainer.collect/update_wm/wm_window_loss` | Subclass collection and autoregressive loss |
| BPTT | `Trainer.update_policy` | Must update direct concatenation/MLP call as well as `WorldModel.step`; imagination bypasses `step` |
| Critic | `Trainer.update_critic`, lambda returns | Reuse with new public task context |
| Benchmark | `evaluate.py:evaluate` | Keep all-clip deterministic original benchmark |

Data convention: G1 CSV at 30 Hz, XYZ in meters, quaternion XYZW and 29 joint angles in dataset order. Conversion uses quaternion sign continuity, SLERP to 50 Hz, WXYZ storage, linear interpolation of position/joints, centered differences for world linear and joint velocities, quaternion differences for world angular velocity. MuJoCo base angular velocity is local; `read_state/write_state` explicitly convert to/from world coordinates. No axis remapping is introduced. FK and model joint ordering are checked against the actual simulator.

Do not deploy this 29-DoF policy to a 23-DoF robot by dropping outputs: actual robot model, joint names/order, actuator gains/limits and retargeting need a separate audit and training configuration.
