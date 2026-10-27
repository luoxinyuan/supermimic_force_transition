# Project: Force-Guided Whole-Body Motion Transitions with SuperMimic

## 1. Objective

We want to develop a new research project, **Force-Guided Whole-Body Motion Transitions for Humanoid Robots**, based on the SuperMimic framework.

The key idea is to train a general whole-body control policy that can transition between different physically feasible humanoid poses or motion states in response to externally applied forces.

Unlike traditional motion tracking, where a complete target trajectory is prescribed, we want the external force to act as an implicit motion guidance signal.

For example:

- A humanoid is standing.
- A person applies a downward force to its upper body.
- The humanoid actively cooperates with the force and transitions into a squatting pose.
- Once the force is removed, the humanoid should remain stable.
- An appropriate upward force can guide the humanoid back toward standing.

Eventually, we want to extend this behavior to many whole-body configurations, including standing, squatting, leaning, stepping, balancing on one foot, and potentially getting up from the ground.

**The external force is not merely a disturbance to reject. It is a signal that guides the robot's behavior.**

We want to exploit SuperMimic's learned differentiable world model and BPTT policy optimization to train this policy efficiently.

A critical requirement is that the final deployed student policy must NOT have access to ground-truth external forces. It must infer a task-relevant force latent from proprioception history.

The training architecture will be:

1. Force-aware differentiable world model.
2. Privileged force-conditioned teacher policy.
3. History-based latent force estimator.
4. Deployable student policy.
5. Teacher–student distillation, followed by optional student fine-tuning.

The goal is to build a working, research-ready implementation rather than just a conceptual prototype.

---

## 2. Starting Repository

Use the following repository:

https://github.com/dhmbb2/supermimic_vibe

Original paper:

https://arxiv.org/abs/2610.28480

The repository is an unofficial SuperMimic reproduction using mjlab and MuJoCo Warp.

It already contains:

- G1 humanoid simulation and motion tracking.
- Motion libraries and data preprocessing.
- Learned differentiable world model.
- Imagined rollout and BPTT policy training.
- Critic training and bootstrapping.
- PPO-Sim and PPO-WM baselines.
- Push recovery experiments.
- Evaluation scripts.
- Slurm training scripts.

Start by cloning the repository if it is not already present:

```bash
git clone https://github.com/dhmbb2/supermimic_vibe.git supermimic_force_transition
cd supermimic_force_transition
```

First inspect the actual codebase and documentation carefully.

In particular:

- `README.md`
- `notes/IMPLEMENTATION.md`
- `code/supermimic/`
- `code/scripts/`
- `slurm/`
- `requirements.txt`

Identify how simulator states, observations, actions, replay transitions, world-model inputs, rewards, imagined rollouts, and policy updates are implemented.

Do not assume module or function names before inspecting the code.

**Preserve the original SuperMimic implementation.** Add our task through new modules, configuration options, and carefully localized modifications.

The existing LAFAN1 tracking task must remain runnable without behavior changes.

Do not rewrite the SuperMimic training algorithm from scratch.

Use its existing world model, replay buffer, BPTT optimization, critic, and training infrastructure wherever possible.

---

## 3. Robot and Simulation

Target robot: Unitree G1 humanoid.

Initially use the existing 29-DoF G1 configuration supported by the repository.

Our eventual physical robot is a 23-DoF G1. However, prioritize getting the algorithm working on the original 29-DoF model first.

Do not assume that a 29-DoF policy can be deployed to a 23-DoF robot simply by dropping six output dimensions. Document the actuator, joint-order, model, and retargeting differences.

Use the repository's existing simulation backend, mjlab / MuJoCo Warp.

Maintain its existing control frequency, PD actuation conventions, simulator stepping, domain randomization, and world-model state representation unless changes are necessary.

All initial development and evaluation should be in simulation only.

---

## 4. Required Motion Data

### 4.1 Reproduce SuperMimic using LAFAN1

Use the G1-retargeted LAFAN1 dataset:

https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset

Start with the 40 G1-retargeted motion clips used in the original reproduction.

Use the existing data pipeline:

```bash
python code/scripts/prepare_motions.py \
    --format lafan \
    --fps-in 30 \
    --input data/raw/lafan1_g1 \
    --output data/processed/lafan1_50hz.npz
```

Verify whether the extracted dataset contains a nested `g1/` directory and adjust the input path accordingly.

Inspect frame rates, root orientation conventions, joint ordering, coordinate systems, and velocity computation.

The first milestone is reproducing normal motion tracking without any external force modifications.

Use the repository's documented commands and configurations for its original baseline.

### 4.2 Motion Transition Dataset

For our force-guided task, we do NOT initially need thousands of complete motion trajectories.

Instead, construct a dataset containing pairs of physically feasible robot states:

\[
\mathcal D_{\mathrm{transition}}
=
\{(s_A,s_B)\}
\]

Here:

- \(s_A\): initial robot state or pose.
- \(s_B\): candidate destination state or pose.

We do not require a reference trajectory between these states.

Initially implement one simple task:

**Standing ↔ Squatting**

Extract suitable poses from the motion dataset if available. Otherwise, generate physically feasible stable poses using joint-space design and/or inverse kinematics.

Validate both poses with the actual simulator:

- Respect joint limits.
- Avoid invalid contact geometry.
- Ensure the robot can remain balanced at the starting and ending poses.
- Validate ground contact and root height.
- Do not assume interpolating joint angles guarantees a dynamically feasible transition.

Provide a small configuration-driven pose-pair dataset interface so that additional transitions can be added later.

Future sources may include:

- LAFAN1.
- BONES-SEED.
- AMASS motions retargeted to G1.
- Our own motion retargeting pipeline.

BONES-SEED is optional and should not block the initial experiments.

### 4.3 Force Interaction Data

No prerecorded ground-truth force dataset is required.

Generate external forces directly in simulation.

During training, record:

\[
(s_t,a_t,F_t,c_t,s_{t+1})
\]

where:

- \(s_t\): full simulator state.
- \(a_t\): executed policy action.
- \(F_t\): applied external force.
- \(c_t\): force application metadata, including the affected body and contact location.
- \(s_{t+1}\): next simulator state.

Record proprioceptive observations and previous actions separately for student training.

All stored transitions must preserve correct time alignment between applied forces, actions, and resulting state changes.

---

## 5. External Force Simulation

Implement a general batched external-force application module.

The module must support:

- Applying forces to selected robot links.
- Specifying the force direction and magnitude.
- Specifying the point of application in the link's local frame.
- Computing the corresponding external wrench correctly.
- Multiple simultaneous force application points.
- Independent force schedules for different simulated environments.
- Force ramps, sustained forces, and force release.

Inspect mjlab / MuJoCo Warp APIs before choosing the implementation.

Use the simulator's supported batched external-wrench interface if available. If implementing generalized forces through Jacobians, verify that they are equivalent to the intended spatial wrench.

Do not approximate arbitrary contact-point forces by directly adding the same vector to joint torques.

Represent each force application as:

\[
u_i^F=(b_i,p_i,F_i,m_i)
\]

where:

- \(b_i\): body index.
- \(p_i\): contact point in the body-local frame.
- \(F_i\): applied force vector, with an explicitly defined coordinate frame.
- \(m_i\): validity mask.

The contact point may introduce a moment about the body's origin.

Support up to three force applications per environment, initially using only one.

Start with:

- Upper body or torso as the force application region.
- Moderate forces, initially around 5–30 N.
- Randomized force onset, duration, direction, and magnitude.
- Smooth onset and release, avoiding artificial impulses from instantaneous discontinuities.

The numerical ranges should be configurable and adjusted based on robot stability and the actual response.

Later, extend the module to arms, pelvis, thighs, and other body locations.

### Required validation

Test the force application independently of policy training.

Verify that:

1. Zero force preserves the original simulator dynamics.
2. Opposite forces produce physically consistent responses.
3. Different contact locations generate the expected torques.
4. Different magnitudes produce distinguishable dynamics.
5. Applied forces do not leak between simulation environments.
6. Recorded force and simulator-state timestamps are consistent.

Save short force-response trajectories for debugging.

---

## 6. Force-Aware Differentiable World Model

The original SuperMimic world model can be abstracted as:

\[
\hat s_{t+1}
=
f_\omega(s_t,a_t,\xi_t)
\]

Modify it to:

\[
\boxed{
\hat s_{t+1}
=
f_\omega(s_t,a_t,u_t^F,\xi_t)
}
\]

where:

- \(s_t\): full dynamics state.
- \(a_t\): policy action.
- \(u_t^F\): external force applications.
- \(\xi_t\): simulator context or domain randomization parameters.

The world model is privileged and training-only.

It is allowed to use ground-truth external forces.

### Force representation

Implement a reusable force feature encoder.

For each force application, consider features containing:

- Body identity embedding.
- Body-local contact location.
- Force vector in a clearly defined frame.
- Application mask.

Use a permutation-invariant aggregation method for multiple force applications, such as shared per-contact MLP embeddings followed by masked summation or averaging.

For the initial one-contact experiment, a simpler fixed-size representation is acceptable.

Keep the interface compatible with future multi-contact experiments.

### Training the world model

Reuse the original SuperMimic world-model training algorithm as much as possible.

Modify:

- Simulator transition collection.
- Replay-buffer transition schema.
- World-model input construction.
- Multi-step autoregressive prediction.
- Imagined rollout transition calls.

Mix nominal and force-disturbed transitions in replay.

Ensure that the replay buffer includes both:

- Dynamics without external forces.
- Dynamics under a variety of applied forces.

Do not train only with random force-perturbed states that the control policy never visits. Include policy-generated trajectories and refresh the replay buffer during training.

### Imagined rollout

An imagined rollout may use a sampled future force schedule as an exogenous disturbance:

\[
\hat s_{t+k+1}
=
f_\omega(\hat s_{t+k},a_{t+k},u_{t+k}^F,\xi)
\]

However, future force values are available only to the world-model transition function and the training environment.

Neither Teacher nor Student may observe future external forces through the policy input.

During imagined rollouts, maintain the same distinction between the force applied during the current transition and the force history observable at the next decision step.

The force sequence must be causally consistent.

### World-model evaluation

Evaluate predictions on held-out simulator transitions.

Report:

- One-step state prediction error.
- 4-step prediction error.
- 8-step prediction error.
- 16-step prediction error.
- Longer-horizon prediction where practical.
- Root position and orientation error.
- Joint position and velocity error.
- Prediction quality under unseen force combinations.

Compare predicted force responses against real simulator rollouts.

Check for model exploitation: a policy may achieve high imagined return by exploiting inaccurate world-model dynamics.

Make sure the original zero-force tracking accuracy is not significantly degraded.

---

## 7. Privileged Force-Conditioned Teacher

Implement a teacher policy with privileged access to applied-force information.

Architecture:

\[
z_t^T=E_T(u_{t-\delta}^F,\ldots)
\]

\[
\boxed{
a_t^T=\pi_T(o_t,g_t,z_t^T)
}
\]

where:

- \(E_T\): privileged force encoder.
- \(z_t^T\): teacher force latent.
- \(o_t\): robot observation.
- \(g_t\): public task or motion context.
- \(a_t^T\): joint action.

**Important: use a causal force-history convention.**

The Teacher should not depend on a newly applied force before the Student could possibly observe its physical effects.

Start with at least a one-control-step delay between force application and force information given to the Teacher, and make the delay configurable.

The Student must not be expected to imitate an action conditioned on an external force that has had no measurable effect on its observations.

The Teacher encoder should map privileged force information into a compact latent representation, rather than directly concatenating raw ground-truth force to the policy decoder.

A reasonable initial latent dimension is 32.

Reuse the existing SuperMimic policy decoder where practical.

### Teacher training

Train the Teacher using SuperMimic's differentiable world model, imagined rollouts, BPTT, and critic.

Preserve the original optimizer and training methodology unless a modification is necessary for the new task.

Do not replace the training framework with conventional PPO.

PPO will be used separately as a baseline.

---

## 8. Force-Guided Motion Transition Task

This is the most important task-design component.

The goal is not simply to reject forces or track a predefined trajectory.

The policy should learn to **actively cooperate with external forces to transition between stable whole-body configurations**.

### Initial task

Use a standing and a squatting configuration:

\[
s_A=\text{Standing}
\]

\[
s_B=\text{Squatting}
\]

Desired behavior:

| Current state | Force condition | Desired behavior |
|---|---|---|
| Standing | No force | Remain standing |
| Standing | Suitable downward force | Actively transition toward squatting |
| Squatting | No force | Maintain a stable squatting configuration |
| Squatting | Suitable upward force | Actively transition toward standing |

The exact force direction, point of application, and required magnitude should be validated with the simulator.

### Critical requirement: no target leakage

Do not simply switch the policy's explicit target pose from Standing to Squatting when the training environment applies a force.

If the policy receives that switched target as an input, it could perform the transition without sensing the force.

That would defeat the main purpose of this project.

The Teacher and Student may receive a common, static representation of the available motion states or a common task context.

However, the selected transition destination must not be directly exposed as an additional command.

The force-induced physical interaction must carry the information that determines the transition.

Privileged transition labels may be used to compute the training reward, but they must not appear in Student policy inputs.

### Reward design

We only have starting and destination states, not a dense reference trajectory.

Therefore, build a reward that encourages:

1. Progress toward the force-indicated destination.
2. Stable completion of the transition.
3. Staying balanced.
4. Smooth, physically plausible joint motions.
5. Avoiding unnecessary effort and excessive joint velocities.
6. Maintaining appropriate contact with the ground.
7. Remaining stable in the absence of external forces.

An initial reward can be formulated as:

\[
r_t =
w_p r_{\mathrm{progress}}
+w_t r_{\mathrm{target}}
+w_s r_{\mathrm{stability}}
+w_c r_{\mathrm{contact}}
-w_v c_{\mathrm{velocity}}
-w_a c_{\mathrm{action}}
\]

Use a progress reward based on reduction in a task-relevant pose distance, plus a terminal target reward.

Avoid rewarding the policy solely for instantly reaching the target.

Avoid letting the policy achieve high reward by collapsing, sliding uncontrollably, or using unrealistic forces.

Design the target reward to be consistent with the current applied force and the desired transition direction.

Use explicit force thresholds and hysteresis only where necessary.

If the training reward maintains an internal transition state or latch, consider how that hidden state affects observability. Do not rely on an unobservable target mode unless the Student has sufficient history or recurrent memory to infer it.

At zero force, prefer stable pose maintenance rather than automatically returning to a predefined standing pose.

### Key distinction from motion tracking

Normal dense tracking:

\[
r_t=r_{\mathrm{tracking}}(s_t,s_t^{ref})
\]

Force-guided transition:

\[
r_t=r_{\mathrm{transition}}(s_t,s_A,s_B,u^F)
\]

There is no prescribed intermediate reference trajectory.

The robot must learn how to move between the states under external force guidance.

### Curriculum

Implement a configurable curriculum:

**Level 1: Single transition**

- Standing to squatting.
- Fixed force application point.
- Fixed or narrowly randomized direction.
- Moderate force magnitude.
- Long enough force duration to make the interaction observable.

**Level 2: Bidirectional transitions**

- Standing to squatting.
- Squatting to standing.
- Different force directions and magnitudes.
- Zero-force hold behavior.

**Level 3: Generalized force input**

- Random force direction.
- Random force magnitude.
- Different upper-body application points.
- Variable force onset and duration.

**Level 4: Multiple transition pairs**

- Add additional feasible motion-state pairs.
- Sample different initial configurations.
- Randomize contact locations and force combinations.
- Evaluate transitions not seen in training.

Do not start by implementing or training all curriculum levels simultaneously.

Build and validate Level 1 first.

---

## 9. Student Policy: Sensorless Force Latent

The final Student must NOT receive true applied force, contact-point labels, or any equivalent privileged force information.

The Student will infer a force latent from proprioception history.

Define:

\[
\boxed{
z_t^S=E_H(h_t)
}
\]

where:

\[
h_t=
(o_{t-H:t},a_{t-H:t-1})
\]

Then:

\[
\boxed{
a_t^S=\pi_S(o_t,g_t,z_t^S)
}
\]

The Student should observe only signals that can realistically be obtained on a physical Unitree G1.

Potential inputs:

- Joint positions.
- Joint velocities.
- IMU angular velocity.
- Projected gravity or IMU orientation features.
- Previous policy actions.
- Public task/motion context.

Do not include:

- Ground-truth external force.
- Ground-truth contact location.
- Simulator-only contact flags.
- Global root linear velocity unless a deployable estimator is explicitly provided.
- Future force information.
- Hidden transition labels or force-derived target commands.

Start with a history length of approximately 10–20 control steps, configurable in the training configuration.

Use an MLP history encoder initially, or reuse an existing history-encoding architecture if available.

A GRU can be added if finite-window history is insufficient for maintaining transition state after the force is released.

### Teacher latent

\[
z_t^T=E_T(u^F_{\mathrm{history}})
\]

### Student latent

\[
z_t^S=E_H(h_t)
\]

The objective is not necessarily to estimate an exact force vector.

Instead, the latent should capture force-related information useful for controlling the robot.

Note that exact force magnitude and contact location may not be uniquely identifiable from proprioception, especially during ambiguous contacts.

Use a compact, behavior-relevant latent, and evaluate whether it is sufficiently observable.

---

## 10. Teacher–Student Distillation

First train a privileged force-conditioned Teacher.

After Teacher training, implement Student distillation.

### Latent matching

\[
\mathcal L_z=
\|z_t^S-\operatorname{sg}(z_t^T)\|_2^2
\]

### Action matching

\[
\mathcal L_a=
\|a_t^S-\operatorname{sg}(a_t^T)\|_2^2
\]

### Combined loss

\[
\mathcal L_{\mathrm{student}}
=
\lambda_z\mathcal L_z+
\lambda_a\mathcal L_a
\]

Initially:

- Freeze the trained Teacher.
- Reuse the Teacher's policy decoder if architectural compatibility allows.
- Train the Student history encoder to predict the Teacher's task-relevant latent.
- Use action distillation to verify that the Student can reproduce Teacher behavior.

After this works, optionally fine-tune the decoder.

### On-policy Student refinement

Pure offline distillation may lead to compounding errors.

Implement optional DAgger-style refinement:

1. Roll out the Student in the physical simulator.
2. Record Student-visited states and observations.
3. Query the privileged Teacher at those states.
4. Add the observations, Teacher latents, and Teacher actions to the distillation dataset.
5. Retrain the Student on the expanded dataset.

The Teacher and Student must be queried on the same physical state and consistent past force/action histories.

Optionally explore Student policy fine-tuning with SuperMimic's BPTT optimization after distillation is stable.

During Student imagined rollouts, the world model may still use true simulated external forces, but the Student policy must only receive its own observation history and inferred latent.

This last step is optional for the first implementation.

---

## 11. Evaluation

Create a dedicated evaluation script for force-guided transitions.

At minimum, evaluate:

### Transition behavior

- Standing → Squatting success rate.
- Squatting → Standing success rate.
- Transition completion time.
- Final pose error.
- Falling rate.
- Stability after force release.

### Force responsiveness

- Success under different force magnitudes.
- Success under different force directions.
- Success under different application points.
- Response latency following force onset.
- Whether the robot actively contributes to the transition rather than being passively displaced.

### Student quality

- Teacher success rate.
- Student success rate.
- Teacher–Student action discrepancy.
- Teacher–Student latent discrepancy.
- Performance under unseen force conditions.
- Zero-force false activation rate.

### World-model quality

- Prediction error with and without external forces.
- Multi-step prediction accuracy.
- Generalization across force magnitudes and locations.
- Actual simulator performance versus imagined rollout performance.

### Efficiency

Compare the following methods under the same force-guided transition task:

1. PPO-Sim with privileged force-conditioned Teacher.
2. SuperMimic with privileged force-conditioned Teacher.
3. SuperMimic Student with inferred force latent.
4. Force-blind policy without latent conditioning.

Reuse the repository's existing PPO-Sim implementation where possible.

The primary efficiency metrics are:

- Training GPU-hours to reach a specified success rate.
- Simulator interaction steps to reach that success rate.
- World-model training time.
- Final policy success rate.

Do not claim a sample-efficiency or compute-efficiency improvement without measuring it on the same task with comparable resources.

---

## 12. Required Software Deliverables

Implement and document:

1. External-force simulation module.
2. Force-conditioned replay-buffer support.
3. Force-aware world model.
4. Force-guided transition environment.
5. Pose-pair dataset loader and validation.
6. Privileged force encoder.
7. Teacher policy training pipeline.
8. Student proprioceptive history encoder.
9. Student distillation and optional DAgger refinement.
10. Transition evaluation scripts.
11. Simulation visualization and recorded videos.
12. Slurm scripts for training and evaluation.
13. Separate configuration files for each training stage.

The exact file structure should follow the existing repository conventions.

Maintain a clean separation between:

- Original SuperMimic tracking.
- Force-aware world-model training.
- Force-guided Teacher training.
- Student distillation.
- Student evaluation.

Provide an `experiments/` or equivalent configuration directory for reproducible experiments.

Document all newly introduced observation dimensions and their precise meanings.

Add automated unit tests for force application, tensor shapes, replay time alignment, observation privilege boundaries, model prediction, and checkpoint loading.

---

## 13. Implementation and Validation Plan

Work incrementally. Do not attempt to implement the full research system without testing intermediate components.

### Milestone 0 — Environment and Baseline

- Clone and inspect the repository.
- Set up the documented Python/mjlab environment.
- Download and preprocess the 40 G1 LAFAN1 clips.
- Run the existing unit tests and a minimal training/evaluation smoke test.
- Verify that the original SuperMimic implementation works.
- Document the exact repository commit and environment versions.

Acceptance: original training, world-model updates, and evaluation run without newly introduced errors.

### Milestone 1 — External Force Infrastructure

- Implement batched link-level force application.
- Add force scheduling and force logging.
- Add replay-buffer fields.
- Run physical force-response tests.

Acceptance: external forces produce correct and repeatable robot responses, with no changes to the zero-force behavior.

### Milestone 2 — Force-Aware World Model

- Add force conditioning to world-model inputs.
- Train on a mixture of nominal and disturbed simulator transitions.
- Validate autoregressive prediction under applied forces.
- Verify differentiability with respect to states and actions.

Acceptance: the model predicts different outcomes for different applied forces, and performs meaningfully better than a force-blind predictor on disturbed held-out transitions.

### Milestone 3 — Force-Guided Teacher

- Create stable standing and squatting poses.
- Implement the Level-1 transition environment.
- Implement privileged force encoder and policy integration.
- Train using SuperMimic imagined rollouts and BPTT.
- Evaluate true simulator rollouts.

Acceptance: the Teacher demonstrates force-responsive transition behavior in simulation without relying on a directly exposed target-switch command.

### Milestone 4 — Student Distillation

- Implement the proprioception history encoder.
- Train latent and action matching.
- Roll out the Student in simulation.
- Check for privileged-information leakage.
- Implement optional DAgger refinement if needed.

Acceptance: the Student performs force-guided transitions using only deployable observations, without reading true external forces.

### Milestone 5 — Extended Experiments

- Bidirectional standing/squatting transitions.
- Variable force magnitude and contact locations.
- No-force stability tests.
- Generalization experiments.
- PPO comparison.
- Optional additional pose transitions.

Acceptance: quantitative evaluation demonstrates the behavior, limitations, and training efficiency of the proposed system.

---

## 14. Important Engineering Requirements

- Preserve the original tracking pipeline.
- Do not modify original baseline defaults unless necessary.
- Keep force-related functionality configurable.
- All privileged force variables must be explicitly identified.
- Do not leak simulator force values through the Student's observation, goal encoding, normalizer, recurrent state, or policy inputs.
- Keep world-model force conditioning separate from policy force inference.
- Make force coordinate frames and timestamps explicit.
- Avoid introducing target-switch commands that bypass force inference.
- Avoid using complete reference transitions for the initial force-guided task.
- Reuse existing checkpointing, replay buffers, optimizers, and infrastructure.
- Support checkpoint resume.
- Use the available GPU environment responsibly.
- Run small smoke tests before expensive experiments.
- Do not automatically launch long training jobs until the corresponding module passes its smoke tests.
- Do not silently change the physics backend to another simulator.
- Do not create simulated success reports or fabricate metrics.
- Do not commit large datasets or model checkpoints.

If the desired architecture is incompatible with an existing implementation detail, explain the conflict and make the smallest justified modification.

---

## 15. Execution Instructions

Please start implementing this project in the repository.

First:

1. Inspect the codebase thoroughly.
2. Produce a concise architecture map showing where each modification belongs.
3. Set up and verify the original SuperMimic baseline.
4. Begin implementing Milestones 1 and 2.
5. Run available tests and fix implementation errors.

Then proceed to the Teacher and Student implementation incrementally.

**Do not stop after producing a plan or pseudocode. Make actual code changes and run the tests available in the environment.**

If data, GPU access, dependencies, or external permissions are unavailable, document the precise blocker, finish all work that can be tested without them, and report the exact remaining commands.

At each milestone, summarize:

- Files created or modified.
- Architecture changes.
- Commands executed.
- Tests passed or failed.
- Outstanding issues.
- Exact command to run the next experiment.

The ultimate deliverable is a reproducible force-guided transition training pipeline built on SuperMimic, with a privileged Teacher and a sensorless latent-conditioned Student.