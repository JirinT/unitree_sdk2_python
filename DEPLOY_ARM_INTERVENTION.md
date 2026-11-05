# H1_2 arm-intervention playback in Unitree MuJoCo

Use `deploy_arm_intervention.py` for the new **Unitree-H1_2-Flat-ArmIntervention**
checkpoint: **108 actor observations -> 27 actions**. The 112-input
Flat-HandTracking checkpoint still uses `deploy_sim_teleop.py` and its original
YAML. `evaluate_sim.py` / `demo_sim.py` remain the original Cartesian hand-tracking
evaluators; they are not the entry point for this new actor.

The new adapter is simulator-only. It always initializes DDS domain **1** on
interface **lo**, defaults to `simulation`, and rejects hardware network-interface
arguments. No IK, Torch or mjlab dependency is needed at runtime. Install/use the
same `numpy`, `pyyaml`, `mujoco`, `onnxruntime`, Unitree SDK and CycloneDDS Linux
environment as the existing simulator adapter.

Keep the training repo as a sibling of `unitree_sdk2_python` on Linux so the YAML's
relative `robot_xml` resolves, or edit it / pass `--robot-xml`. Keep its mesh assets.
Use the training robot geometry, joint names, default pose, gains and action scales.
The bridge must publish H1_2 HG LowState/LowCmd messages and its IMU frame must
match `imu_quaternion_frame`. The new runner inherits the old adapter's IMU-to-
pelvis gravity conversion, joint/motor mapping, 4-second pose ramp, 50 Hz policy
inference and 500 Hz low-command publication.

## First validate without DDS

From `unitree_sdk2_python`, with your newly trained/exported policy:

```bash
python deploy_arm_intervention.py deploy_arm_intervention.yaml \
  /path/to/h1_2_arm_intervention/policy.onnx --validate-only
```

This loads the model/config/kinematics and performs an offline neutral inference.
It never initializes SDK channels and does not require a terminal TTY. The model
must carry the `intervention_v1` export metadata added by the new training runner;
re-export an older model if its metadata is missing. Wrong dimensions, arm order,
contract, trajectory limits and mismatched exported gain/scale/pose metadata are
rejected before connecting. Flat actors only; rough actors require terrain inputs.

## Deterministic demonstration

Start the separate Unitree MuJoCo H1_2 validator with domain 1 / loopback first:

```bash
python deploy_arm_intervention.py deploy_arm_intervention.yaml \
  /path/to/h1_2_arm_intervention/policy.onnx --arm-control demo
```

It repeats the same 32-second sequence as `play.py --arm-control demo`:
stand -> reach -> carry -> walk forward at 0.3 m/s -> stop -> place gesture ->
neutral -> release arms to PPO. These are unloaded joint-space gestures; there is
no object, grasping or Cartesian hand-position task. Demo time starts after the
4-second startup ramp. Add `--duration 32` for one cycle and exit (about 36 seconds
including startup, excluding the LowState wait). No keyboard or interactive TTY
is required for demo mode. Ctrl+C also exits.

`--arm-control policy` runs standing with PPO controlling all 27 joints. For
walking with PPO-owned arms, use keyboard mode, leave both masks off, and issue
walking commands. Random training interventions are not implemented in this
SDK adapter; use `play.py` for that mode.

## Keyboard evaluation

```bash
python deploy_arm_intervention.py deploy_arm_intervention.yaml \
  /path/to/h1_2_arm_intervention/policy.onnx
```

Keyboard mode is the default. Focus the launching Linux terminal; tap keys without
Enter. SSH requires an interactive TTY. Walking commands remain latched.

| Keys | Action |
|---|---|
| W/S | Forward/backward command, +/-0.1 m/s |
| Q/E | Strafe left/right command, +/-0.1 m/s |
| Z/X | Yaw left/right command, +/-0.2 rad/s |
| B or Space | Stop walking |
| 1 / 2 / 3 | Select left / right / both arms |
| 4 / 5 | Toggle left / right external takeover |
| 0 | Smoothly return both arms to PPO |
| 6 or C / 7 / 8 / 9 | Neutral / reach / wide / carry pose |
| [ / ] | Select previous / next arm joint |
| + or = / - | Adjust selected joint by +/-0.1 rad |
| Ctrl+C | Exit and restore terminal settings |

Arms start under PPO control. Presets/edits enable the selected arm(s); unselected
arm goals stay unchanged. Joint order is shoulder pitch, shoulder roll, shoulder
yaw, elbow, wrist roll, wrist pitch, wrist yaw. Edits are clamped to the training
range/default-pose envelope and soft limits. Walking keys do not restart the arm
path. New arm commands schedule waypoints, rather than restarting them every
policy tick. Rapid repeated edits can interrupt acceleration continuity; the
reference speed limit remains applied. The release transition is also speed-
limited. These limits govern references, not actual measured motor acceleration.

## Controller/observation contract

The NumPy arm mux reproduces the training action term's deterministic path:
quintic joint waypoints, duration growth for 1.2 rad/s and 3 rad/s² reference
bounds, soft-limit factor 0.9 with 0.03 rad margin, per-step speed limiting,
independent left/right ownership and a 1-second release blend towards live PPO
arm targets. The same node combines all motor commands; do not run the old
policy node concurrently.

Actor input order:

```
gyro[3], pelvis_gravity[3], twist[3], phase[2],
q_minus_default[27], dq[27], previous_executed_action[27],
left_mask, right_mask, executed_arm_target_minus_default[14]
```

The arm-reference order is left seven joints then right seven joints. Inactive
reference entries are zero; masks include the release transition. The previous
executed action is `(sent_position_target - action_offset) / action_scale`,
including overridden arms. It is initialized to zero after startup. The new
observation therefore contains neither discarded arm outputs nor Cartesian hand
errors. Measured joints come from one LowState snapshot. Known commands/references
are not privileged simulator measurements.

Deployment YAML defaults match the new recipe. If you change training parameters,
update and validate the deployment configuration too. Joint limits do not certify
collision-free motion; simulation contacts/model and the policy must handle the
selected arm poses. No claim of maximum reach or learned robustness follows from
an offline check.

## Checks

```bash
python -m unittest discover -s tests -v
```

Tests cover observation layout, discarded-action masking, executed-target history,
manual/release transitions, simulator-only routing, metadata rejection, invalid-
action stop behavior, existing evaluator compatibility, and NumPy-versus-training
controller parity when the sibling training dependencies are available. An actual
Linux DDS/validator rollout still needs to be checked on the Linux machine.
