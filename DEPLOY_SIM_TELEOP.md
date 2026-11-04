# H1_2 hand-tracking validator playback

This adapter uses the current **Flat-HandTracking** actor: 112 observations and
27 PPO joint-position actions. The old `deploy_tray.yaml` (92 observations) and
older hand actor layouts are incompatible. `deploy_policy.py` is unchanged.

On Linux, keep the SDK and training repo as sibling directories, or update
`robot_xml` in `deploy_hand_tracking.yaml` to point to the training H1_2 MJCF.
Keep its referenced mesh assets with it. The adapter needs `numpy`, `pyyaml`,
`onnxruntime`, `mujoco`, and the installed `unitree_sdk2py`/CycloneDDS environment.
It does not import Torch or mjlab.

First validate the actual exported model without DDS or simulator commands:

```sh
python deploy_sim_teleop.py deploy_hand_tracking.yaml /path/to/policy.onnx simulation --validate-only
```

Then start your H1_2 validator with the HG `LowCmd_`/`LowState_` message types,
DDS domain **1**, loopback interface **lo**, and run:

```sh
python deploy_sim_teleop.py deploy_hand_tracking.yaml /path/to/policy.onnx simulation
```

The adapter waits for LowState, ramps to the default pose over four seconds, then
runs the actor at 50 Hz and publishes PD targets at 500 Hz. Focus this terminal
(not the validator viewer) for input; SSH needs a TTY (`ssh -t`).

| Keys | Control |
|---|---|
| W/S | Increase forward/backward velocity by 0.1 m/s |
| Q/E | Increase left/right velocity by 0.1 m/s |
| Z/X | Increase left/right yaw rate by 0.2 rad/s |
| B or Space | Stop walking |
| 1/2/3 | Select left/right/both hands |
| I/K | Increase/decrease target X |
| J/L | Increase/decrease target Y |
| U/O | Increase/decrease target Z |
| 4/5 | Toggle left/right tracking mask |
| 0 | Disable both hands |
| C | Center selected targets |
| Ctrl+C | Exit and restore the terminal |

Walking commands are **latched**, not hold-to-move; B/Space stops them. Hands
start disabled; moving a selected target enables it. Targets move in 2 cm steps,
clamped to the YAML workspace boxes, and remain fixed in the full moving pelvis
frame. Inactive target/error observation entries are zero, while measured palm
feedback remains present. No IK or manual arm action override is used.

The exact input order is:
`gyro(3), pelvis_gravity(3), twist(3), phase(2), q_rel(27), palm_pos(6),
masked_palm_error(6), dq(27), previous_action(27), masked_targets(6), masks(2)`.
Palm FK uses the same measured q snapshot as q_rel. Exported ONNX normalization
is retained. Input/output dimensions, joint/order metadata, and exported PD /
pose / scale metadata are checked before connecting DDS.

`imu_quaternion_frame` must match your validator's `imu_quat` sensor definition:
`imu` means the quaternion describes the torso-mounted IMU site; the adapter
uses encoder FK to transform projected gravity into the pelvis frame. Set
`pelvis` if the quaternion already describes the pelvis. Gyroscope values must
be in the training IMU site's axes. Stock/custom validator sensor layouts may
differ, so use the setting matching the actual XML and bridge. The FK MJCF uses
the training palm offsets; the validator's robot geometry must agree with them.

No reaching curriculum is active during evaluation: the full YAML boxes apply.
The current adapter supports the flat actor without terrain scans or observation
history. Do not use a rough-terrain actor whose input includes a height scan.

Offline tests:

```sh
python -m unittest discover -s tests -p test_deploy_sim_teleop.py -v
```

A Linux DDS/validator rollout must be checked on your Linux machine. Offline
validation does not establish learned balance or reaching quality.
