# Physical reach stress test

`evaluate_sim.py` now defaults to `--reach-profile physical`. Run the existing
Unitree MuJoCo H1_2 validator first, then from the SDK directory:

```sh
python evaluate_sim.py deploy_hand_tracking.yaml /path/to/policy.onnx
```

This retains the original standing/walking/reaching protocol and adds a stationary
physical-reach block: stop, then both hands progress through 25%, 50%, 75%, and
100% of the displacement from the training-box center to the estimated forward,
outward, and upward workspace limits. Each move takes 5 seconds, followed by a
3-second hold. Hands recenter between directions. One further segment asks for
the estimated full forward reach while walking at 0.2 m/s. Existing stop/return/
release segments finish the run. `--time-scale 1.5` slows all moves and holds.

The estimator samples 8,192 configurations per arm within the model's arm joint
limits, retaining 0.02 rad at the limits. It uses forward kinematics and screens
modeled self-contacts. Several directional candidates are checked together to
find simultaneous two-arm poses without modeled penetration. Torso and legs stay
at their default pose during this geometric calculation. Only the resulting
Cartesian palm targets are passed to PPO. No IK, imitation target, or direct joint
position override is used. The random seed is fixed (7) for reproducibility.

These are **sampled directional estimates**, not proven physical maxima. The model
may omit collisions, loads, and compliance. A reference pose being geometrically
reachable does not establish dynamic feasibility or guarantee that this policy
can reach it. The reference joint poses are included in JSON only for auditing
FK, and are never used to command the robot. The new Cartesian commands deliberately
go beyond the trained command boxes. Poor tracking is a valid outcome of this test;
reliable full-extension control may require expanding the training workspace and
retraining. Smooth command interpolation does not guarantee smooth actual motion.

`summary.json` includes `physical_reach_estimates`: seed, sample count, accepted
candidates, target coordinates, reference joint configurations, and estimated
shoulder-to-palm distances. Per-stage summaries and CSV now also include measured
shoulder-to-palm extension for enabled hands. Read those together with settled
tracking errors and pelvis tilt to assess how much reach the policy achieved.
Completed means the sequence finished, not that it reached every target.

To retain the original interior-workspace protocol:

```sh
python evaluate_sim.py deploy_hand_tracking.yaml /path/to/policy.onnx --reach-profile standard
```

Offline trajectory/ONNX check (no DDS connection):

```sh
python evaluate_sim.py deploy_hand_tracking.yaml /path/to/policy.onnx --validate-only
```

`--reach-samples 32768` explores more candidate poses; `--reach-seed` changes the
seed. The same simulator-only DDS domain 1 / lo connection, 112-value actor layout,
startup ramp, tilt abort, and LowState timeout remain in effect. The Linux validator
must run in real time and unpaused because the sequence uses controller time.
