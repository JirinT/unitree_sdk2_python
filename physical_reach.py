"""Estimate directional arm workspace limits by joint-space sampling and FK.

This is not an IK controller or a proof of a global maximum. Reference joint
poses are used only to derive Cartesian test goals, never as policy actions.
"""

import mujoco
import numpy as np


def estimate_reach(fk, default_q, joint_names, samples=8192, seed=7, margin=.02):
    if samples < 64 or not np.isfinite(margin) or margin < 0:
        raise ValueError("Use at least 64 reach samples and a finite nonnegative joint margin")
    model = fk.model
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    names = list(joint_names)
    arm_ids = [[i for i, name in enumerate(names) if name.startswith(side+"_")
                and any(part in name for part in ("shoulder", "elbow", "wrist"))]
               for side in ("left", "right")]
    if any(len(ids) != 7 for ids in arm_ids):
        raise ValueError("Physical reach estimation expects seven joints in each H1_2 arm")
    ranges = np.array([model.jnt_range[model.joint(name).id] for name in names])
    limits = ranges.copy()
    for ids in arm_ids:
        limits[ids, 0] += margin
        limits[ids, 1] -= margin
    if (limits[:, 0] >= limits[:, 1]).any():
        raise ValueError("Joint margin leaves an empty joint range")

    def evaluate(q):
        data.qpos[:] = model.qpos0
        data.qpos[fk.q_addresses] = q
        mujoco.mj_fwdPosition(model, data)  # FK/contact geometry only; no integration
        for contact in data.contact[:data.ncon]:
            if (contact.dist < -.002 and model.geom_bodyid[contact.geom1] != 0
                    and model.geom_bodyid[contact.geom2] != 0):
                return None
        rotation = data.xmat[fk.root].reshape(3, 3)
        return ((data.site_xpos[fk.sites]-data.xpos[fk.root]) @ rotation).copy()

    # Keep several candidates per direction so simultaneous arm self-collisions
    # can be checked rather than combining incompatible independent extrema.
    candidates = [{direction: [] for direction in ("forward", "outward", "up")} for _ in range(2)]
    valid = [0, 0]
    for side, ids in enumerate(arm_ids):
        points = rng.uniform(limits[ids, 0], limits[ids, 1], size=(samples, len(ids)))
        sign = 1 if side == 0 else -1
        for point in points:
            q = np.array(default_q, dtype=float).copy()
            q[ids] = point
            palms = evaluate(q)
            if palms is None or palms[side, 0] < 0 or sign*palms[side, 1] < .12:
                continue
            valid[side] += 1
            p = palms[side]
            for direction, score in (("forward", p[0]), ("outward", sign*p[1]), ("up", p[2])):
                bucket = candidates[side][direction]
                if len(bucket) < 16 or score > bucket[-1][0]:
                    bucket.append((float(score), q.copy()))
                    bucket.sort(key=lambda x: x[0], reverse=True)
                    del bucket[16:]
    goals = {}
    references = {}
    for direction in ("forward", "outward", "up"):
        pairs = [(left[0]+right[0], left[1], right[1])
                 for left in candidates[0][direction] for right in candidates[1][direction]]
        pairs.sort(key=lambda x: x[0], reverse=True)
        for _, left, right in pairs:
            q = np.array(default_q, dtype=float).copy()
            q[arm_ids[0]] = left[arm_ids[0]]
            q[arm_ids[1]] = right[arm_ids[1]]
            palms = evaluate(q)
            if palms is not None:
                goals[direction] = palms
                references[direction] = q.tolist()
                break
        if direction not in goals:
            raise ValueError(f"No collision-screened simultaneous {direction} reach candidates; check FK model or increase --reach-samples")
    # The model's shoulder origins move with torso yaw; this estimate fixes the
    # torso/legs at their default values and changes only arm joints.
    data.qpos[:] = model.qpos0
    data.qpos[fk.q_addresses] = default_q
    mujoco.mj_kinematics(model, data)
    root_rotation = data.xmat[fk.root].reshape(3, 3)
    shoulders = np.stack([(data.xpos[model.body(side+"_shoulder_pitch_link").id]-data.xpos[fk.root]) @ root_rotation
                          for side in ("left", "right")])
    info = dict(method="seeded arm joint-space sampling, FK and modeled self-contact screening",
                samples_per_arm=samples, seed=seed, joint_limit_margin_rad=margin,
                accepted_samples_per_arm=valid,
                note="Directional sampled estimates; not proven global maxima, dynamic feasibility, or PPO tracking guarantees. No IK or direct joint override.",
                goals={name: dict(palms_pelvis_m=value.tolist(),
                    shoulder_to_palm_m=np.linalg.norm(value-shoulders, axis=1).tolist(),
                    reference_joint_positions_rad=references[name]) for name, value in goals.items()})
    return goals, info
