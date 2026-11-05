"""Repeatable H1_2 policy evaluation through the separate Unitree MuJoCo validator.

Only simulator DDS domain 1 on loopback is used. Start the validator separately.
Commands use quintic interpolation; CSV/JSON report tracking and tilt, not actual
walking speed (LowState supplies no world position or base linear velocity).
"""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import time

import numpy as np

from deploy_sim_teleop import CONTROL_DT, PolicyRunner


def smoothstep(u):
    u = np.clip(u, 0., 1.)
    return u*u*u*(10. + u*(-15. + 6.*u))


@dataclass
class Stage:
    name: str
    move_s: float
    hold_s: float
    velocity: np.ndarray
    palms: np.ndarray
    masks: np.ndarray
    disable_after: tuple = ()


def make_stages(boxes, velocity_ranges, rest, time_scale=1.):
    if not math.isfinite(time_scale) or time_scale <= 0:
        raise ValueError("time_scale must be finite and positive; >1 makes moves slower")
    boxes = np.asarray(boxes)
    center = boxes.mean(axis=2)
    half = (boxes[:, :, 1]-boxes[:, :, 0])/2
    # Normalized offsets keep all test goals comfortably inside configured boxes.
    raised = center + half*np.array([[.55, .35, .55], [.55, -.35, .55]])
    asymmetric = center + half*np.array([[.55, -.25, .4], [-.35, .25, -.35]])
    left = center.copy()
    left[0] = raised[0]
    right = center.copy()
    right[1] = raised[1]
    limits = np.asarray(velocity_ranges)
    def stage(name, move, hold, velocity, palms, masks, disable=()):
        velocity = np.asarray(velocity, dtype=float)
        if (velocity < limits[:, 0]).any() or (velocity > limits[:, 1]).any():
            raise ValueError(f"Scenario velocity outside YAML limits: {name}: {velocity}")
        return Stage(name, move*time_scale, hold*time_scale, velocity,
                     np.array(palms, dtype=float), np.array(masks, dtype=float), disable)
    return [
        stage("stand_baseline", 1, 3, [0, 0, 0], rest, [0, 0]),
        stage("left_enable", 3, 2, [0, 0, 0], center, [1, 0]),
        stage("left_reach", 4, 2, [0, 0, 0], left, [1, 0]),
        stage("left_return_release", 4, 2, [0, 0, 0], rest, [1, 0], (0,)),
        stage("right_enable", 3, 2, [0, 0, 0], center, [0, 1]),
        stage("right_reach", 4, 2, [0, 0, 0], right, [0, 1]),
        stage("right_return_release", 4, 2, [0, 0, 0], rest, [0, 1], (1,)),
        stage("both_enable", 4, 2, [0, 0, 0], center, [1, 1]),
        stage("both_reach", 4, 2, [0, 0, 0], raised, [1, 1]),
        stage("walk_forward", 3, 5, [.3, 0, 0], raised, [1, 1]),
        stage("walk_asymmetric_hands", 4, 3, [.3, 0, 0], asymmetric, [1, 1]),
        stage("strafe_left", 3, 4, [0, .15, 0], center, [1, 1]),
        stage("strafe_right", 3, 4, [0, -.15, 0], raised, [1, 1]),
        stage("turn_left", 3, 4, [.15, 0, .35], center, [1, 1]),
        stage("turn_right", 3, 4, [.15, 0, -.35], asymmetric, [1, 1]),
        stage("walk_backward", 3, 4, [-.15, 0, 0], center, [1, 1]),
        stage("stop_and_center", 3, 3, [0, 0, 0], center, [1, 1]),
        stage("both_return_release", 4, 2, [0, 0, 0], rest, [1, 1], (0, 1)),
        stage("stand_finish", 1, 3, [0, 0, 0], rest, [0, 0]),
    ]


class Sequence:
    def __init__(self, stages, rest):
        self.stages = stages
        self.ends = np.cumsum([s.move_s+s.hold_s for s in stages])
        self.duration = float(self.ends[-1])
        self.index = -1
        self.velocity = np.zeros(3)
        self.palms = np.array(rest, dtype=float).copy()
        self.masks = np.zeros(2)
        self.last_t = -1.

    def sample(self, t, measured_palms):
        if not math.isfinite(t) or t < 0 or t < self.last_t:
            raise ValueError("Sequence time must be finite, nonnegative and monotonic")
        self.last_t = t
        index = min(int(np.searchsorted(self.ends, t, side="right")), len(self.stages)-1)
        # Usually advances one stage; skipped stages still propagate their endpoints.
        while self.index < index:
            if self.index >= 0:
                old = self.stages[self.index]
                self.velocity = old.velocity.copy()
                self.palms = old.palms.copy()
                self.masks = old.masks.copy()
                self.masks[list(old.disable_after)] = 0
            self.index += 1
            current = self.stages[self.index]
            self.start_velocity = self.velocity.copy()
            self.start_palms = self.palms.copy()
            enabled = (current.masks == 1) & (self.masks == 0)
            # Binary masks match training. Starting at the measured palm gives
            # zero initial error, even if the rest pose lies outside the task box.
            self.start_palms[enabled] = np.asarray(measured_palms)[enabled]
        stage = self.stages[self.index]
        start_t = 0. if self.index == 0 else self.ends[self.index-1]
        elapsed = t-start_t
        blend = smoothstep(elapsed/stage.move_s)
        self.velocity = self.start_velocity + blend*(stage.velocity-self.start_velocity)
        self.palms = self.start_palms + blend*(stage.palms-self.start_palms)
        self.masks = stage.masks.copy()
        if elapsed >= stage.move_s:
            self.masks[list(stage.disable_after)] = 0
        return dict(stage=stage.name, velocity=self.velocity.copy(), palms=self.palms.copy(),
                    masks=self.masks.copy(), settled=elapsed >= stage.move_s+.5,
                    done=t >= self.duration)


def stats(values):
    if not values:
        return None
    a = np.asarray(values)
    return {"mean": float(a.mean()), "p95": float(np.percentile(a, 95)), "max": float(a.max())}


class Report:
    def __init__(self, output, stages, metadata):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.stream = (self.output/"samples.csv").open("x", newline="")
        self.writer = csv.writer(self.stream)
        self.writer.writerow(["time_s", "stage", "vx_cmd", "vy_cmd", "yaw_rate_cmd", "left_enabled", "right_enabled",
                              *[f"{side}_{kind}_{axis}" for kind in ("target", "palm") for side in ("left", "right") for axis in "xyz"],
                              "left_error_m", "right_error_m", "pelvis_tilt_deg", "palm_speed_max_m_s",
                              "action_rate_rms_s", "joint_target_rate_rms_rad_s", "settled"])
        self.by_stage = {s.name: dict(samples=0, left=[], right=[], settled_left=[], settled_right=[],
                                     tilt=[], palm_speed=[], action_rate=[], joint_target_rate=[]) for s in stages}
        self.metadata = metadata
        self.previous = None
        self.sample_count = 0

    def add(self, t, sample, obs, action, target_q, dt):
        palms = obs[38:44].reshape(2, 3).astype(float)
        errors = np.linalg.norm(sample["palms"]-palms, axis=1)
        g = obs[3:6].astype(float)
        tilt = math.degrees(math.acos(float(np.clip(-g[2]/np.linalg.norm(g), -1, 1))))
        palm_speed = action_rate = joint_rate = 0.
        if self.previous is not None:
            old_palms, old_action, old_q = self.previous
            palm_speed = float(np.linalg.norm((palms-old_palms)/dt, axis=1).max())
            action_rate = float(np.sqrt(np.mean(((action-old_action)/dt)**2)))
            joint_rate = float(np.sqrt(np.mean(((target_q-old_q)/dt)**2)))
        self.previous = (palms.copy(), action.copy(), target_q.copy())
        bucket = self.by_stage[sample["stage"]]
        bucket["samples"] += 1
        for side, name in enumerate(("left", "right")):
            if sample["masks"][side]:
                bucket[name].append(float(errors[side]))
                if sample["settled"]:
                    bucket["settled_"+name].append(float(errors[side]))
        for name, value in (("tilt", tilt), ("palm_speed", palm_speed), ("action_rate", action_rate), ("joint_target_rate", joint_rate)):
            bucket[name].append(value)
        error_fields = [float(errors[s]) if sample["masks"][s] else "" for s in range(2)]
        self.writer.writerow([t, sample["stage"], *sample["velocity"], *sample["masks"],
                              *sample["palms"].ravel(), *palms.ravel(), *error_fields, tilt,
                              palm_speed, action_rate, joint_rate, int(sample["settled"])])
        self.sample_count += 1
        if self.sample_count % 50 == 0:
            self.stream.flush()
        return tilt

    def finish(self, status, reason=None):
        self.stream.close()
        report = dict(self.metadata, status=status, reason=reason, samples=self.sample_count,
                      note="Controller-time measurements; no ground-truth base velocity, distance, or height is available from LowState.",
                      stages={})
        for name, values in self.by_stage.items():
            report["stages"][name] = {"samples": values["samples"],
                "left_error_m": stats(values["left"]), "right_error_m": stats(values["right"]),
                "settled_left_error_m": stats(values["settled_left"]), "settled_right_error_m": stats(values["settled_right"]),
                "pelvis_tilt_deg": stats(values["tilt"]), "palm_speed_max_m_s": stats(values["palm_speed"]),
                "action_rate_rms_s": stats(values["action_rate"]), "joint_target_rate_rms_rad_s": stats(values["joint_target_rate"])}
        (self.output/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(f"Evaluation {status}; results: {self.output.resolve()}")
        return report


class EvaluationRunner(PolicyRunner):
    def __init__(self, deploy_yaml, onnx_path, output, time_scale=1., robot_xml=None,
                 abort_tilt_deg=60., tilt_grace_s=.25, state_timeout=1.):
        super().__init__(deploy_yaml, onnx_path, simulation=True, robot_xml=robot_xml)
        if (not 0 < abort_tilt_deg < 180 or not math.isfinite(tilt_grace_s) or tilt_grace_s < 0
                or not math.isfinite(state_timeout) or state_timeout <= 0):
            raise ValueError("Invalid tilt or LowState timeout settings")
        rest, _ = self.fk.compute(self.default_q)
        self.stages = make_stages(self.controls.boxes, self.controls.velocity_ranges, rest, time_scale)
        self.sequence = Sequence(self.stages, rest)
        self.report = Report(output, self.stages, dict(policy=str(Path(onnx_path).resolve()),
            deploy_yaml=str(Path(deploy_yaml).resolve()), controller_dt=self.step_dt,
            scenario_duration_s=self.sequence.duration, time_scale=time_scale,
            abort_tilt_deg=abort_tilt_deg, tilt_grace_s=tilt_grace_s, state_timeout_s=state_timeout))
        self.abort_tilt_deg, self.tilt_grace_s = abort_tilt_deg, tilt_grace_s
        self.state_timeout = state_timeout
        self.tilt_time = 0.
        self.received_at = None
        self.completed = False
        self.current_stage = None
        self.last_obs = None

    def LowStateHandler(self, msg):
        super().LowStateHandler(msg)
        self.received_at = time.monotonic()

    def _control_step(self):
        if self.received_at is not None and time.monotonic()-self.received_at > self.state_timeout:
            raise TimeoutError("LowState stopped updating (validator paused, disconnected, or stalled)")
        super()._control_step()

    def build_obs(self):
        obs = super().build_obs()
        self.last_obs = obs
        return obs

    def infer(self):
        t = self.policy_time
        # Read-only kinematics to seed a newly enabled hand at its current position.
        ls = self.low_state
        q = np.array([ls.motor_state[i].q for i in self.joint_map])
        palms, _ = self.fk.compute(q)
        sample = self.sequence.sample(t, palms)
        if sample["stage"] != self.current_stage:
            self.current_stage = sample["stage"]
            print(f"[{t:6.2f}s / {self.sequence.duration:.1f}s] {self.current_stage}")
        with self.controls.lock:
            self.controls.velocity[:] = sample["velocity"]
            self.controls.targets[:] = sample["palms"]
            self.controls.masks[:] = sample["masks"]
        super().infer()
        tilt = self.report.add(t, sample, self.last_obs, self.last_action, self.target_q, self.step_dt)
        self.tilt_time = self.tilt_time+self.step_dt if tilt >= self.abort_tilt_deg else 0.
        if tilt >= self.abort_tilt_deg and self.tilt_time >= self.tilt_grace_s:
            raise RuntimeError(f"Sustained pelvis tilt {tilt:.1f} degrees exceeded evaluation threshold")
        if sample["done"]:
            self.completed = True
            self.stop_event.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deploy_yaml")
    parser.add_argument("onnx_path")
    parser.add_argument("--robot-xml")
    parser.add_argument("--time-scale", type=float, default=1., help="Multiply all move/hold times; 1.5 is slower")
    parser.add_argument("--output", default=None, help="Results directory (default: evaluation_results/timestamp)")
    parser.add_argument("--abort-tilt-deg", type=float, default=60.)
    parser.add_argument("--tilt-grace-s", type=float, default=.25)
    parser.add_argument("--state-timeout", type=float, default=1.)
    parser.add_argument("--validate-only", action="store_true", help="Offline trajectory and neutral-state ONNX check, without DDS")
    args = parser.parse_args()
    if args.validate_only:
        base = PolicyRunner(args.deploy_yaml, args.onnx_path, simulation=True, robot_xml=args.robot_xml)
        rest, _ = base.fk.compute(base.default_q)
        stages = make_stages(base.controls.boxes, base.controls.velocity_ranges, rest, args.time_scale)
        sequence = Sequence(stages, rest)
        for t in np.arange(0., sequence.duration+base.step_dt, base.step_dt):
            sample = sequence.sample(t, rest)
            assert np.isfinite(sample["palms"]).all() and np.isfinite(sample["velocity"]).all()
        base.validate()
        print(f"Offline trajectory OK: {len(stages)} stages, {sequence.duration:.1f}s plus startup ramp; no physical rollout")
        return
    output = args.output or str(Path("evaluation_results")/datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    runner = EvaluationRunner(args.deploy_yaml, args.onnx_path, output, args.time_scale, args.robot_xml,
                              args.abort_tilt_deg, args.tilt_grace_s, args.state_timeout)
    status, reason = "interrupted", None
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        ChannelFactoryInitialize(1, "lo")
        runner.Init()
        print(f"Evaluation: {runner.sequence.duration:.1f}s after the four-second startup ramp.")
        runner.Start()
        while not runner.stop_event.wait(.1):
            pass
        status = "complete" if runner.completed else "failed"
        reason = None if runner.error is None else str(runner.error)
    except KeyboardInterrupt:
        reason = "Ctrl+C"
    except Exception as error:
        status, reason = "failed", str(error)
        raise
    finally:
        try:
            runner.Stop()
        finally:
            runner.report.finish(status, reason)
    if runner.error is not None:
        raise RuntimeError("Evaluation failed; see summary.json") from runner.error


if __name__ == "__main__":
    main()
