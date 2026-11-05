"""Evaluation protocol tests, without DDS or physics integration."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_sim as evaluate
import test_deploy_sim_teleop as adapter_test


class Tests(unittest.TestCase):
    setUp = adapter_test.Tests.setUp
    state = adapter_test.Tests.state

    def base(self):
        return adapter_test.Tests.runner(self)

    def runner(self, **kwargs):
        kwargs.setdefault("reach_profile", "standard")
        self.path.write_text(yaml.safe_dump(self.cfg))
        return evaluate.EvaluationRunner(self.path, "unused.onnx", Path(self.temp.name)/"results", **kwargs)

    def test_expressive_demo_duration_actions_and_targets(self):
        base = self.base()
        rest, _ = base.fk.compute(base.default_q)
        stages = evaluate.make_demo_stages(base.controls.boxes, base.controls.velocity_ranges, rest)
        self.assertAlmostEqual(sum(s.move_s+s.hold_s for s in stages)+evaluate.RAMP_DURATION, 27.)
        self.assertEqual([s.name for s in stages], ["demo_stand", "demo_open_approach", "demo_reach_to_grasp", "demo_lift_to_carry", "demo_walk_carrying", "demo_stop", "demo_present_forward", "demo_place", "demo_open_release", "demo_withdraw_release", "demo_stand_finish"])
        for stage in stages[1:9]:
            self.assertTrue((stage.masks == 1).all())
            self.assertTrue((stage.palms >= base.controls.boxes[:, :, 0]).all())
            self.assertTrue((stage.palms <= base.controls.boxes[:, :, 1]).all())
        by_name = {stage.name: stage for stage in stages}
        walk = by_name["demo_walk_carrying"]
        self.assertGreater(walk.velocity[0], 0)
        self.assertGreaterEqual(walk.move_s+walk.hold_s, 4.5)
        for name in ("demo_stop", "demo_present_forward", "demo_place", "demo_open_release"):
            self.assertFalse(by_name[name].velocity.any())
        grasp = by_name["demo_reach_to_grasp"].palms
        carry = by_name["demo_lift_to_carry"].palms
        place = by_name["demo_place"].palms
        self.assertTrue((carry[:, 2]-grasp[:, 2] >= .30).all())
        self.assertTrue((carry[:, 2]-place[:, 2] >= .30).all())
        approach = by_name["demo_open_approach"].palms
        gap = lambda palms: palms[0, 1]-palms[1, 1]
        self.assertGreater(gap(approach)-gap(grasp), .30)
        self.assertGreater(gap(by_name["demo_open_release"].palms)-gap(place), .30)
        sequence = evaluate.Sequence(stages, rest)
        for t in np.arange(0., sequence.duration+.021, .02):
            sample = sequence.sample(t, rest)
            self.assertTrue(np.isfinite(sample["palms"]).all())
        self.assertTrue(sample["done"])
        self.assertFalse(sample["masks"].any())
        self.assertFalse(sample["velocity"].any())
        slow = evaluate.make_demo_stages(base.controls.boxes, base.controls.velocity_ranges, rest, 1.5)
        self.assertAlmostEqual(sum(s.move_s+s.hold_s for s in slow), 1.5*sequence.duration)

    def test_demo_runner_does_not_sample_extreme_reach(self):
        with patch.object(evaluate, "estimate_reach", side_effect=AssertionError("physical reach used for expressive demo")):
            runner = self.runner(reach_profile="demo")
        try:
            self.state(runner)
            runner.infer()
            self.assertAlmostEqual(runner.sequence.duration+evaluate.RAMP_DURATION, 27.)
            self.assertEqual(runner.current_stage, "demo_stand")
            self.assertIsNone(runner.report.metadata["physical_reach_estimates"])
        finally:
            runner.report.finish("test")

    def test_physical_reach_uses_fk_limits_and_progressive_goals(self):
        base = self.base()
        goals, info = evaluate.estimate_reach(base.fk, base.default_q, base.joint_names, samples=256, seed=7)
        self.assertEqual(info["samples_per_arm"], 256)
        outside = False
        for direction, palms in goals.items():
            q = np.asarray(info["goals"][direction]["reference_joint_positions_rad"])
            reconstructed, _ = base.fk.compute(q)
            np.testing.assert_allclose(palms, reconstructed, atol=1e-8)
            for i, name in enumerate(base.joint_names):
                limits = base.fk.model.jnt_range[base.fk.model.joint(name).id]
                self.assertTrue(limits[0] <= q[i] <= limits[1])
            outside |= bool(((palms < base.controls.boxes[:, :, 0]) | (palms > base.controls.boxes[:, :, 1])).any())
        self.assertTrue(outside)
        rest, _ = base.fk.compute(base.default_q)
        standard = evaluate.make_stages(base.controls.boxes, base.controls.velocity_ranges, rest)
        extended = evaluate.add_physical_reach(standard, base.controls.boxes, goals)
        for direction in goals:
            endpoint = next(s for s in extended if s.name == f"maxreach_{direction}_100")
            np.testing.assert_allclose(endpoint.palms, goals[direction])
            previous = base.controls.boxes.mean(axis=2)
            for fraction in (25, 50, 75, 100):
                stage = next(s for s in extended if s.name == f"maxreach_{direction}_{fraction}")
                self.assertTrue((stage.masks == 1).all())
                self.assertTrue(np.isfinite(stage.palms).all())
                previous = stage.palms
        self.assertEqual(extended[-1].name, "stand_finish")

    def test_smoothstep_zero_endpoint_velocity_acceleration(self):
        self.assertEqual(evaluate.smoothstep(0), 0)
        self.assertEqual(evaluate.smoothstep(1), 1)
        eps = 1e-4
        for endpoint in (0., 1.):
            self.assertLess(abs((evaluate.smoothstep(endpoint+eps)-evaluate.smoothstep(endpoint-eps))/(2*eps)), 1e-6)
            self.assertLess(abs((evaluate.smoothstep(endpoint+eps)-2*evaluate.smoothstep(endpoint)+evaluate.smoothstep(endpoint-eps))/eps**2), .002)

    def test_sequence_activation_release_continuity_and_limits(self):
        base = self.base()
        rest, _ = base.fk.compute(base.default_q)
        stages = evaluate.make_stages(base.controls.boxes, base.controls.velocity_ranges, rest)
        sequence = evaluate.Sequence(stages, rest)
        old = None
        max_target_step = 0.
        for t in np.arange(0., sequence.duration+.021, .02):
            sample = sequence.sample(t, rest)
            self.assertTrue(np.isfinite(sample["palms"]).all())
            self.assertTrue(np.isin(sample["masks"], [0, 1]).all())
            self.assertTrue((sample["velocity"] >= base.controls.velocity_ranges[:, 0]).all())
            self.assertTrue((sample["velocity"] <= base.controls.velocity_ranges[:, 1]).all())
            if old is not None:
                shared = (old["masks"] == 1) & (sample["masks"] == 1)
                if shared.any():
                    max_target_step = max(max_target_step, float(np.linalg.norm(sample["palms"][shared]-old["palms"][shared], axis=1).max()))
            old = sample
        self.assertTrue(old["done"])
        self.assertFalse(old["velocity"].any())
        self.assertFalse(old["masks"].any())
        self.assertLess(max_target_step, .01)
        with self.assertRaises(ValueError): sequence.sample(0, rest)
        # Activation at its exact boundary starts at the measured hand, not a box corner.
        sequence = evaluate.Sequence(stages, rest)
        measured = rest+.017
        sequence.sample(0, rest)
        at_enable = sequence.sample(stages[0].move_s+stages[0].hold_s, measured)
        np.testing.assert_allclose(at_enable["palms"][0], measured[0])
        self.assertEqual(at_enable["masks"][0], 1)

    def test_time_scaling_and_invalid_velocity_limits(self):
        base = self.base()
        rest, _ = base.fk.compute(base.default_q)
        stages = evaluate.make_stages(base.controls.boxes, base.controls.velocity_ranges, rest)
        slow = evaluate.make_stages(base.controls.boxes, base.controls.velocity_ranges, rest, 2.)
        self.assertEqual(sum(s.move_s+s.hold_s for s in slow), 2*sum(s.move_s+s.hold_s for s in stages))
        for value in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                evaluate.make_stages(base.controls.boxes, base.controls.velocity_ranges, rest, value)
        with self.assertRaisesRegex(ValueError, "velocity"):
            evaluate.make_stages(base.controls.boxes, np.array([[-.1, .1]]*3), rest)

    def test_report_ignores_disabled_hands_and_records_settled_errors(self):
        runner = self.runner()
        try:
            self.state(runner)
            for _ in range(2): runner.infer()
            sample = dict(stage="both_reach", velocity=np.zeros(3), palms=np.zeros((2, 3)), masks=np.array([1., 0.]), settled=True)
            obs = runner.last_obs.copy()
            obs[38:44] = [0.1, 0., 0., 999., 999., 999.]
            runner.report.add(.1, sample, obs, runner.last_action, runner.target_q, runner.step_dt)
        finally:
            summary = runner.report.finish("complete")
        both = summary["stages"]["both_reach"]
        self.assertIsNone(both["right_error_m"])
        self.assertIsNone(both["settled_right_error_m"])
        self.assertAlmostEqual(both["settled_left_error_m"]["mean"], .1, places=6)
        parsed = json.loads((runner.report.output/"summary.json").read_text())
        self.assertEqual(parsed["status"], "complete")
        self.assertEqual(parsed["samples"], 3)
        self.assertTrue((runner.report.output/"samples.csv").is_file())

    def test_runner_commands_feed_actor_and_completion(self):
        runner = self.runner()
        try:
            self.state(runner)
            runner.policy_time = 65.
            runner.infer()
            self.assertEqual(runner.session.feed["obs"].shape, (1, 112))
            stage_stats = runner.report.by_stage[runner.current_stage]
            self.assertTrue(stage_stats["left_extension"])
            self.assertTrue(stage_stats["right_extension"])
            velocity, targets, masks = runner.controls.snapshot()
            np.testing.assert_allclose(runner.session.feed["obs"][0, 6:9], velocity)
            np.testing.assert_allclose(runner.session.feed["obs"][0, 104:110], np.where(masks[:, None].astype(bool), targets, 0).ravel(), atol=1e-6)
            runner.policy_time = runner.sequence.duration
            runner.infer()
            self.assertTrue(runner.completed)
            self.assertTrue(runner.stop_event.is_set())
        finally:
            runner.report.finish("test")

    def test_stale_state_and_sustained_tilt_abort(self):
        runner = self.runner(abort_tilt_deg=45., tilt_grace_s=.04)
        try:
            self.state(runner)
            runner.received_at = 0.
            with self.assertRaisesRegex(TimeoutError, "LowState"):
                runner._control_step()
            runner.low_state.imu_state.quaternion = np.array([0., 1., 0., 0.])
            runner.infer()
            with self.assertRaisesRegex(RuntimeError, "tilt"):
                runner.infer()
            self.assertFalse(runner.completed)
        finally:
            runner.report.finish("test")


if __name__ == "__main__":
    unittest.main()
