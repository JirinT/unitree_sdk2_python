"""Offline tests: no Unitree SDK import, DDS channel, robot, or viewer is opened."""

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import deploy_sim_teleop as deploy


class Session:
    metadata = {}
    inputs = [SimpleNamespace(name="obs", shape=[1, 112], type="tensor(float)")]
    outputs = [SimpleNamespace(name="actions", shape=[1, 27])]

    def __init__(self, *args, **kwargs):
        self.action = np.linspace(-.3, .3, 27, dtype=np.float32)[None]

    def get_inputs(self): return self.inputs
    def get_outputs(self): return self.outputs
    def get_modelmeta(self): return SimpleNamespace(custom_metadata_map=self.metadata)
    def run(self, _, feed):
        self.feed = feed
        return [self.action.copy()]


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg = yaml.safe_load((ROOT / "deploy_hand_tracking.yaml").read_text())
        self.cfg["robot_xml"] = str((ROOT / self.cfg["robot_xml"]).resolve())
        self.path = Path(self.temp.name) / "deploy.yaml"
        self.patch = patch.object(deploy.onnxruntime, "InferenceSession", Session)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        Session.metadata = {}
        Session.inputs = [SimpleNamespace(name="obs", shape=[1, 112], type="tensor(float)")]

    def runner(self):
        self.path.write_text(yaml.safe_dump(self.cfg))
        return deploy.PolicyRunner(self.path, "unused.onnx", simulation=True)

    def state(self, runner, q=None, dq=None):
        q = runner.default_q if q is None else q
        dq = np.zeros(27) if dq is None else dq
        motors = [SimpleNamespace(q=0., dq=0.) for _ in range(27)]
        for i, motor in enumerate(runner.joint_map):
            motors[motor].q, motors[motor].dq = q[i], dq[i]
        _, rotation = runner.fk.compute(q)
        imu_local = np.zeros(4)
        mujoco.mju_mat2Quat(imu_local, rotation.ravel())
        root_quat = np.array([.7, .1, -.2, .3])
        root_quat /= np.linalg.norm(root_quat)
        imu_world = np.zeros(4)
        mujoco.mju_mulQuat(imu_world, root_quat, imu_local)
        runner.low_state = SimpleNamespace(
            motor_state=motors, mode_machine=7,
            imu_state=SimpleNamespace(gyroscope=[.1, -.2, .3], quaternion=imu_world),
        )
        matrix = np.zeros(9)
        mujoco.mju_quat2Mat(matrix, root_quat)
        return matrix.reshape(3, 3).T @ np.array([0., 0., -1.]), root_quat

    def control_ready(self, runner):
        self.state(runner)
        runner.LowStateHandler(runner.low_state)
        runner.low_cmd = SimpleNamespace(motor_cmd=[SimpleNamespace() for _ in range(27)])
        runner.crc = SimpleNamespace(Crc=lambda cmd: 123)
        runner.lowcmd_publisher_ = Mock()
        runner.time_ = deploy.RAMP_DURATION
        runner.time_since_infer = runner.step_dt - deploy.CONTROL_DT

    def test_exact_actor_layout_and_pelvis_imu_transform(self):
        runner = self.runner()
        q = runner.default_q + np.arange(27)*.01
        dq = np.arange(27)*.03
        expected_gravity, root_quat = self.state(runner, q, dq)
        for key in "4WQ": runner.controls.press(key)
        runner.policy_time = .1
        runner.last_action = np.linspace(0, 1, 27, dtype=np.float32)
        obs = runner.build_obs()
        palms, _ = runner.fk.compute(q)
        velocity, targets, masks = runner.controls.snapshot()
        np.testing.assert_allclose(obs[:3], [.1, -.2, .3], atol=1e-7)
        np.testing.assert_allclose(obs[3:6], expected_gravity, atol=1e-7)
        np.testing.assert_allclose(obs[6:9], velocity)
        angle = 2*np.pi*.1/.6
        np.testing.assert_allclose(obs[9:11], [np.sin(angle), np.cos(angle)], atol=1e-7)
        np.testing.assert_allclose(obs[11:38], q-runner.default_q, atol=1e-7)
        np.testing.assert_allclose(obs[38:44], palms.ravel(), atol=1e-7)
        np.testing.assert_allclose(obs[44:50], np.where(masks[:, None].astype(bool), targets-palms, 0).ravel(), atol=1e-7)
        np.testing.assert_allclose(obs[50:77], dq, atol=1e-7)
        np.testing.assert_allclose(obs[77:104], runner.last_action)
        np.testing.assert_allclose(obs[104:110], np.where(masks[:, None].astype(bool), targets, 0).ravel(), atol=1e-7)
        np.testing.assert_allclose(obs[110:], masks)
        runner.imu_quaternion_frame = "pelvis"
        runner.low_state.imu_state.quaternion = root_quat
        np.testing.assert_allclose(runner.build_obs()[3:6], expected_gravity, atol=1e-7)

    def test_masks_and_keyboard_clamps(self):
        runner = self.runner()
        self.state(runner)
        self.assertFalse(runner.build_obs()[44:50].any())
        self.assertFalse(runner.build_obs()[104:].any())
        for key in "3" + "I"*50 + "J"*50 + "U"*50 + "W"*50:
            runner.controls.press(key)
        velocity, targets, masks = runner.controls.snapshot()
        np.testing.assert_allclose(targets, runner.controls.boxes[:, :, 1])
        np.testing.assert_allclose(masks, [1, 1])
        self.assertEqual(velocity[0], 1)
        for key in "B0C": runner.controls.press(key)
        velocity, targets, masks = runner.controls.snapshot()
        self.assertFalse(velocity.any())
        self.assertFalse(masks.any())
        np.testing.assert_allclose(targets, runner.controls.boxes.mean(axis=2))
        self.assertFalse(runner.controls.press("?"))

    def test_encoder_fk_matches_training_torch_fk(self):
        training = ROOT.parent / "unitree_rl_mjlab/src/tasks/velocity/mdp/hand_kinematics.py"
        if not training.exists() or importlib.util.find_spec("torch") is None or importlib.util.find_spec("mjlab") is None:
            self.skipTest("Optional local training package is unavailable")
        import torch
        spec = importlib.util.spec_from_file_location("training_hand_fk", training)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        runner = self.runner()
        fk = module.PalmKinematics(runner.fk.model, runner.joint_names, self.cfg["hand_sites"], "cpu")
        q = np.random.default_rng(4).normal(size=(8, 27))*.3
        expected = fk(torch.tensor(q, dtype=torch.float32)).numpy()
        actual = np.stack([runner.fk.compute(row)[0] for row in q])
        np.testing.assert_allclose(actual, expected, atol=2e-6)

    def test_all_arm_and_leg_targets_are_policy_actions(self):
        runner = self.runner()
        self.state(runner)
        runner.infer()
        np.testing.assert_allclose(runner.target_q, runner.session.action[0]*runner.action_scale+runner.action_offset)
        np.testing.assert_array_equal(runner.last_action, runner.session.action[0])
        self.assertEqual(runner.session.feed["obs"].shape, (1, 112))
        self.assertEqual(runner.session.feed["obs"].dtype, np.float32)
        self.assertAlmostEqual(runner.policy_time, .02)

    def test_motor_mapping_and_policy_decimation(self):
        self.cfg["joint_ids_map"] = list(reversed(self.cfg["joint_ids_map"]))
        runner = self.runner()
        self.control_ready(runner)
        with patch.object(runner, "infer", wraps=runner.infer) as infer:
            for _ in range(21): runner.ControlStep()
            self.assertEqual(infer.call_count, 3)
        for index, motor in enumerate(runner.joint_map):
            cmd = runner.low_cmd.motor_cmd[motor]
            self.assertAlmostEqual(cmd.q, runner.target_q[index])
            self.assertEqual(cmd.kp, runner.stiffness[index])
        self.assertEqual(runner.low_cmd.mode_pr, 0)
        self.assertEqual(runner.low_cmd.mode_machine, 7)
        self.assertEqual(runner.low_cmd.crc, 123)

    def test_invalid_policy_config_and_metadata_fail_before_dds(self):
        self.cfg["obs_dim"] = 92
        with self.assertRaisesRegex(ValueError, "112"):
            self.runner()
        self.cfg["obs_dim"] = 112
        Session.inputs = [SimpleNamespace(name="obs", shape=[1, 92], type="tensor(float)")]
        with self.assertRaisesRegex(ValueError, "ONNX"):
            self.runner()
        Session.inputs[0].shape = [1, 112]
        Session.metadata = {"observation_names": "base_ang_vel,joint_pos"}
        with self.assertRaisesRegex(ValueError, "observation_names"):
            self.runner()
        Session.metadata = {"action_scale": ",".join(["0"]*27)}
        with self.assertRaisesRegex(ValueError, "action_scale"):
            self.runner()

    def test_offline_validation_never_imports_sdk_and_invalid_actions_stop(self):
        with patch.dict(sys.modules, {"unitree_sdk2py": None}):
            runner = self.runner()
            runner.validate()
        self.control_ready(runner)
        runner.session.action[:] = np.nan
        runner.ControlStep()
        self.assertTrue(runner.stop_event.is_set())
        self.assertIsInstance(runner.error, ValueError)
        runner.lowcmd_publisher_.Write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
