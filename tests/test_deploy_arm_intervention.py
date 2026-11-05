"""Offline SDK adapter tests; never initialize DDS or contact a robot."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import deploy_sim_teleop as base
import deploy_arm_intervention as arm
import test_deploy_sim_teleop as legacy


class Session(legacy.Session):
    inputs=[SimpleNamespace(name="obs",shape=[1,108],type="tensor(float)")]


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg=yaml.safe_load((ROOT/'deploy_arm_intervention.yaml').read_text())
        self.cfg['robot_xml']=str((ROOT/self.cfg['robot_xml']).resolve())
        self.path=Path(self.temp.name)/'deploy.yaml'
        Session.inputs=[SimpleNamespace(name="obs",shape=[1,108],type="tensor(float)")]
        Session.metadata={
            'arm_control_contract':'intervention_v1',
            'previous_action_semantics':'executed_target_minus_offset_div_scale',
            'arm_control_observation':'mask_left,mask_right,executed_arm_target_minus_default[14]',
            'arm_joint_names':','.join(arm.ARM_JOINT_NAMES),
            'arm_action_indices':','.join(f'{i:.3f}' for i in range(13,27)),
            'arm_max_joint_speed':'1.2','arm_max_joint_acceleration':'3.0','arm_handover_time':'1.0',
            'observation_names':','.join(arm.ACTOR_OBS_ORDER),
        }
        self.mock=patch.object(base.onnxruntime,'InferenceSession',Session)
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def runner(self,**kwargs):
        self.path.write_text(yaml.safe_dump(self.cfg))
        return arm.ArmInterventionRunner(self.path,'unused.onnx',**kwargs)

    def state(self,runner,q=None,dq=None):
        return legacy.Tests.state(self,runner,q,dq)

    def test_exact_layout_and_measured_state_without_privileged_palms(self):
        r=self.runner()
        q=r.default_q+np.arange(27)*.001
        dq=np.arange(27)*.03
        gravity,_=self.state(r,q,dq)
        for key in '17W': r.controls.press(key)
        obs=r.build_obs()
        np.testing.assert_allclose(obs[:3],[.1,-.2,.3],atol=1e-7)
        np.testing.assert_allclose(obs[3:6],gravity,atol=1e-7)
        np.testing.assert_allclose(obs[6:9],[.1,0,0])
        np.testing.assert_allclose(obs[11:38],q-r.default_q,atol=1e-7)
        np.testing.assert_allclose(obs[38:65],dq,atol=1e-7)
        self.assertFalse(obs[65:92].any())
        np.testing.assert_array_equal(obs[92:94],[1,0])
        self.assertFalse(obs[94:].any())  # Executed reference is still neutral.
        self.assertEqual(obs.shape,(108,))
        self.assertEqual(obs.dtype,np.float32)

    def test_discarded_arm_outputs_and_executed_action_history(self):
        r=self.runner()
        self.state(r)
        for key in '17': r.controls.press(key)
        before=r.target_q.copy()
        r.infer()
        np.testing.assert_allclose(r.target_q[:13],r.raw_action[:13]*r.action_scale[:13]+r.action_offset[:13])
        self.assertFalse(np.allclose(r.last_action[13:20],r.raw_action[13:20]))
        self.assertLessEqual(np.abs(r.target_q[13:20]-before[13:20]).max(),r.mux.max_speed*r.step_dt+1e-7)
        np.testing.assert_allclose(r.last_action,(r.target_q-r.action_offset)/r.action_scale,atol=1e-7)
        obs=r.build_obs()
        np.testing.assert_allclose(obs[65:92],r.last_action)
        expected=(r.target_q[r.mux.indices]-r.default_q[r.mux.indices])*np.repeat([1,0],7)
        np.testing.assert_allclose(obs[94:],expected,atol=1e-7)

    def test_walking_changes_do_not_restart_arm_interpolation(self):
        r=self.runner()
        self.state(r)
        r.controls.press('7')
        for _ in range(30): r.infer()
        elapsed=r.mux.elapsed
        r.controls.press('W')
        r.infer()
        self.assertAlmostEqual(r.mux.elapsed,elapsed+r.step_dt)
        r.controls.press('0')
        for _ in range(300): r.infer()
        self.assertFalse(r.mux.control_mask.any())
        np.testing.assert_allclose(r.target_q,r.raw_action*r.action_scale+r.action_offset)

    def test_policy_only_and_bad_actions_stop_before_publish(self):
        r=self.runner(arm_control='policy')
        self.state(r)
        r.controls.press('7')
        r.infer()
        self.assertFalse(r.mux.control_mask.any())
        np.testing.assert_allclose(r.target_q,r.raw_action*r.action_scale+r.action_offset)
        legacy.Tests.control_ready(self,r)
        r.session.action[:]=np.nan
        r.ControlStep()
        self.assertTrue(r.stop_event.is_set())
        r.lowcmd_publisher_.Write.assert_not_called()

    def test_metadata_shape_limits_and_hardware_rejection_before_dds(self):
        with self.assertRaisesRegex(ValueError,'simulation only'): self.runner(simulation=False)
        Session.metadata['arm_control_contract']='old'
        with self.assertRaisesRegex(ValueError,'arm_control_contract'): self.runner()
        Session.metadata['arm_control_contract']='intervention_v1'
        Session.metadata['arm_action_indices']='0,1'
        with self.assertRaisesRegex(ValueError,'indices'): self.runner()
        Session.metadata['arm_action_indices']=','.join(str(i) for i in range(13,27))
        Session.inputs[0].shape=[1,112]
        with self.assertRaisesRegex(ValueError,'ONNX'): self.runner()
        Session.inputs[0].shape=[1,108]
        self.cfg['arm_max_joint_speed']=0
        with self.assertRaisesRegex(ValueError,'limits'): self.runner()

    def test_demo_sequence_and_validation_do_not_import_sdk(self):
        with patch.dict(sys.modules,{'unitree_sdk2py':None}):
            r=self.runner(arm_control='demo')
            r.validate()
            r.policy_time=2.
            r.infer()
            self.assertTrue(r.mux.masks.all())
            r.policy_time=6.
            r.infer()
            age=r.mux.elapsed
            r.policy_time=10.
            r.infer()
            self.assertAlmostEqual(r.command[0],.3)
            self.assertAlmostEqual(r.mux.elapsed,age+r.step_dt)
            r.policy_time=29.
            r.infer()
            self.assertFalse(r.mux.masks.any())
            self.assertTrue(r.mux.releasing.all())

    def test_numpy_target_mux_matches_training_through_takeover_and_release(self):
        training=ROOT.parent/'unitree_rl_mjlab'
        helper=training/'tests/test_h1_2_arm_intervention.py'
        if not helper.exists() or importlib.util.find_spec('torch') is None:
            self.skipTest('Optional local training dependency is unavailable')
        sys.path.insert(0,str(training))
        try:
            import torch
            spec=importlib.util.spec_from_file_location('training_arm_helpers',helper)
            module=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            env,term,_=module.make_env(n=1,mode=(1.,0.,0.,0.))
            r=self.runner()
            mux=r.mux
            np.testing.assert_allclose(mux.lower,term.lower[0].numpy(),atol=1e-6)
            np.testing.assert_allclose(mux.upper,term.upper[0].numpy(),atol=1e-6)
            np.testing.assert_allclose(mux.ranges,term.ranges.numpy(),atol=1e-6)
            for step in range(360):
                if step in (0,110,210):
                    name,masks={0:('reach',[1,0]),110:('wide',[1,1]),210:('neutral',[0,0])}[step]
                    target=arm.pose(mux,name).ravel()
                    mux.set_targets(target,masks,duration=1.5)
                    term.set_external_targets(torch.tensor(target)[None],torch.tensor(masks)[None],duration=1.5)
                action=(np.sin(np.arange(27)+step*.01)*.1).astype(np.float32)
                actual=mux.advance(action,.02)
                env.action_manager.process_action(torch.tensor(action)[None])
                np.testing.assert_allclose(actual,term.executed_targets[0].numpy(),atol=3e-5,rtol=1e-5)
                expected=np.r_[term.control_mask[0].numpy(),
                    ((term.executed_targets[0,term.arm_action_ids]-term.arm_default[0])*
                      term.control_mask[0].repeat_interleave(7)).numpy()]
                np.testing.assert_allclose(mux.observation(),expected,atol=3e-5,rtol=1e-5)
        finally:
            sys.path.remove(str(training))


if __name__=='__main__': unittest.main()
