"""Run an exported mjlab policy (policy.onnx) on the Unitree H1_2 via unitree_sdk2py.

Replaces the hand-crafted arm controller with the trained network. Structure mirrors
the classical-control script: a LowState subscriber + a RecurrentThread control loop.

Two-phase startup:
  1. ramp every joint from its power-on pose to the policy's default pose (safe, no jerk)
  2. hand control to the policy: 50 Hz inference, 500 Hz PD toward the held target

USAGE:
  # sim-to-sim against unitree_mujoco (recommended first):
  python deploy_policy.py deploy_tray.yaml policy.onnx simulation
  # real robot:
  python deploy_policy.py deploy_tray.yaml policy.onnx <network_interface>

BEFORE THE REAL ROBOT: validate in unitree_mujoco, and ADD THE TRAY to that sim's
scene (weld it to the hands, as in your training XML) so sim-to-sim actually exercises
the tray dynamics the policy trained with. On the real robot the tray is physical
(velcro) -- nothing to add in this script.
"""

import sys
import time
import math

import numpy as np
import yaml
import onnxruntime

from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

H1_2_NUM_MOTOR = 27
CONTROL_DT = 0.002        # 500 Hz control / PD send rate
RAMP_DURATION = 4.0       # s, power-on pose -> default pose


class Mode:
    PR = 0
    AB = 1


def quat_rotate_inverse(q, v):
    """Rotate vector v by the INVERSE of quaternion q=[w,x,y,z]. Numpy, single sample.

    This is how projected_gravity is computed in training: the world -Z axis
    expressed in the base frame. Upright robot -> ~[0, 0, -1].
    """
    w, x, y, z = q
    qvec = np.array([x, y, z])
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(qvec, v) * 2.0 * w
    c = qvec * (2.0 * np.dot(qvec, v))
    return a - b + c


class PolicyRunner:
    def __init__(self, deploy_yaml, onnx_path, simulation=False):
        self.simulation = simulation

        cfg = yaml.safe_load(open(deploy_yaml))
        self.joint_map = list(cfg["joint_ids_map"])            # policy idx -> SDK motor idx
        self.stiffness = np.array(cfg["stiffness"], dtype=float)
        self.damping = np.array(cfg["damping"], dtype=float)
        self.default_q = np.array(cfg["default_joint_pos"], dtype=float)
        self.action_scale = np.array(cfg["action_scale"], dtype=float)
        self.action_offset = np.array(cfg["action_offset"], dtype=float)
        self.step_dt = float(cfg["step_dt"])
        self.gait_period = float(cfg["gait_period"])
        self.obs_dim = int(cfg["obs_dim"])
        self.n = len(self.joint_map)

        # Standing command by default. [vx, vy, wz]. Change to walk.
        self.command = np.array([0.0, 0.0, 0.0], dtype=float)

        self.session = onnxruntime.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.in_name = self.session.get_inputs()[0].name
        exp_in = self.session.get_inputs()[0].shape[-1]
        assert exp_in == self.obs_dim, (
            f"ONNX expects obs dim {exp_in}, yaml says {self.obs_dim} -- obs mismatch"
        )

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.crc = CRC()

        self.mode_machine_ = 0
        self.update_mode_machine_ = False
        self.q_start = None            # power-on pose, captured on first LowState

        self.time_ = 0.0
        self.policy_time = 0.0         # elapsed time since policy engaged (for gait phase)
        self.time_since_infer = 1e9    # forces inference on the first policy tick
        self.last_action = np.zeros(self.n, dtype=np.float32)  # raw policy output (obs term)
        self.target_q = None           # held between inferences

    # ---- SDK plumbing ------------------------------------------------------

    def Init(self):
        if not self.simulation:
            self.msc = MotionSwitcherClient()
            self.msc.SetTimeout(5.0)
            self.msc.Init()
            status, result = self.msc.CheckMode()
            while result["name"]:
                self.msc.ReleaseMode()
                status, result = self.msc.CheckMode()
                time.sleep(1)

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

    def Start(self):
        self.thread = RecurrentThread(
            interval=CONTROL_DT, target=self.ControlStep, name="policy_control"
        )
        while not self.update_mode_machine_:
            time.sleep(1)
        self.target_q = self.default_q.copy()
        self.thread.Start()

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg
        if not self.update_mode_machine_:
            self.mode_machine_ = self.low_state.mode_machine
            self.q_start = np.array(
                [self.low_state.motor_state[i].q for i in range(H1_2_NUM_MOTOR)]
            )
            self.update_mode_machine_ = True

    # ---- observation -------------------------------------------------------

    def build_obs(self):
        ls = self.low_state
        # joints in POLICY order via the map
        q = np.array([ls.motor_state[self.joint_map[i]].q for i in range(self.n)])
        dq = np.array([ls.motor_state[self.joint_map[i]].dq for i in range(self.n)])

        base_ang_vel = np.array(ls.imu_state.gyroscope, dtype=float)          # (3,)
        quat = np.array(ls.imu_state.quaternion, dtype=float)                 # [w,x,y,z]
        proj_g = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))        # (3,)

        # gait phase: zeros when standing (|cmd| < 0.1), else sin/cos clock
        if np.linalg.norm(self.command) < 0.1:
            phase = np.zeros(2)
        else:
            gp = (self.policy_time % self.gait_period) / self.gait_period
            phase = np.array([math.sin(gp * 2 * math.pi), math.cos(gp * 2 * math.pi)])

        joint_pos_rel = q - self.default_q
        joint_vel_rel = dq

        obs = np.concatenate([
            base_ang_vel,          # 3
            proj_g,                # 3
            self.command,          # 3
            phase,                 # 2
            joint_pos_rel,         # n
            joint_vel_rel,         # n
            self.last_action,      # n
        ]).astype(np.float32)
        return obs

    def infer(self):
        obs = self.build_obs()
        action = self.session.run(None, {self.in_name: obs[None, :]})[0][0]
        self.last_action = action.astype(np.float32)          # raw output feeds next obs
        self.target_q = action * self.action_scale + self.action_offset

    # ---- control loop ------------------------------------------------------

    def ControlStep(self):
        self.time_ += CONTROL_DT

        if self.time_ < RAMP_DURATION:
            # Phase 1: ramp power-on pose -> default pose.
            r = self.time_ / RAMP_DURATION
            q_des = self.q_start + (self.default_q - self.q_start) * r
        else:
            # Phase 2: policy. Inference at step_dt, hold target between.
            self.policy_time += CONTROL_DT
            self.time_since_infer += CONTROL_DT
            if self.time_since_infer >= self.step_dt:
                self.time_since_infer = 0.0
                self.infer()
            q_des = self.target_q

        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine_
        for i in range(self.n):
            m = self.joint_map[i]                 # SDK motor index
            mc = self.low_cmd.motor_cmd[m]
            mc.mode = 1
            mc.q = float(q_des[i])
            mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = float(self.stiffness[i])
            mc.kd = float(self.damping[i])

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python deploy_policy.py <deploy.yaml> <policy.onnx> [simulation|<iface>]")
        sys.exit(1)

    deploy_yaml, onnx_path = sys.argv[1], sys.argv[2]
    arg = sys.argv[3] if len(sys.argv) > 3 else None
    simulation = arg == "simulation"

    print("WARNING: this releases the built-in balance controller and runs a learned policy.")
    print("Keep the e-stop ready. Start with the robot hung / supported.")
    input("Press Enter to continue...")

    if simulation:
        ChannelFactoryInitialize(1, "lo")     # unitree_mujoco defaults
    elif arg is not None:
        ChannelFactoryInitialize(0, arg)      # real robot on the given interface
    else:
        ChannelFactoryInitialize(0)

    runner = PolicyRunner(deploy_yaml, onnx_path, simulation=simulation)
    runner.Init()
    runner.Start()

    while True:
        time.sleep(1)