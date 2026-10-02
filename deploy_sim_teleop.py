import sys
import time
import math
import threading
import select

import numpy as np
import yaml
import onnxruntime

try:
    import termios
    import tty
    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False

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
    w, x, y, z = q
    qvec = np.array([x, y, z])
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(qvec, v) * 2.0 * w
    c = qvec * (2.0 * np.dot(qvec, v))
    return a - b + c


class TeleopThread(threading.Thread):
    """Non-blocking keyboard reader for Linux/Mac (works flawlessly over SSH)."""
    def __init__(self, runner):
        super().__init__()
        self.runner = runner
        self.daemon = True

    def run(self):
        if not HAS_TERMIOS:
            print("Termios not found. Teleop is disabled (Windows unsupported).")
            return

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        new_settings = termios.tcgetattr(fd)
        # Disable canonical mode and echo so keys are read instantly and silently
        new_settings[3] = new_settings[3] & ~termios.ICANON & ~termios.ECHO

        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)
            print("\n=== TELEOP ACTIVE ===")
            print("Locomotion (Hold to move):")
            print("  W/S : Forward / Backward")
            print("  A/D : Strafe Left / Right")
            print("  Q/E : Rotate Left / Right")
            print("  Space : Force Stop")
            print("\nArms (Tap to adjust):")
            print("  U/J : Shoulders Up / Down")
            print("  I/K : Elbows Flex / Extend")
            print("  X   : Reset arms to default")
            print("=====================\n")

            while True:
                # Wait 0.05s for input. If none, loop continues.
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    key = sys.stdin.read(1).lower()
                    
                    if key == '\x03':  # Ctrl+C
                        print("\nExiting...")
                        sys.exit(0)

                    self.runner.time_since_last_cmd = 0.0

                    if key == 'w': self.runner.cmd_vx = 0.7
                    elif key == 's': self.runner.cmd_vx = -0.4
                    elif key == 'a': self.runner.cmd_vy = 0.3
                    elif key == 'd': self.runner.cmd_vy = -0.3
                    elif key == 'q': self.runner.cmd_wz = 0.6
                    elif key == 'e': self.runner.cmd_wz = -0.6
                    elif key == 'u': self.runner.arm_shoulder_offset -= 0.05
                    elif key == 'j': self.runner.arm_shoulder_offset += 0.05
                    elif key == 'i': self.runner.arm_elbow_offset -= 0.05
                    elif key == 'k': self.runner.arm_elbow_offset += 0.05
                    elif key == 'x': 
                        self.runner.arm_shoulder_offset = 0.0
                        self.runner.arm_elbow_offset = 0.0
                    elif key == ' ':
                        self.runner.cmd_vx = 0.0
                        self.runner.cmd_vy = 0.0
                        self.runner.cmd_wz = 0.0
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class PolicyRunner:
    def __init__(self, deploy_yaml, onnx_path, simulation=False):
        self.simulation = simulation

        cfg = yaml.safe_load(open(deploy_yaml))
        self.joint_map = list(cfg["joint_ids_map"])            
        self.stiffness = np.array(cfg["stiffness"], dtype=float)
        self.damping = np.array(cfg["damping"], dtype=float)
        self.default_q = np.array(cfg["default_joint_pos"], dtype=float)
        self.action_scale = np.array(cfg["action_scale"], dtype=float)
        self.action_offset = np.array(cfg.get("action_offset", self.default_q), dtype=float)
        
        self.step_dt = float(cfg["step_dt"])
        self.gait_period = float(cfg["gait_period"])
        self.obs_dim = int(cfg["obs_dim"])
        self.n = len(self.joint_map)

        self.obs_scales = {
            "lin_vel": float(cfg.get("obs_scale_lin_vel", 1.0)),
            "ang_vel": float(cfg.get("obs_scale_ang_vel", 1.0)),
            "dof_pos": float(cfg.get("obs_scale_dof_pos", 1.0)),
            "dof_vel": float(cfg.get("obs_scale_dof_vel", 1.0)),
        }

        self.command = np.array([0.0, 0.0, 0.0], dtype=float)
        
        # Teleop state variables
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        self.arm_shoulder_offset = 0.0
        self.arm_elbow_offset = 0.0
        self.time_since_last_cmd = 0.0

        self.session = onnxruntime.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.in_name = self.session.get_inputs()[0].name
        
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.crc = CRC()

        self.mode_machine_ = 0
        self.update_mode_machine_ = False
        self.q_start = None            

        self.time_ = 0.0
        self.policy_time = 0.0         
        self.time_since_infer = 1e9    
        self.last_action = np.zeros(self.n, dtype=np.float32)
        self.target_q = np.zeros(self.n, dtype=np.float32)

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
            time.sleep(0.1)
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

    def build_obs(self):
        ls = self.low_state
        q = np.array([ls.motor_state[self.joint_map[i]].q for i in range(self.n)])
        dq = np.array([ls.motor_state[self.joint_map[i]].dq for i in range(self.n)])

        base_ang_vel = np.array(ls.imu_state.gyroscope, dtype=float) * self.obs_scales["ang_vel"]
        quat = np.array(ls.imu_state.quaternion, dtype=float)
        proj_g = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))

        cmd_scaled = np.array([
            self.command[0] * self.obs_scales["lin_vel"],
            self.command[1] * self.obs_scales["lin_vel"],
            self.command[2] * self.obs_scales["ang_vel"]
        ], dtype=float)

        if np.linalg.norm(self.command) < 0.1:
            phase = np.zeros(2)
        else:
            gp = (self.policy_time % self.gait_period) / self.gait_period
            phase = np.array([math.sin(gp * 2 * math.pi), math.cos(gp * 2 * math.pi)])

        joint_pos_rel = (q - self.default_q) * self.obs_scales["dof_pos"]
        joint_vel_rel = dq * self.obs_scales["dof_vel"]

        obs = np.concatenate([
            base_ang_vel, proj_g, cmd_scaled, phase, 
            joint_pos_rel, joint_vel_rel, self.last_action,
        ]).astype(np.float32)
        
        return np.clip(obs, -100.0, 100.0)

    def infer(self):
        obs = self.build_obs()
        action = self.session.run(None, {self.in_name: obs[None, :]})[0][0]
        action = np.clip(action, -100.0, 100.0)
        
        self.last_action = action.astype(np.float32)
        self.target_q = action * self.action_scale + self.action_offset
        
        # --- MANUAL ARM OVERRIDES ---
        # The policy calculates standard balance targets for the arms, but we overwrite 
        # them here if manual offsets exist. The policy adapts on the next step.
        L_SHOULDER, L_ELBOW = 13, 16
        R_SHOULDER, R_ELBOW = 20, 23
        
        self.target_q[L_SHOULDER] = self.default_q[L_SHOULDER] + self.arm_shoulder_offset
        self.target_q[R_SHOULDER] = self.default_q[R_SHOULDER] + self.arm_shoulder_offset
        
        self.target_q[L_ELBOW] = self.default_q[L_ELBOW] + self.arm_elbow_offset
        self.target_q[R_ELBOW] = self.default_q[R_ELBOW] + self.arm_elbow_offset


    def ControlStep(self):
        self.time_ += CONTROL_DT
        self.time_since_last_cmd += CONTROL_DT

        # Teleop Auto-Stop Safety: 
        # Halt locomotion if no keys have been pressed for 0.3s.
        if self.time_since_last_cmd > 0.3:
            self.cmd_vx, self.cmd_vy, self.cmd_wz = 0.0, 0.0, 0.0
            
        self.command[0] = self.cmd_vx
        self.command[1] = self.cmd_vy
        self.command[2] = self.cmd_wz

        if self.time_ < RAMP_DURATION:
            r = self.time_ / RAMP_DURATION
        else:
            r = 1.0
            self.policy_time += CONTROL_DT
            self.time_since_infer += CONTROL_DT
            if self.time_since_infer >= self.step_dt:
                self.time_since_infer = 0.0
                self.infer()

        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine_

        for m in range(H1_2_NUM_MOTOR):
            mc = self.low_cmd.motor_cmd[m]
            mc.mode = 1
            mc.dq = 0.0
            mc.tau = 0.0

            if m in self.joint_map:
                idx = self.joint_map.index(m)
                if self.time_ < RAMP_DURATION:
                    q_target = self.q_start[m] + (self.default_q[idx] - self.q_start[m]) * r
                else:
                    q_target = self.target_q[idx]

                mc.q = float(q_target)
                mc.kp = float(self.stiffness[idx])
                mc.kd = float(self.damping[idx])
            else:
                mc.q = float(self.q_start[m])
                mc.kp = 60.0   
                mc.kd = 2.0    

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python deploy_policy.py <deploy.yaml> <policy.onnx> [simulation|<iface>]")
        sys.exit(1)

    deploy_yaml, onnx_path = sys.argv[1], sys.argv[2]
    arg = sys.argv[3] if len(sys.argv) > 3 else None
    simulation = arg == "simulation"

    if simulation:
        ChannelFactoryInitialize(1, "lo")
    elif arg is not None:
        ChannelFactoryInitialize(0, arg)
    else:
        ChannelFactoryInitialize(0)

    runner = PolicyRunner(deploy_yaml, onnx_path, simulation=simulation)
    runner.Init()
    runner.Start()
    
    # Start the background keyboard listener
    teleop_thread = TeleopThread(runner)
    teleop_thread.start()

    while True:
        time.sleep(1)