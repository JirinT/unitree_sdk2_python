"""Linux terminal teleoperation for the H1_2 whole-body hand-tracking actor.

Simulation: python deploy_sim_teleop.py deploy_hand_tracking.yaml policy.onnx simulation
Use --validate-only for an offline ONNX/FK check without opening any DDS channels.
"""

import argparse
import math
import os
from pathlib import Path
import select
import sys
import threading
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import onnxruntime
import yaml

H1_2_NUM_MOTOR = 27
CONTROL_DT = 0.002
RAMP_DURATION = 4.0
ACTOR_OBS_ORDER = ["base_ang_vel", "projected_gravity", "command", "phase",
                   "joint_hand_state", "joint_vel", "actions", "hand_command"]


class Mode:
    PR = 0
    AB = 1


def quat_rotate_inverse(q, v):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(q).all() or norm < 1e-8:
        raise ValueError("IMU quaternion must be finite nonzero wxyz")
    w, x, y, z = q / norm
    qvec = np.array([x, y, z])
    return v * (2*w*w - 1) - np.cross(qvec, v) * (2*w) + qvec * (2*np.dot(qvec, v))


class PalmKinematics:
    """Measured encoders -> palm positions and IMU orientation in pelvis frame."""
    def __init__(self, xml_path, joint_names, site_names, imu_site="imu", root="pelvis"):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.root = self.model.body(root).id
        self.sites = [self.model.site(name).id for name in site_names]
        self.imu = self.model.site(imu_site).id
        ids = [self.model.joint(name).id for name in joint_names]
        if any(self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in ids):
            raise ValueError("Expected 27 named hinge joints for H1_2 encoder FK")
        self.q_addresses = self.model.jnt_qposadr[ids]
        self.count = len(ids)

    def compute(self, q):
        q = np.asarray(q)
        if q.shape != (self.count,) or not np.isfinite(q).all():
            raise ValueError("Invalid encoder positions for palm FK")
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self.q_addresses] = q
        mujoco.mj_kinematics(self.model, self.data)  # no IK or physics integration
        rotation = self.data.xmat[self.root].reshape(3, 3)
        palms = (self.data.site_xpos[self.sites] - self.data.xpos[self.root]) @ rotation
        imu_rotation = rotation.T @ self.data.site_xmat[self.imu].reshape(3, 3)
        return palms.copy(), imu_rotation.copy()


class KeyboardCommands:
    """Thread-safe, latched commands. Layout matches the mjlab play controls."""
    def __init__(self, boxes, velocity_ranges):
        self.boxes = np.asarray(boxes, dtype=float)
        self.velocity_ranges = np.asarray(velocity_ranges, dtype=float)
        if (self.boxes.shape != (2, 3, 2) or not np.isfinite(self.boxes).all()
                or (self.boxes[:, :, 0] > self.boxes[:, :, 1]).any()):
            raise ValueError("Hand ranges must contain two finite ordered XYZ boxes")
        if (self.velocity_ranges.shape != (3, 2) or not np.isfinite(self.velocity_ranges).all()
                or (self.velocity_ranges[:, 0] > 0).any() or (self.velocity_ranges[:, 1] < 0).any()):
            raise ValueError("Velocity ranges must contain zero for all three axes")
        self.lock = threading.Lock()
        self.velocity = np.zeros(3)
        self.targets = self.boxes.mean(axis=2)
        self.masks = np.zeros(2)
        self.selected = (0,)

    def press(self, key):
        key = key.upper()
        with self.lock:
            if key in (" ", "B"):
                self.velocity[:] = 0
            elif key in "WSQEZX" and len(key) == 1:
                axis, delta = {"W": (0, .1), "S": (0, -.1), "Q": (1, .1),
                               "E": (1, -.1), "Z": (2, .2), "X": (2, -.2)}[key]
                self.velocity[axis] = np.clip(self.velocity[axis] + delta, *self.velocity_ranges[axis])
            elif key in ("1", "2", "3"):
                self.selected = {"1": (0,), "2": (1,), "3": (0, 1)}[key]
            elif key in ("4", "5"):
                side = int(key) - 4
                self.masks[side] = 1 - self.masks[side]
            elif key == "0":
                self.masks[:] = 0
            elif key == "C":
                for side in self.selected:
                    self.targets[side] = self.boxes[side].mean(axis=1)
            elif key in "IKJLUO" and len(key) == 1:
                axis, delta = {"I": (0, .02), "K": (0, -.02), "J": (1, .02),
                               "L": (1, -.02), "U": (2, .02), "O": (2, -.02)}[key]
                for side in self.selected:
                    self.targets[side, axis] = np.clip(self.targets[side, axis] + delta, *self.boxes[side, axis])
                    self.masks[side] = 1
            else:
                return False
        return True

    def snapshot(self):
        with self.lock:
            return self.velocity.copy(), self.targets.copy(), self.masks.copy()


class TeleopThread(threading.Thread):
    def __init__(self, runner):
        super().__init__(name="terminal_teleop", daemon=True)
        self.runner = runner

    @staticmethod
    def check_terminal():
        if not sys.stdin.isatty():
            raise ValueError("Run from an interactive Linux terminal (SSH: allocate a TTY with ssh -t)")

    def run(self):
        import termios
        import tty
        fd = sys.stdin.fileno()
        settings = termios.tcgetattr(fd)
        escape = None
        try:
            tty.setcbreak(fd)
            print("\nKeyboard: focus this terminal; tap keys, no Enter needed.")
            print("W/S forward/back; Q/E strafe; Z/X yaw; B or Space STOP walking.")
            print("1 left / 2 right / 3 both hands; I/K X, J/L Y, U/O Z (2 cm).")
            print("4/5 toggle left/right; 0 both off; C center selected targets. Ctrl+C exits.")
            print("Walking is latched until changed/stopped; hands start disabled.\n")
            while not self.runner.stop_event.is_set():
                if not select.select([fd], [], [], .05)[0]:
                    continue
                data = os.read(fd, 64)
                if not data:
                    self.runner.stop_event.set()
                    break
                for key in data.decode("ascii", errors="ignore"):
                    if key == "\x1b":
                        escape = "start"
                    elif escape == "start":
                        escape = "sequence" if key in "[O" else None
                    elif escape == "sequence":
                        if "@" <= key <= "~":
                            escape = None
                    elif key == "\x03":
                        self.runner.stop_event.set()
                    elif self.runner.controls.press(key):
                        velocity, targets, masks = self.runner.controls.snapshot()
                        print(f"twist={velocity.round(2)} masks={masks.astype(int)} "
                              f"left={targets[0].round(3)} right={targets[1].round(3)}")
        except Exception as error:
            self.runner.error = error
            self.runner.stop_event.set()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, settings)


class PolicyRunner:
    def __init__(self, deploy_yaml, onnx_path, simulation=False, robot_xml=None):
        self.simulation = simulation
        path = Path(deploy_yaml).resolve()
        with path.open() as stream:
            cfg = yaml.safe_load(stream)
        self.joint_map = list(cfg["joint_ids_map"])
        self.n = len(self.joint_map)
        self.joint_names = list(cfg["joint_names"])
        if (self.n != H1_2_NUM_MOTOR or set(self.joint_map) != set(range(H1_2_NUM_MOTOR))
                or len(self.joint_names) != self.n or len(set(self.joint_names)) != self.n):
            raise ValueError("Expected 27 unique policy joint names and an SDK motor permutation of 0..26")
        for attr, key in (("stiffness", "stiffness"), ("damping", "damping"),
                          ("default_q", "default_joint_pos"), ("action_scale", "action_scale")):
            values = np.asarray(cfg[key], dtype=float)
            if values.shape != (self.n,) or not np.isfinite(values).all():
                raise ValueError(f"{key} must have 27 finite values")
            setattr(self, attr, values)
        self.action_offset = np.asarray(cfg.get("action_offset", self.default_q), dtype=float)
        if self.action_offset.shape != (self.n,) or not np.isfinite(self.action_offset).all():
            raise ValueError("action_offset must have 27 finite values")
        self.step_dt, self.gait_period = float(cfg["step_dt"]), float(cfg["gait_period"])
        if (not math.isfinite(self.step_dt) or self.step_dt < CONTROL_DT
                or not math.isfinite(self.gait_period) or self.gait_period <= 0):
            raise ValueError("step_dt must be >= control dt and gait_period positive")
        self.obs_dim = int(cfg["obs_dim"])
        if self.obs_dim != 11 + 3*self.n + 20 or cfg.get("actor_obs_order") != ACTOR_OBS_ORDER:
            raise ValueError("Use deploy_hand_tracking.yaml: the current flat hand actor requires 112 values in its trained order")
        self.obs_scales = {key: float(cfg.get(name, 1.0)) for key, name in (
            ("lin_vel", "obs_scale_lin_vel"), ("ang_vel", "obs_scale_ang_vel"),
            ("dof_pos", "obs_scale_dof_pos"), ("dof_vel", "obs_scale_dof_vel"))}
        if not all(math.isfinite(v) and v > 0 for v in self.obs_scales.values()):
            raise ValueError("Observation scales must be finite and positive")
        # Current training uses raw twist commands; angular sensor scale is independent.
        self.command_scale = np.asarray(cfg.get("command_scale", [1., 1., 1.]), dtype=float)
        if self.command_scale.shape != (3,) or not np.isfinite(self.command_scale).all():
            raise ValueError("command_scale must have three finite values")
        xml = Path(robot_xml) if robot_xml else path.parent / cfg["robot_xml"]
        if not xml.is_file():
            raise FileNotFoundError(f"FK model not found: {xml}. Set robot_xml in YAML or pass --robot-xml.")
        self.fk = PalmKinematics(xml, self.joint_names, cfg["hand_sites"], cfg.get("imu_site", "imu"))
        self.imu_quaternion_frame = cfg.get("imu_quaternion_frame", "imu")
        if self.imu_quaternion_frame not in {"imu", "pelvis"}:
            raise ValueError("imu_quaternion_frame must be imu or pelvis, matching the validator's imu_quat sensor")
        self.controls = KeyboardCommands((cfg["left_hand_ranges"], cfg["right_hand_ranges"]), cfg["velocity_ranges"])
        self.session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or len(outputs) != 1 or len(inputs[0].shape) != 2
                or len(outputs[0].shape) != 2 or inputs[0].shape[1] != self.obs_dim
                or outputs[0].shape[1] != self.n
                or (isinstance(inputs[0].shape[0], int) and inputs[0].shape[0] != 1)
                or (isinstance(outputs[0].shape[0], int) and outputs[0].shape[0] != 1)):
            raise ValueError("Expected feedforward ONNX [1,112] -> [1,27]; export the current Flat-HandTracking actor")
        if inputs[0].type != "tensor(float)":
            raise ValueError("Expected float32 ONNX actor input")
        self.in_name = inputs[0].name
        metadata = self.session.get_modelmeta().custom_metadata_map
        for key, expected in (("observation_names", ACTOR_OBS_ORDER), ("joint_names", self.joint_names)):
            if key in metadata and metadata[key].split(",") != expected:
                raise ValueError(f"ONNX {key} do not match the deployment config; export the latest actor")
        for key, expected in (("default_joint_pos", self.default_q), ("action_scale", self.action_scale),
                              ("joint_stiffness", self.stiffness), ("joint_damping", self.damping)):
            if key in metadata:
                actual = np.fromstring(metadata[key], sep=",")
                # Export metadata rounds to 3 decimals; YAML retains full precision.
                if actual.shape != expected.shape or not np.allclose(actual, expected, atol=0.00051, rtol=0):
                    raise ValueError(f"ONNX {key} differs from deployment YAML")
        self.motor_to_policy = {motor: index for index, motor in enumerate(self.joint_map)}
        self.command = np.zeros(3)
        self.low_state = None
        self.mode_machine_ = 0
        self.update_mode_machine_ = False
        self.q_start = None
        self.time_ = self.policy_time = self.time_since_infer = 0.
        self.last_action = np.zeros(self.n, dtype=np.float32)
        self.target_q = self.default_q.copy()
        self.stop_event = threading.Event()
        self.error = None
        self.thread = None

    def Init(self):
        # SDK and Linux timerfd are imported only for a live validator connection.
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC
        if not self.simulation:
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
            self.msc = MotionSwitcherClient()
            self.msc.SetTimeout(5.)
            self.msc.Init()
            _, result = self.msc.CheckMode()
            while result["name"]:
                self.msc.ReleaseMode()
                _, result = self.msc.CheckMode()
                time.sleep(1)
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.crc = CRC()
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

    def Start(self):
        from unitree_sdk2py.utils.thread import RecurrentThread
        deadline = time.monotonic() + 15
        while not self.update_mode_machine_:
            if time.monotonic() >= deadline:
                raise TimeoutError("No H1_2 LowState received; check validator, HG DDS message type, domain=1 and loopback interface")
            time.sleep(.05)
        self.thread = RecurrentThread(interval=CONTROL_DT, target=self.ControlStep, name="policy_control")
        self.thread.Start()

    def Stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.Wait(timeout=1.)
        for name in ("lowstate_subscriber", "lowcmd_publisher_"):
            channel = getattr(self, name, None)
            if channel is not None and hasattr(channel, "Close"):
                channel.Close()

    def LowStateHandler(self, msg):
        self.low_state = msg
        if not self.update_mode_machine_:
            self.mode_machine_ = msg.mode_machine
            self.q_start = np.array([msg.motor_state[i].q for i in range(H1_2_NUM_MOTOR)])
            self.update_mode_machine_ = True

    def build_obs(self):
        ls = self.low_state  # One DDS message snapshot supplies all measured joints.
        if ls is None:
            raise RuntimeError("No LowState available")
        q = np.array([ls.motor_state[i].q for i in self.joint_map])
        dq = np.array([ls.motor_state[i].dq for i in self.joint_map])
        gyro = np.asarray(ls.imu_state.gyroscope, dtype=float)
        if dq.shape != (self.n,) or gyro.shape != (3,) or not np.isfinite(dq).all() or not np.isfinite(gyro).all():
            raise ValueError("Nonfinite joint velocity or IMU sample")
        palms, imu_rotation = self.fk.compute(q)
        gravity = quat_rotate_inverse(np.asarray(ls.imu_state.quaternion), np.array([0., 0., -1.]))
        if self.imu_quaternion_frame == "imu":
            gravity = imu_rotation @ gravity  # Train projected_gravity uses pelvis frame.
        command, targets, masks = self.controls.snapshot()
        self.command = command
        errors = np.where(masks[:, None].astype(bool), targets - palms, 0.)
        masked_targets = np.where(masks[:, None].astype(bool), targets, 0.)
        if np.linalg.norm(command) < .1:
            phase = np.zeros(2)
        else:
            angle = 2*math.pi*(self.policy_time % self.gait_period)/self.gait_period
            phase = np.array([math.sin(angle), math.cos(angle)])
        obs = np.concatenate((
            gyro*self.obs_scales["ang_vel"], gravity, command*self.command_scale, phase,
            (q-self.default_q)*self.obs_scales["dof_pos"], palms.ravel(), errors.ravel(),
            dq*self.obs_scales["dof_vel"], self.last_action, masked_targets.ravel(), masks,
        )).astype(np.float32)
        if obs.shape != (self.obs_dim,) or not np.isfinite(obs).all():
            raise ValueError("Invalid actor observation")
        return obs

    def infer(self):
        obs = self.build_obs()
        result = np.asarray(self.session.run(None, {self.in_name: obs[None]})[0])
        if result.shape != (1, self.n) or not np.isfinite(result).all():
            raise ValueError("ONNX actor returned invalid actions")
        # The training action term has no additional clipping. PPO controls ALL joints.
        self.last_action = result[0].astype(np.float32, copy=True)
        self.target_q = self.last_action*self.action_scale + self.action_offset
        self.policy_time += self.step_dt

    def ControlStep(self):
        if self.stop_event.is_set():
            return
        try:
            self._control_step()
        except Exception as error:
            self.error = error
            self.stop_event.set()
            print(f"Control stopped: {error}", file=sys.stderr)

    def _control_step(self):
        self.time_ += CONTROL_DT
        ramp = min(1., self.time_/RAMP_DURATION)
        if self.time_ >= RAMP_DURATION:
            self.time_since_infer += CONTROL_DT
            if self.time_since_infer + 1e-10 >= self.step_dt:
                self.time_since_infer = max(0., self.time_since_infer - self.step_dt)
                self.infer()
        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine_
        for motor in range(H1_2_NUM_MOTOR):
            index = self.motor_to_policy[motor]
            command = self.low_cmd.motor_cmd[motor]
            command.mode, command.dq, command.tau = 1, 0., 0.
            target = self.q_start[motor] + (self.default_q[index]-self.q_start[motor])*ramp
            command.q = float(target if self.time_ < RAMP_DURATION else self.target_q[index])
            command.kp, command.kd = float(self.stiffness[index]), float(self.damping[index])
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

    def validate(self):
        """One neutral-state inference, with no SDK import or DDS publishing."""
        motors = [SimpleNamespace(q=0., dq=0.) for _ in range(H1_2_NUM_MOTOR)]
        for index, motor in enumerate(self.joint_map):
            motors[motor].q = self.default_q[index]
        _, imu_rotation = self.fk.compute(self.default_q)
        quat = np.zeros(4)
        if self.imu_quaternion_frame == "imu":
            mujoco.mju_mat2Quat(quat, imu_rotation.ravel())
        else:
            quat[0] = 1
        self.low_state = SimpleNamespace(motor_state=motors, imu_state=SimpleNamespace(gyroscope=np.zeros(3), quaternion=quat))
        self.infer()
        print(f"Offline validation OK: actor {self.obs_dim} observations -> {self.n} finite actions; no DDS connection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deploy_yaml")
    parser.add_argument("onnx_path")
    parser.add_argument("target", nargs="?", default="simulation", help="simulation (domain 1, lo), or an explicit robot network interface")
    parser.add_argument("--robot-xml", help="Override the FK MJCF model (use the training robot geometry)")
    parser.add_argument("--validate-only", action="store_true", help="Offline config/FK/ONNX check; never initialize DDS")
    args = parser.parse_args()
    runner = PolicyRunner(args.deploy_yaml, args.onnx_path, args.target == "simulation", args.robot_xml)
    if args.validate_only:
        runner.validate()
        return
    TeleopThread.check_terminal()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    if runner.simulation:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, args.target)
    teleop = TeleopThread(runner)
    try:
        runner.Init()
        runner.Start()
        teleop.start()
        while not runner.stop_event.wait(.1):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop_event.set()
        if teleop.is_alive():
            teleop.join(timeout=1.)
        runner.Stop()
    if runner.error is not None:
        raise RuntimeError("Teleoperation stopped") from runner.error


if __name__ == "__main__":
    main()
