import time
import sys
import math

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

H1_2_NUM_MOTOR = 27


class H1_2_JointIndex:
    LeftHipYaw = 0
    LeftHipPitch = 1
    LeftHipRoll = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleRoll = 5
    RightHipYaw = 6
    RightHipPitch = 7
    RightHipRoll = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleRoll = 11
    WaistYaw = 12
    LeftShoulderPitch = 13
    LeftShoulderRoll = 14
    LeftShoulderYaw = 15
    LeftElbow = 16
    LeftWristRoll = 17
    LeftWristPitch = 18
    LeftWristYaw = 19
    RightShoulderPitch = 20
    RightShoulderRoll = 21
    RightShoulderYaw = 22
    RightElbow = 23
    RightWristRoll = 24
    RightWristPitch = 25
    RightWristYaw = 26


class Mode:
    PR = 0
    AB = 1

# settle the arms to the true HW zero first
ZERO_DURATION = 2.0
SHOULDER_TARGET_POS = -1.0
POSITION_DURATION = 4.0

# The elbow zero is the angle that keeps the forearm level in the world frame
ZERO_ELBOW_LEFT = math.pi / 2 - 0.2
ZERO_ELBOW_RIGHT = math.pi / 2 - 0.2

ZERO_WRIST_ROLL_LEFT = 0.0
ZERO_WRIST_ROLL_RIGHT = 0.0

GAIN_PITCH = 1.0 # proportional gain on torso pitch angle error
RATE_GAIN_PITCH = 0.2 # derivative gain on torso pitch angular velocity - damping
SIGN_PITCH = 1.0

# tune these mby:
GAIN_ROLL = 1.0
RATE_GAIN_ROLL = 0.2
SIGN_ROLL = 1.0
FILTER_ALPHA = 1 # low pass on both filtered_pitch(/roll) and their rates

UPPER_ARM_LENGTH = 0.3425 # m, shoulder to elbow
FOREARM_LENGTH = 0.3425 # m, elbow to hand

ELBOW_SHARE = 0.8 # how much of correction is done by elbow vs shouldr

MAX_CORRECTION_PITCH = 0.8 # rad, safety clamp on the combined pitch correction
MAX_SHOULDER_CORRECTION_RATE = 1.0  # rad/s, safety limit
MAX_ELBOW_CORRECTION_RATE = 1.0 # rad/s, safety limit

MAX_CORRECTION_ROLL = 0.8
MAX_WRIST_ROLL_CORRECTION_RATE = 1.0

HOLD_KP = 100.0
HOLD_KP_ARM = 50.0
ACTIVE_KP_ARM = 80.0
HOLD_KD = 1.0

CONTROL_DT = 0.002


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def smoothstep(ratio):
    """
    Cubic smoothstep: maps ratio in [0,1] to a position fraction with zero
    velocity at both ratio=0 and ratio=1 - SEE GEOGEBRA
    """
    u = clamp(ratio, 0.0, 1.0)
    return 3.0 * u * u - 2.0 * u * u * u


def elbow_from_shoulder(shoulder_angle, zero_elbow, torso_pitch=0.0):
    """
    Returns the elbow angle (rad) that keeps the forearm level in the world frame
    """
    return zero_elbow - (math.pi/2) - shoulder_angle - torso_pitch


def pitch_from_quaternion(quaternion):
    """
    Returns pitch (rad) from an imu_state.quaternion [w, x, y, z].
    """
    w, x, y, z = quaternion
    return math.asin(clamp(2 * (w * y - z * x), -1.0, 1.0))


def roll_from_quaternion(quaternion):
    """
    Returns roll (rad) from an imu_state.quaternion [w, x, y, z].
    """
    w, x, y, z = quaternion
    return math.atan2(2 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))


def hand_offset_from_shoulder(shoulder_angle, elbow_angle, torso_pitch, zero_elbow, l1=UPPER_ARM_LENGTH, l2=FOREARM_LENGTH):
    """
    Forward kinematics check: returns the hand position (x, z) in the torso
    frame, given shoulder and elbow angles
    """
    upper_arm_world = shoulder_angle + torso_pitch
    forearm_world = shoulder_angle + elbow_angle - zero_elbow + torso_pitch
    x = -l1 * math.sin(upper_arm_world) - l2 * math.sin(forearm_world)
    z = -l1 * math.cos(upper_arm_world) - l2 * math.cos(forearm_world)
    return x, z


class Arm:
    """
    Represents one arm of the robot, with its own shoulder, elbow, and wrist-
    roll joints

    Phases (by self.update time_ argument):
      0: [0, ZERO_DURATION): ramp to raw joint zero
      1: [ZERO_DURATION, ZERO_DURATION+POSITION_DURATION): ramp from zero to working pose
      2: [ZERO_DURATION+POSITION_DURATION, inf): feedback loop balancing
    """
    def __init__(self, name, shoulder_index, elbow_index, wrist_roll_index, zero_elbow, zero_wrist_roll):
        self.name = name
        self.shoulder_index = shoulder_index
        self.elbow_index = elbow_index
        self.wrist_roll_index = wrist_roll_index
        self.zero_elbow = zero_elbow
        self.zero_wrist_roll = zero_wrist_roll

        self.shoulder_start = None
        self.elbow_start = None
        self.wrist_roll_start = None
        self.shoulder_correction = 0.0
        self.elbow_correction = 0.0
        self.wrist_roll_correction = 0.0
        self.phase2_entered = False

        self.shoulder_target = None
        self.elbow_target = None
        self.wrist_roll_target = None

    def update(self, q_start, time_, correction_target_pitch, correction_target_roll, torso_pitch, torso_roll):
        """
        Updates the target angles for this arm shoulder, elbow, and wrist roll
        """
        shoulder_start_q = q_start[self.shoulder_index]
        elbow_start_q = q_start[self.elbow_index]
        wrist_roll_start_q = q_start[self.wrist_roll_index]

        if time_ < ZERO_DURATION:
            # PHASE 0: get to the zero HW position
            ratio = smoothstep(time_ / ZERO_DURATION)

            self.shoulder_target = shoulder_start_q * (1.0 - ratio)
            self.elbow_target = elbow_start_q * (1.0 - ratio)
            self.wrist_roll_target = wrist_roll_start_q * (1.0 - ratio)
            return

        if time_ < ZERO_DURATION + POSITION_DURATION:
            # Phase 1: ramp the shoulder from zero to the working pose, and derive
            # the elbow from it every tick via elbow_from_shoulder() to keep the elbow leveled.
            ratio = smoothstep((time_ - ZERO_DURATION) / POSITION_DURATION)

            self.shoulder_target = SHOULDER_TARGET_POS * ratio
            blended_zero_elbow = math.pi / 2 + ratio * (self.zero_elbow - math.pi / 2)
            self.elbow_target = elbow_from_shoulder(self.shoulder_target, blended_zero_elbow, torso_pitch * ratio)

            self.wrist_roll_target = ratio * (self.zero_wrist_roll - torso_roll)
            return

        if not self.phase2_entered:
            self.phase2_entered = True
            # start exactly where phase 1 ends
            self.shoulder_start = self.shoulder_target
            self.elbow_start = self.elbow_target
            self.wrist_roll_start = self.wrist_roll_target

        elbow_correction_target = ELBOW_SHARE * correction_target_pitch
        shoulder_correction_target = (1.0 - ELBOW_SHARE) * correction_target_pitch

        max_elbow_step = MAX_ELBOW_CORRECTION_RATE * CONTROL_DT
        self.elbow_correction += clamp(
            elbow_correction_target - self.elbow_correction, -max_elbow_step, max_elbow_step
        )
        max_shoulder_step = MAX_SHOULDER_CORRECTION_RATE * CONTROL_DT
        self.shoulder_correction += clamp(
            shoulder_correction_target - self.shoulder_correction, -max_shoulder_step, max_shoulder_step
        )

        max_wrist_roll_step = MAX_WRIST_ROLL_CORRECTION_RATE * CONTROL_DT
        self.wrist_roll_correction += clamp(
            correction_target_roll - self.wrist_roll_correction, -max_wrist_roll_step, max_wrist_roll_step
        )

        self.shoulder_target = self.shoulder_start + self.shoulder_correction
        self.elbow_target = self.elbow_start + self.elbow_correction
        self.wrist_roll_target = self.wrist_roll_start + self.wrist_roll_correction

    def hand_offset(self, torso_pitch):
        return hand_offset_from_shoulder(self.shoulder_target, self.elbow_target, torso_pitch, self.zero_elbow)


class Custom:
    def __init__(self, simulation=False):
        self.simulation = simulation
        self.time_ = 0.0
        self.mode_machine_ = 0
        self.update_mode_machine_ = False
        self.q_start = None
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.crc = CRC()

        self.filtered_pitch = 0.0
        self.filtered_pitch_rate = 0.0
        self.pitch_zero = None
        self.filtered_roll = 0.0
        self.filtered_roll_rate = 0.0
        self.roll_zero = None
        self.phase2_entered = False

        self.arms = [
            Arm("left", H1_2_JointIndex.LeftShoulderPitch, H1_2_JointIndex.LeftElbow, H1_2_JointIndex.LeftWristRoll,
                ZERO_ELBOW_LEFT, ZERO_WRIST_ROLL_LEFT),
            Arm("right", H1_2_JointIndex.RightShoulderPitch, H1_2_JointIndex.RightElbow, H1_2_JointIndex.RightWristRoll,
                ZERO_ELBOW_RIGHT, ZERO_WRIST_ROLL_RIGHT),
        ]

        self.active_indices = {
            i for arm in self.arms for i in (arm.shoulder_index, arm.elbow_index, arm.wrist_roll_index)
        }

        # for debug printing only
        self.print_counter = 0
        self.last_wall_time = None
        self.dt_sum = 0.0
        self.dt_max = 0.0
        self.dt_samples = 0

    def Init(self):
        if not self.simulation:
            self.msc = MotionSwitcherClient()
            self.msc.SetTimeout(5.0)
            self.msc.Init()

            status, result = self.msc.CheckMode()
            while result['name']:
                self.msc.ReleaseMode()
                status, result = self.msc.CheckMode()
                time.sleep(1)

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=CONTROL_DT, target=self.LowCmdWrite, name="control"
        )
        while not self.update_mode_machine_:
            time.sleep(1)
        self.lowCmdWriteThreadPtr.Start()

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg
        if not self.update_mode_machine_:
            self.mode_machine_ = self.low_state.mode_machine
            self.q_start = [self.low_state.motor_state[i].q for i in range(H1_2_NUM_MOTOR)]

            self.filtered_pitch = pitch_from_quaternion(self.low_state.imu_state.quaternion)
            self.filtered_roll = roll_from_quaternion(self.low_state.imu_state.quaternion)

            self.update_mode_machine_ = True

    def LowCmdWrite(self):
        self.time_ += CONTROL_DT

        # for debug printing only
        now = time.perf_counter()
        if self.last_wall_time is not None:
            dt_actual = now - self.last_wall_time
            self.dt_sum += dt_actual
            self.dt_max = max(self.dt_max, dt_actual)
            self.dt_samples += 1
        self.last_wall_time = now

        pitch = pitch_from_quaternion(self.low_state.imu_state.quaternion)
        roll = roll_from_quaternion(self.low_state.imu_state.quaternion)
        pitch_rate = self.low_state.imu_state.gyroscope[1]
        roll_rate = self.low_state.imu_state.gyroscope[0]

        self.filtered_pitch += FILTER_ALPHA * (pitch - self.filtered_pitch)
        self.filtered_pitch_rate += FILTER_ALPHA * (pitch_rate - self.filtered_pitch_rate)
        self.filtered_roll += FILTER_ALPHA * (roll - self.filtered_roll)
        self.filtered_roll_rate += FILTER_ALPHA * (roll_rate - self.filtered_roll_rate)

        if self.time_ >= ZERO_DURATION + POSITION_DURATION:
            # IMU pitch+roll compensation
            if not self.phase2_entered:
                self.phase2_entered = True
                self.pitch_zero = self.filtered_pitch
                self.roll_zero = self.filtered_roll

            delta_pitch = self.filtered_pitch - self.pitch_zero
            correction_target_pitch = clamp(
                -SIGN_PITCH * (GAIN_PITCH * delta_pitch + RATE_GAIN_PITCH * self.filtered_pitch_rate),
                -MAX_CORRECTION_PITCH, MAX_CORRECTION_PITCH
            )

            delta_roll = self.filtered_roll - self.roll_zero
            correction_target_roll = clamp(
                -SIGN_ROLL * (GAIN_ROLL * delta_roll + RATE_GAIN_ROLL * self.filtered_roll_rate),
                -MAX_CORRECTION_ROLL, MAX_CORRECTION_ROLL
            )
        else:
             # phase 0 / 1
            correction_target_pitch = 0.0
            correction_target_roll = 0.0

        targets = {}
        for arm in self.arms:
            arm.update(self.q_start, self.time_, correction_target_pitch, correction_target_roll, pitch, roll)
            targets[arm.shoulder_index] = arm.shoulder_target
            targets[arm.elbow_index] = arm.elbow_target
            targets[arm.wrist_roll_index] = arm.wrist_roll_target

        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine_

        for i in range(H1_2_NUM_MOTOR):
            self.low_cmd.motor_cmd[i].mode = 1
            self.low_cmd.motor_cmd[i].tau = 0.0
            self.low_cmd.motor_cmd[i].q = targets.get(i, self.q_start[i])
            self.low_cmd.motor_cmd[i].dq = 0.0
            if i in self.active_indices:
                self.low_cmd.motor_cmd[i].kp = ACTIVE_KP_ARM
            else:
                self.low_cmd.motor_cmd[i].kp = HOLD_KP if i < 13 else HOLD_KP_ARM
            self.low_cmd.motor_cmd[i].kd = HOLD_KD

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

        self.print_counter += 1
        if self.print_counter % 250 == 0:
            phase = 0 if self.time_ < ZERO_DURATION else (
                1 if self.time_ < ZERO_DURATION + POSITION_DURATION else 2
            )
            loop_str = "loop: n/a"
            if self.dt_samples:
                avg_ms = self.dt_sum / self.dt_samples * 1000.0
                loop_str = f"loop: avg={avg_ms:.2f}ms ({1000.0/avg_ms:.0f}Hz) max={self.dt_max*1000:.2f}ms (target 2.00ms/500Hz)"
            self.dt_sum = 0.0
            self.dt_max = 0.0
            self.dt_samples = 0
            parts = [
                loop_str,
                f"t={self.time_:5.2f} phase={phase}",
                f"pitch={pitch:+.3f} pitch_rate={pitch_rate:+.3f}",
                f"roll={roll:+.3f} roll_rate={roll_rate:+.3f}",
            ]
            for arm in self.arms:
                hand_x, hand_z = arm.hand_offset(pitch)
                parts.append(
                    f"{arm.name}: shoulder={arm.shoulder_target:+.3f} elbow={arm.elbow_target:+.3f} "
                    f"wrist_roll={arm.wrist_roll_target:+.3f} hand=(x={hand_x:+.4f}, z={hand_z:+.4f})"
                )
            print("  ".join(parts))


if __name__ == '__main__':
    print("WARNING: this releases the robots built in balance controller for ALL joints.")
    input("Press Enter to continue...")

    simulation = len(sys.argv) > 1 and sys.argv[1] == "simulation"

    if simulation:
        # unitree_mujoco defaults, see simulate_python/config.py
        ChannelFactoryInitialize(1, "lo")
    elif len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom(simulation=simulation)
    custom.Init()
    custom.Start()

    while True:
        time.sleep(1)