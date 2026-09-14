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

SHOULDER_TARGET_POS = -1.0
POSITION_DURATION = 4.0 # seconds

ZERO_ELBOW_LEFT = 1.3   # rad, elbow angle when arm is straight down
ZERO_ELBOW_RIGHT = 1.3  # rad, elbow angle when arm is straight down

GAIN = 1.5
SIGN_PITCH = 1.0
FILTER_ALPHA = 1.0

UPPER_ARM_LENGTH = 0.3425   # m, shoulder to elbow
FOREARM_LENGTH = 0.3425     # m, elbow to hand

ELBOW_SHARE = 0.8

MAX_CORRECTION = 0.8                # rad, safety clamp on the combined correction
MAX_SHOULDER_CORRECTION_RATE = 1.0  # rad/s, safety slew limit
MAX_ELBOW_CORRECTION_RATE = 1.0     # rad/s, safety slew limit

HOLD_KP = 100.0
HOLD_KP_ARM = 50.0
HOLD_KD = 1.0

CONTROL_DT = 0.002


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def elbow_from_shoulder(shoulder_angle, zero_elbow):
    """
    Returns the elbow angle (rad) that keeps the forearm level for the given
    shoulder angle.
    """
    return zero_elbow - (math.pi/2) - shoulder_angle


def pitch_from_quaternion(quaternion):
    """
    Returns pitch (rad) from an imu_state.quaternion [w, x, y, z].
    """
    w, x, y, z = quaternion
    return math.asin(clamp(2 * (w * y - z * x), -1.0, 1.0))


def hand_offset_from_shoulder(shoulder_angle, elbow_angle, torso_pitch, zero_elbow,
                               l1=UPPER_ARM_LENGTH, l2=FOREARM_LENGTH):
    """
    Forward kinematics check: returns the hand position (x, z) in the torso
    frame, given shoulder and elbow angles (rad).
    """
    upper_arm_world = shoulder_angle + torso_pitch
    forearm_world = shoulder_angle + elbow_angle - zero_elbow + torso_pitch
    x = -l1 * math.sin(upper_arm_world) - l2 * math.sin(forearm_world)
    z = -l1 * math.cos(upper_arm_world) - l2 * math.cos(forearm_world)
    return x, z


class Arm:
    """
    Represents one arm of the robot, with its own shoulder and elbow joints.
    """
    def __init__(self, name, shoulder_index, elbow_index, zero_elbow):
        self.name = name
        self.shoulder_index = shoulder_index
        self.elbow_index = elbow_index
        self.zero_elbow = zero_elbow
        self.elbow_target_pos = elbow_from_shoulder(SHOULDER_TARGET_POS, zero_elbow)

        self.shoulder_start = None       # frozen at the phase1 / phase2 transition
        self.elbow_start = None          # frozen at the phase1 / phase2 transition
        self.shoulder_correction = 0.0   # rate-limited, added onto shoulder_start
        self.elbow_correction = 0.0      # rate-limited, added onto elbow_start
        self.phase2_entered = False

        self.shoulder_target = None
        self.elbow_target = None

    def update(self, q_start, time_, correction_target):
        """
        correction_target is the shared (arm-independent) raw pitch
        correction; both arms split and rate-limit it independently so a
        difference in starting pose between the arms doesn't couple them.
        """
        shoulder_start_q = q_start[self.shoulder_index]
        elbow_start_q = q_start[self.elbow_index]

        if time_ < POSITION_DURATION:
            ratio = time_ / POSITION_DURATION
            self.elbow_target = elbow_start_q + (self.elbow_target_pos - elbow_start_q) * ratio
            self.shoulder_target = shoulder_start_q + (SHOULDER_TARGET_POS - shoulder_start_q) * ratio
            return

        if not self.phase2_entered:
            self.phase2_entered = True
            self.shoulder_start = SHOULDER_TARGET_POS
            self.elbow_start = self.elbow_target_pos

        elbow_correction_target = ELBOW_SHARE * correction_target
        shoulder_correction_target = (1.0 - ELBOW_SHARE) * correction_target

        max_elbow_step = MAX_ELBOW_CORRECTION_RATE * CONTROL_DT
        self.elbow_correction += clamp(
            elbow_correction_target - self.elbow_correction, -max_elbow_step, max_elbow_step
        )
        max_shoulder_step = MAX_SHOULDER_CORRECTION_RATE * CONTROL_DT
        self.shoulder_correction += clamp(
            shoulder_correction_target - self.shoulder_correction, -max_shoulder_step, max_shoulder_step
        )

        self.shoulder_target = self.shoulder_start + self.shoulder_correction
        self.elbow_target = self.elbow_start + self.elbow_correction

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

        self.filtered_pitch = 0.0     # seeded from the first real reading, see LowStateHandler
        self.pitch_zero = None        # captured at the phase1 / phase2 transition, see LowCmdWrite
        self.phase2_entered = False

        self.arms = [
            Arm("left", H1_2_JointIndex.LeftShoulderPitch, H1_2_JointIndex.LeftElbow, ZERO_ELBOW_LEFT),
            Arm("right", H1_2_JointIndex.RightShoulderPitch, H1_2_JointIndex.RightElbow, ZERO_ELBOW_RIGHT),
        ]

        self.print_counter = 0

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

            self.update_mode_machine_ = True

    def LowCmdWrite(self):
        self.time_ += CONTROL_DT

        if self.time_ >= POSITION_DURATION:
            # 2. IMU pitch compensation
            pitch = pitch_from_quaternion(self.low_state.imu_state.quaternion)

            if not self.phase2_entered:
                self.phase2_entered = True
                self.pitch_zero = pitch
                self.filtered_pitch = pitch

            self.filtered_pitch += FILTER_ALPHA * (pitch - self.filtered_pitch)
            delta_pitch = self.filtered_pitch - self.pitch_zero
            correction_target = clamp(-SIGN_PITCH * GAIN * delta_pitch, -MAX_CORRECTION, MAX_CORRECTION)
        else:
            correction_target = 0.0  # unused during phase 1

        targets = {}
        for arm in self.arms:
            arm.update(self.q_start, self.time_, correction_target)
            targets[arm.shoulder_index] = arm.shoulder_target
            targets[arm.elbow_index] = arm.elbow_target

        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine_

        for i in range(H1_2_NUM_MOTOR):
            self.low_cmd.motor_cmd[i].mode = 1
            self.low_cmd.motor_cmd[i].tau = 0.0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].q = targets.get(i, self.q_start[i])
            self.low_cmd.motor_cmd[i].kp = HOLD_KP if i < 13 else HOLD_KP_ARM
            self.low_cmd.motor_cmd[i].kd = HOLD_KD

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

        self.print_counter += 1
        if self.print_counter % 250 == 0 and self.phase2_entered:
            # hand_z drifting away from its phase2-start value as pitch grows
            # is expected with ELBOW_SHARE > 0 -- that's the precision being
            # traded away for elbow-dominant motion. Use this to tune it.
            parts = [f"delta_pitch={self.filtered_pitch - self.pitch_zero:+.3f}"]
            for arm in self.arms:
                hand_x, hand_z = arm.hand_offset(self.filtered_pitch)
                parts.append(
                    f"{arm.name}: shoulder={arm.shoulder_target:+.3f} elbow={arm.elbow_target:+.3f} "
                    f"hand=(x={hand_x:+.4f}, z={hand_z:+.4f})"
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
