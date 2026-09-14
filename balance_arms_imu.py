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

# ZERO_ELBOW = 1.48
ZERO_ELBOW = 1.3  # rad, elbow angle when arm is straight down

GAIN = 1.5
SIGN_PITCH = 1.0
FILTER_ALPHA = .3

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
    Returns the elbow angle (rad)
    """
    return zero_elbow - (math.pi/2) - shoulder_angle


ELBOW_TARGET_POS = elbow_from_shoulder(SHOULDER_TARGET_POS, ZERO_ELBOW)


def pitch_from_quaternion(quaternion):
    """
    Returns pitch (rad) from an imu_state.quaternion [w, x, y, z].
    """
    w, x, y, z = quaternion
    return math.asin(clamp(2 * (w * y - z * x), -1.0, 1.0))


def hand_offset_from_shoulder(shoulder_angle, elbow_angle, torso_pitch, zero_elbow,
                               l1=UPPER_ARM_LENGTH, l2=FOREARM_LENGTH):
    """
    Forward kinematics check: returns the hand position (x, z) in the torso frame, given shoulder and elbow angles (rad)
    """
    upper_arm_world = shoulder_angle + torso_pitch
    forearm_world = shoulder_angle + elbow_angle - zero_elbow + torso_pitch
    x = -l1 * math.sin(upper_arm_world) - l2 * math.sin(forearm_world)
    z = -l1 * math.cos(upper_arm_world) - l2 * math.cos(forearm_world)
    return x, z


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

        self.shoulder_start = None       # frozen at the phase1 / phase2 transition
        self.elbow_start = None          # frozen at the phase1 / phase2 transition
        self.shoulder_correction = 0.0   # rate-limited, added onto shoulder_start
        self.elbow_correction = 0.0      # rate-limited, added onto elbow_start
        self.pitch_zero = None           # captured at the phase1 / phase2 transition, see LowCmdWrite
        self.phase2_entered = False

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

        elbow_start = self.q_start[H1_2_JointIndex.LeftElbow]
        shoulder_start = self.q_start[H1_2_JointIndex.LeftShoulderPitch]

        if self.time_ < POSITION_DURATION:
            ratio = self.time_ / POSITION_DURATION
            elbow_target = elbow_start + (ELBOW_TARGET_POS - elbow_start) * ratio
            shoulder_target = shoulder_start + (SHOULDER_TARGET_POS - shoulder_start) * ratio

        else:
            # 2. IMU pitch compensation
            pitch = pitch_from_quaternion(self.low_state.imu_state.quaternion)

            if not self.phase2_entered:
                self.phase2_entered = True
                self.shoulder_start = SHOULDER_TARGET_POS
                self.elbow_start = ELBOW_TARGET_POS
                self.pitch_zero = pitch
                self.filtered_pitch = pitch

            self.filtered_pitch += FILTER_ALPHA * (pitch - self.filtered_pitch)
            delta_pitch = self.filtered_pitch - self.pitch_zero

            correction_target = clamp(-SIGN_PITCH * GAIN * delta_pitch, -MAX_CORRECTION, MAX_CORRECTION)
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

            shoulder_target = self.shoulder_start + self.shoulder_correction
            elbow_target = self.elbow_start + self.elbow_correction

        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine_

        for i in range(H1_2_NUM_MOTOR):
            self.low_cmd.motor_cmd[i].mode = 1
            self.low_cmd.motor_cmd[i].tau = 0.0
            self.low_cmd.motor_cmd[i].dq = 0.0

            if i == H1_2_JointIndex.LeftElbow:
                self.low_cmd.motor_cmd[i].q = elbow_target
            elif i == H1_2_JointIndex.LeftShoulderPitch:
                self.low_cmd.motor_cmd[i].q = shoulder_target
            else:
                self.low_cmd.motor_cmd[i].q = self.q_start[i]

            self.low_cmd.motor_cmd[i].kp = HOLD_KP if i < 13 else HOLD_KP_ARM
            self.low_cmd.motor_cmd[i].kd = HOLD_KD

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

        self.print_counter += 1
        if self.print_counter % 250 == 0 and self.phase2_entered:
            # hand_z drifting away from its phase2-start value as pitch grows
            # is expected with ELBOW_SHARE > 0 -- that's the precision being
            # traded away for elbow-dominant motion. Use this to tune it.
            hand_x, hand_z = hand_offset_from_shoulder(
                shoulder_target, elbow_target, self.filtered_pitch, ZERO_ELBOW
            )
            print(
                f"shoulder={shoulder_target:+.3f}  elbow={elbow_target:+.3f}  "
                f"delta_pitch={self.filtered_pitch - self.pitch_zero:+.3f}  "
                f"hand=(x={hand_x:+.4f}, z={hand_z:+.4f})"
            )


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