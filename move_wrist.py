import time
import sys

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

H1_2_NUM_MOTOR = 27

# left elbow from apprx 1.26 rad to 0.33 rad
# left shoulder roll from apprx 0.026 rad to -1.15 rad
# then left wrist roll rotate from -0.119 rad  to 1.6 to -1.4 

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


# what moves
TARGET_JOINT = H1_2_JointIndex.LeftWristRoll
TARGET_DELTA = 0.15   # rad
TARGET_KP = 30.0      # in examples there is 50
TARGET_KD = 1.0

HOLD_KP = 100.0  # legs/waist matches the official examples
HOLD_KP_ARM = 50.0
HOLD_KD = 1.0

RAMP_DURATION = 3.0   # seconds to reach TARGET_DELTA
CONTROL_DT = 0.002    # 2ms, matches the official example


class Custom:
    def __init__(self):
        self.time_ = 0.0
        self.mode_machine_ = 0
        self.update_mode_machine_ = False
        self.q_start = None
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.crc = CRC()

    def Init(self):
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
            self.update_mode_machine_ = True

    def LowCmdWrite(self):
        self.time_ += CONTROL_DT
        ratio = min(self.time_ / RAMP_DURATION, 1.0)

        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine_

        for i in range(H1_2_NUM_MOTOR):
            self.low_cmd.motor_cmd[i].mode = 1
            self.low_cmd.motor_cmd[i].tau = 0.0
            self.low_cmd.motor_cmd[i].dq = 0.0

            if i == TARGET_JOINT:
                # linear ramp from the captured start toward start+DELTA
                self.low_cmd.motor_cmd[i].q = self.q_start[i] + TARGET_DELTA * ratio
                self.low_cmd.motor_cmd[i].kp = TARGET_KP
                self.low_cmd.motor_cmd[i].kd = TARGET_KD
            else:
                # hold at the FIXED captured start -- not live q, see note above
                self.low_cmd.motor_cmd[i].q = self.q_start[i]
                self.low_cmd.motor_cmd[i].kp = HOLD_KP if i < 13 else HOLD_KP_ARM
                self.low_cmd.motor_cmd[i].kd = HOLD_KD

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)


if __name__ == '__main__':
    print("WARNING: this releases the robot's built-in balance controller for ALL joints,")
    print("not just the target one. Confirm the robot is secured/supported, not freestanding,")
    print("before continuing.")
    input("Press Enter to continue...")

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()
    custom.Init()
    custom.Start()

    while True:
        time.sleep(1)