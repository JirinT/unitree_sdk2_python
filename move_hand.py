import time
import sys

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

from matplotlib import pyplot as plt

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


# Motion sequence
# Each entry: (duration_seconds, {joint_index: target_q, ...}).
# A joint not listed holds at wherever it already is!!!
# it does NOT snap back to its original q_start!!!!!
MOTION_SEQUENCE = [
    (3.0, {
        H1_2_JointIndex.LeftElbow: 0.33,
        H1_2_JointIndex.LeftShoulderPitch: -1.15,
    }),
    (2.0, {
        H1_2_JointIndex.LeftWristRoll: 1.6,
    }),
    (2.0, {
        H1_2_JointIndex.LeftWristRoll: -1.4,
    })
]

# output_torque = kp*(q_target - q_actual) + kd*(0 - q_actual_vel) !!!!!! THESE NUMBERS ARE FROM THE OFFICIAL EXAMPLE 
HOLD_KP = 100.0     # legs/waist
HOLD_KP_ARM = 50.0  # all arm joints, moving or holding
HOLD_KD = 1.0

CONTROL_DT = 0.002  # matches official example


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

        self.q_current = None          # joint commanded target, persists across segments
        self.segment_index = 0
        self.segment_start_time = 0.0
        self.segment_start_q = None    # snapshot of q_current when the current segment began
        self.roll_buffer = []

    def Init(self):
        # The simulator has no built-in motion control service to release, and
        # unitree_mujoco serves no rpc at all, so CheckMode() would time out.
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
            self.q_current = list(self.q_start)
            self.segment_start_q = list(self.q_start)
            self.update_mode_machine_ = True

    def LowCmdWrite(self):
        self.time_ += CONTROL_DT

        roll, pitch, yaw = self.low_state.imu_state.rpy
        self.roll_buffer.append(roll)
        print(roll, pitch, yaw)


        if self.segment_index < len(MOTION_SEQUENCE):
            duration, targets = MOTION_SEQUENCE[self.segment_index]
            elapsed = self.time_ - self.segment_start_time
            ratio = min(elapsed / duration, 1.0)

            for joint, target_q in targets.items():
                start_q = self.segment_start_q[joint]
                self.q_current[joint] = start_q + (target_q - start_q) * ratio

            if elapsed >= duration:
                self.segment_start_q = list(self.q_current)
                self.segment_start_time = self.time_
                self.segment_index += 1
        # else: sequence finished, q_current holds the final pose foreever!!!

        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine_

        for i in range(H1_2_NUM_MOTOR):
            self.low_cmd.motor_cmd[i].mode = 1
            self.low_cmd.motor_cmd[i].tau = 0.0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].q = self.q_current[i]
            self.low_cmd.motor_cmd[i].kp = HOLD_KP if i < 13 else HOLD_KP_ARM
            self.low_cmd.motor_cmd[i].kd = HOLD_KD

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)


if __name__ == '__main__':
    print("WARNING: this releases the robots built in balance controller for ALL joints")
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

    if len(custom.roll_buffer) > 500:
        plt.plot(custom.roll_buffer)
        plt.title("Roll Angle")
        plt.xlabel("Time Steps")
        plt.ylabel("Roll Angle (rad)")
        plt.grid()
        plt.show()

    while True:
        time.sleep(1)
