import time
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

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


class ReadSensors:
    def __init__(self):
        self.print_counter = 0

    def on_low_state(self, msg: LowState_):
        message_shoulder = msg.motor_state[H1_2_JointIndex.LeftShoulderPitch]
        message_elbow = msg.motor_state[H1_2_JointIndex.LeftElbow]
        roll, pitch, yaw = msg.imu_state.rpy
        
        self.print_counter += 1
        if self.print_counter % 500 == 0:
            print("---------------------------------")
            print(f"Shoulder: q={message_shoulder.q:.3f} rad, dq={message_shoulder.dq:.3f} rad/s, tau_est={message_shoulder.tau_est:.3f} Nm")
            print(f"Elbow: q={message_elbow.q:.3f} rad, dq={message_elbow.dq:.3f} rad/s, tau_est={message_elbow.tau_est:.3f} Nm")
            print(f"IMU: roll={roll:.3f} rad, pitch={pitch:.3f} rad, yaw={yaw:.3f} rad")

    def start(self):
        ChannelFactoryInitialize(0, "enx00e04c68005c")
        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(self.on_low_state, 10)


if __name__ == '__main__':
    sensor_reader = ReadSensors()
    sensor_reader.start()
    while True:
        time.sleep(1)

