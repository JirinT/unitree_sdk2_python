import time
import matplotlib.pyplot as plt
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


start_time = time.time()
timestamps = []
roll_history = []


def on_low_state(msg: LowState_):
    m = msg.motor_state[H1_2_JointIndex.LeftWristRoll]
    # print(f"q={m.q:.3f} rad  dq={m.dq:.3f} rad/s  tau_est={m.tau_est:.3f} Nm")

    roll, pitch, yaw = msg.imu_state.rpy
    timestamps.append(time.time() - start_time)
    roll_history.append(roll)


ChannelFactoryInitialize(0, "enx00e04c68005c")
sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(on_low_state, 10)

print("Recording IMU roll. Press Ctrl+C to stop and plot.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print(f"\nStopped. Captured {len(roll_history)} samples over {timestamps[-1]:.1f}s.")

    plt.figure(figsize=(10, 4))
    plt.plot(timestamps, roll_history)
    plt.xlabel("Time (s)")
    plt.ylabel("Roll (rad)")
    plt.title("Torso IMU Roll")
    plt.grid(True)
    plt.savefig("roll_plot.png")
    print("Saved plot to roll_plot.png")
    print(f"Initial Pose: Roll = {roll_history[0]:.3f} rad")
    print(f"Maximal Roll: {max(roll_history):.3f} rad")
    print(f"Minimal Roll: {min(roll_history):.3f} rad")
    plt.show()