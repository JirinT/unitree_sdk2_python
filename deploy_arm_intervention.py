"""H1_2 ArmIntervention actor in the separate Unitree MuJoCo simulator.

Defaults to simulation domain 1 / loopback, never real hardware.
Keyboard: python deploy_arm_intervention.py deploy_arm_intervention.yaml policy.onnx
Demo: add --arm-control demo. Offline check: add --validate-only (no DDS imports).
"""

import argparse
from bisect import bisect_right
import math
import threading
import time

import mujoco
import numpy as np

from deploy_sim_teleop import PolicyRunner, TeleopThread, quat_rotate_inverse

ARM_JOINT_NAMES = tuple(f"{side}_{part}_joint" for side in ("left", "right")
    for part in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                 "wrist_roll", "wrist_pitch", "wrist_yaw"))
ACTOR_OBS_ORDER = ["base_ang_vel", "projected_gravity", "command", "phase",
                   "joint_pos", "joint_vel", "actions", "arm_control"]


def smoothstep(t):
    t = np.clip(t, 0., 1.)
    return t**3 * (10. + t*(-15. + 6.*t))


class ArmTargetMux:
    """NumPy equivalent of training's deterministic external-arm action path."""
    def __init__(self, cfg, joint_names, default, scale, offset, model):
        self.indices = np.array([joint_names.index(n) for n in ARM_JOINT_NAMES])
        self.default, self.scale, self.offset = default.copy(), scale.copy(), offset.copy()
        if (scale <= 0).any():
            raise ValueError("Arm intervention requires positive action scales")
        self.max_speed = float(cfg["arm_max_joint_speed"])
        self.max_acceleration = float(cfg["arm_max_joint_acceleration"])
        self.handover_time = float(cfg["arm_handover_time"])
        self.waypoint_duration = float(cfg["arm_waypoint_duration"])
        factor, margin = float(cfg["soft_joint_pos_limit_factor"]), float(cfg["arm_limit_margin"])
        if (not 0 < factor <= 1 or not math.isfinite(margin) or margin < 0
                or any(not math.isfinite(x) or x <= 0 for x in (
                    self.max_speed, self.max_acceleration, self.handover_time, self.waypoint_duration))):
            raise ValueError("Invalid arm trajectory limits")
        ids = [model.joint(n).id for n in ARM_JOINT_NAMES]
        if not np.asarray(model.jnt_limited[ids]).all():
            raise ValueError("Expected bounded H1_2 arm joints")
        limits = model.jnt_range[ids]
        center = limits.mean(axis=1)
        half = (limits[:,1]-limits[:,0])*factor/2
        self.lower, self.upper = center-half+margin, center+half-margin
        if (self.lower >= self.upper).any():
            raise ValueError("Arm margin leaves no workspace")
        self.ranges = np.asarray(cfg["arm_target_ranges"], dtype=float)
        if (cfg.get("arm_joint_names") != list(ARM_JOINT_NAMES) or self.ranges.shape != (14,2)
                or not np.isfinite(self.ranges).all() or (self.ranges[:,0] > self.ranges[:,1]).any()):
            raise ValueError("Expected 14 ordered arm target ranges and canonical arm names")
        self.executed = default.copy()
        self.start = self.goal = default[self.indices].copy()
        self.reference = self.start.copy()
        self.duration, self.elapsed = np.float32(1.), np.float32(0.)
        self.masks = np.zeros(2,dtype=bool)
        self.releasing = np.zeros(2,dtype=bool)
        self.release_start = self.reference.copy()
        self.release_elapsed = np.zeros(2,dtype=np.float32)

    @property
    def control_mask(self):
        return self.masks | self.releasing

    def set_targets(self, targets, masks, duration=None):
        target = np.asarray(targets,dtype=float).reshape(-1)
        masks = np.asarray(masks)
        duration = self.waypoint_duration if duration is None else float(duration)
        if (target.shape != (14,) or masks.shape != (2,) or not np.isfinite(target).all()
                or not np.isin(masks,[0,1]).all() or not math.isfinite(duration) or duration <= 0):
            raise ValueError("Expected finite 14-joint targets, two binary masks, positive duration")
        old = self.control_mask.copy()
        self.reference = self.executed[self.indices].copy()
        self.start = self.reference.copy()
        self.goal = np.clip(target,self.lower,self.upper)
        delta = np.abs(self.goal-self.start).max()
        self.duration = max(duration,1.875*delta/self.max_speed,
                            math.sqrt((10/math.sqrt(3))*delta/self.max_acceleration))
        self.duration = np.float32(self.duration)
        self.elapsed = np.float32(0.)
        self.release_start = self.reference.copy()
        self.release_elapsed[:] = 0.
        self.masks = masks.astype(bool)
        self.releasing = old & ~self.masks

    def observation(self):
        mask = self.control_mask
        return np.concatenate((mask.astype(float),
            (self.executed[self.indices]-self.default[self.indices])*np.repeat(mask,7)))

    def advance(self, raw_action, dt):
        raw_action = np.asarray(raw_action)
        if raw_action.shape != (27,) or not np.isfinite(raw_action).all():
            raise ValueError("ONNX actor returned invalid actions")
        policy = raw_action*self.scale+self.offset
        self.elapsed += np.float32(dt)
        self.reference = self.start+(self.goal-self.start)*smoothstep(self.elapsed/self.duration)
        arms = policy[self.indices]
        target = np.where(np.repeat(self.masks,7),self.reference,arms)
        self.release_elapsed += dt
        blend = np.repeat(smoothstep(self.release_elapsed/self.handover_time),7)
        release = self.release_start+(arms-self.release_start)*blend
        target = np.where(np.repeat(self.releasing,7),release,target)
        previous = self.executed[self.indices]
        bounded = np.clip(previous+np.clip(target-previous,-self.max_speed*dt,self.max_speed*dt),self.lower,self.upper)
        policy[self.indices] = np.where(np.repeat(self.control_mask,7),bounded,arms)
        self.executed = policy.copy()
        close = np.abs(bounded-arms).reshape(2,7).max(axis=1) < 1e-5
        self.releasing &= ~((self.release_elapsed >= self.handover_time) & close)
        return policy


def pose(mux, name):
    target = mux.default[mux.indices].copy().reshape(2,7)
    if name != "neutral":
        pitch,roll,elbow = {"reach":(-.85,.25,.8),"wide":(-.4,.85,1.05),
                            "carry":(-.55,.2,1.4),"place":(-.95,.25,.5)}[name]
        target[:,0],target[:,1],target[:,2],target[:,3] = pitch,[roll,-roll],0.,elbow
        target[:,4:] = 0.
    low = np.maximum(mux.lower,np.minimum(mux.ranges[:,0],mux.default[mux.indices]))
    high = np.minimum(mux.upper,np.maximum(mux.ranges[:,1],mux.default[mux.indices]))
    return np.clip(target.ravel(),low,high).reshape(2,7)


class ArmKeyboardCommands:
    help_lines = (
        "1/2/3 select left/right/both arms; 4/5 toggle takeover; 0 release both to PPO.",
        "6/C neutral; 7 reach; 8 wide; 9 carry. Presets enable selected arms.",
        "[ / ] select joint; + (or =) / - adjust by 0.1 rad. No IK. Ctrl+C exits.",
    )
    def __init__(self, mux, velocity_ranges):
        self.mux = mux
        self.velocity_ranges = np.asarray(velocity_ranges,dtype=float)
        if (self.velocity_ranges.shape != (3,2) or not np.isfinite(self.velocity_ranges).all()
                or (self.velocity_ranges[:,0] > 0).any() or (self.velocity_ranges[:,1] < 0).any()):
            raise ValueError("Velocity ranges must contain zero")
        self.lock = threading.Lock()
        self.velocity = np.zeros(3)
        self.targets = pose(mux,"neutral")
        self.masks = np.zeros(2)
        self.selected, self.joint, self.version = (0,1),0,0

    def press(self,key):
        key = key.upper()
        with self.lock:
            if key in (" ","B"):
                self.velocity[:] = 0.
            elif len(key) == 1 and key in "WSQEZX":
                axis,delta = {"W":(0,.1),"S":(0,-.1),"Q":(1,.1),"E":(1,-.1),"Z":(2,.2),"X":(2,-.2)}[key]
                self.velocity[axis] = np.clip(self.velocity[axis]+delta,*self.velocity_ranges[axis])
            elif key in ("1","2","3"):
                self.selected = {"1":(0,),"2":(1,),"3":(0,1)}[key]
            elif key in ("4","5"):
                side=int(key)-4
                self.masks[side]=1-self.masks[side]
                self.version+=1
            elif key == "0":
                self.masks[:]=0
                self.version+=1
            elif key in ("[","]"):
                self.joint=(self.joint+(1 if key == "]" else -1))%7
                print(f"Selected arm joint: {ARM_JOINT_NAMES[self.joint].removeprefix('left_')}")
            elif key in ("6","7","8","9","C"):
                target=pose(self.mux,{"6":"neutral","C":"neutral","7":"reach","8":"wide","9":"carry"}[key])
                for side in self.selected:
                    self.targets[side]=target[side]
                    self.masks[side]=1
                self.version+=1
            elif key in ("+","=","-"):
                for side in self.selected:
                    i=side*7+self.joint
                    low=max(self.mux.lower[i],min(self.mux.ranges[i,0],self.mux.default[self.mux.indices[i]]))
                    high=min(self.mux.upper[i],max(self.mux.ranges[i,1],self.mux.default[self.mux.indices[i]]))
                    self.targets[side,self.joint]=np.clip(self.targets[side,self.joint]+(-.1 if key == "-" else .1),low,high)
                    self.masks[side]=1
                self.version+=1
            else:
                return False
        return True

    def snapshot_versioned(self):
        with self.lock:
            return self.velocity.copy(),self.targets.copy(),self.masks.copy(),self.version

    def snapshot(self):
        return self.snapshot_versioned()[:3]


class ArmDemo:
    PHASES = ((0.,None,(0.,0.,0.)),(2.,"reach",(0.,0.,0.)),(6.,"carry",(0.,0.,0.)),
              (10.,"carry",(.3,0.,0.)),(18.,"carry",(0.,0.,0.)),(22.,"place",(0.,0.,0.)),
              (26.,"neutral",(0.,0.,0.)),(29.,None,(0.,0.,0.)))
    def __init__(self):
        self.last_phase=self.last_pose=None

    def update(self,controls,policy_time):
        phase=bisect_right([p[0] for p in self.PHASES],policy_time%32.+1e-8)-1
        _,name,velocity=self.PHASES[phase]
        changed=phase!=self.last_phase
        with controls.lock:
            controls.velocity[:]=velocity
            if changed and (name!=self.last_pose or self.last_phase is None):
                controls.masks[:]=float(name is not None)
                if name is not None: controls.targets[:]=pose(controls.mux,name)
                controls.version+=1
        if changed: print(f"[ARM DEMO {policy_time%32.:04.1f}s] {name or 'PPO arms'}; vx={velocity[0]:.2f}")
        self.last_phase,self.last_pose=phase,name


class ArmInterventionRunner(PolicyRunner):
    OBS_DIM = 108
    ACTOR_OBS_ORDER = ACTOR_OBS_ORDER

    def _make_controls(self,cfg):
        self.mux=ArmTargetMux(cfg,self.joint_names,self.default_q,self.action_scale,self.action_offset,self.fk.model)
        return ArmKeyboardCommands(self.mux,cfg["velocity_ranges"])

    def __init__(self,deploy_yaml,onnx_path,simulation=True,robot_xml=None,arm_control="keyboard"):
        if not simulation:
            raise ValueError("This arm-intervention adapter supports Unitree MuJoCo simulation only")
        if arm_control not in {"keyboard","demo","policy"}:
            raise ValueError("Unknown arm control mode")
        super().__init__(deploy_yaml,onnx_path,simulation=True,robot_xml=robot_xml)
        metadata=self.session.get_modelmeta().custom_metadata_map
        required={"arm_control_contract":"intervention_v1",
                  "previous_action_semantics":"executed_target_minus_offset_div_scale",
                  "arm_control_observation":"mask_left,mask_right,executed_arm_target_minus_default[14]"}
        for key,value in required.items():
            if metadata.get(key)!=value: raise ValueError(f"Missing/incompatible ONNX {key}; re-export the ArmIntervention policy")
        exported_indices=np.fromstring(metadata.get("arm_action_indices",""),sep=",")
        if (metadata.get("arm_joint_names","").split(",")!=list(ARM_JOINT_NAMES)
                or not np.array_equal(exported_indices,self.mux.indices)):
            raise ValueError("ONNX arm joint names/action indices do not match")
        for key,value in (("arm_max_joint_speed",self.mux.max_speed),
                          ("arm_max_joint_acceleration",self.mux.max_acceleration),
                          ("arm_handover_time",self.mux.handover_time)):
            if key not in metadata or not math.isclose(float(metadata[key]),value,abs_tol=1e-6):
                raise ValueError(f"ONNX {key} differs from deployment YAML")
        self.arm_control=arm_control
        self.demo=ArmDemo() if arm_control=="demo" else None
        self._last_version=-1
        self.raw_action=np.zeros(27,dtype=np.float32)

    def _sync_controls(self):
        if self.demo is not None: self.demo.update(self.controls,self.policy_time)
        command,targets,masks,version=self.controls.snapshot_versioned()
        if self.arm_control=="policy": masks=np.zeros(2)
        if version!=self._last_version:
            self.mux.set_targets(targets,masks,duration=2.5 if self.demo else None)
            self._last_version=version
        self.command=command

    def build_obs(self):
        self._sync_controls()
        state=self.low_state
        if state is None: raise RuntimeError("No LowState available")
        q=np.array([state.motor_state[i].q for i in self.joint_map])
        dq=np.array([state.motor_state[i].dq for i in self.joint_map])
        gyro=np.asarray(state.imu_state.gyroscope,dtype=float)
        if dq.shape!=(27,) or gyro.shape!=(3,) or not np.isfinite(dq).all() or not np.isfinite(gyro).all():
            raise ValueError("Nonfinite joint velocity or IMU sample")
        _,rotation=self.fk.compute(q)
        gravity=quat_rotate_inverse(np.asarray(state.imu_state.quaternion),np.array([0.,0.,-1.]))
        if self.imu_quaternion_frame=="imu": gravity=rotation@gravity
        phase=np.zeros(2)
        if np.linalg.norm(self.command)>=.1:
            angle=2*math.pi*(self.policy_time%self.gait_period)/self.gait_period
            phase=np.array([math.sin(angle),math.cos(angle)])
        obs=np.concatenate((gyro*self.obs_scales["ang_vel"],gravity,self.command*self.command_scale,phase,
            (q-self.default_q)*self.obs_scales["dof_pos"],dq*self.obs_scales["dof_vel"],
            self.last_action,self.mux.observation())).astype(np.float32)
        if obs.shape!=(108,) or not np.isfinite(obs).all(): raise ValueError("Invalid actor observation")
        return obs

    def infer(self):
        obs=self.build_obs()
        result=np.asarray(self.session.run(None,{self.in_name:obs[None]})[0])
        if result.shape!=(1,27) or not np.isfinite(result).all(): raise ValueError("ONNX actor returned invalid actions")
        self.raw_action=result[0].astype(np.float32,copy=True)
        self.target_q=self.mux.advance(self.raw_action,self.step_dt)
        self.last_action=((self.target_q-self.action_offset)/self.action_scale).astype(np.float32)
        self.policy_time+=self.step_dt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deploy_yaml")
    parser.add_argument("onnx_path")
    parser.add_argument("target",nargs="?",default="simulation",choices=["simulation"])
    parser.add_argument("--arm-control",choices=["keyboard","demo","policy"],default="keyboard")
    parser.add_argument("--robot-xml")
    parser.add_argument("--duration",type=float,help="Seconds after startup ramp; omit to run until Ctrl+C")
    parser.add_argument("--validate-only",action="store_true")
    args=parser.parse_args()
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration<=0): parser.error("duration must be positive")
    runner=ArmInterventionRunner(args.deploy_yaml,args.onnx_path,robot_xml=args.robot_xml,arm_control=args.arm_control)
    if args.validate_only:
        runner.validate()
        return
    if args.arm_control=="keyboard": TeleopThread.check_terminal()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(1,"lo")
    teleop=None
    try:
        runner.Init()
        runner.Start()
        if args.arm_control=="keyboard":
            teleop=TeleopThread(runner)
            teleop.start()
        while not runner.stop_event.wait(.1):
            if args.duration is not None and runner.policy_time>=args.duration: break
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop_event.set()
        if teleop is not None and teleop.is_alive(): teleop.join(timeout=1.)
        runner.Stop()
    if runner.error is not None: raise RuntimeError("Arm intervention stopped") from runner.error


if __name__=="__main__": main()
