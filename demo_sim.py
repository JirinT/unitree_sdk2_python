"""Expressive simulated grasp/carry/place gesture; ~27 seconds with startup.

Start the H1_2 Unitree MuJoCo validator, then run:
  python demo_sim.py deploy_hand_tracking.yaml /path/to/policy.onnx
"""

from evaluate_sim import main


if __name__ == "__main__":
    main(default_profile="demo")
