"""Inverse kinematics for the fly's legs, measured from the model itself.

Rather than guessing which joint swings a leg forward, we measure it: perturb
each joint, see where the foot goes, and build a damped least-squares inverse.
The resulting 6x3 matrix per leg turns a desired foot displacement into joint
angles. Those matrices are exported to web/legmap.json so the browser applies
exactly the same ones.

Valid for small displacements around the model's natural stance, which is all a
step needs -- the foot travels about a fifth of a body length.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from .check_kinematics import forward
from .gait import LEGS, LEG_JOINTS, foot_offset

EPS = 0.02          # perturbation used to measure the Jacobian, radians
DAMPING = 0.06      # Levenberg-Marquardt damping; keeps angles small and stable


def jacobians(skeleton: dict):
    """For each leg: d(foot position) / d(joint angle), a 3x6 matrix."""
    base = forward(skeleton, {})
    out = {}
    for leg in LEGS:
        foot = f"claw_{leg}"
        J = np.zeros((3, len(LEG_JOINTS)))
        for k, joint in enumerate(LEG_JOINTS):
            moved = forward(skeleton, {f"{joint}_{leg}": EPS})[foot][0]
            J[:, k] = (moved - base[foot][0]) / EPS
        out[leg] = J
    return out


def inverses(skeleton: dict):
    """Damped pseudo-inverse: joint angles = M @ desired foot displacement."""
    out = {}
    for leg, J in jacobians(skeleton).items():
        JT = J.T
        out[leg] = JT @ np.linalg.inv(J @ JT + DAMPING**2 * np.eye(3))
    return out


def pose_from(skeleton_inverses: dict, phase: float, speed: float, turn: float) -> dict:
    angles = {}
    for leg in LEGS:
        delta = np.array(foot_offset(phase, leg, speed, turn))
        theta = skeleton_inverses[leg] @ delta
        for k, joint in enumerate(LEG_JOINTS):
            angles[f"{joint}_{leg}"] = float(theta[k])
    return angles


def export(skeleton_path="web/skeleton.json", out="web/legmap.json"):
    skeleton = json.loads(Path(skeleton_path).read_text())
    inv = inverses(skeleton)
    Path(out).write_text(json.dumps(
        {"joints": LEG_JOINTS, "legs": {k: v.tolist() for k, v in inv.items()}},
        separators=(",", ":")))
    return inv


if __name__ == "__main__":
    inv = export()
    skeleton = json.loads(Path("web/skeleton.json").read_text())
    base = forward(skeleton, {})
    # how faithfully does the linear IK actually place the foot?
    worst = 0.0
    for ph in np.linspace(0, 2 * np.pi, 32, endpoint=False):
        angles = pose_from(inv, float(ph), 1.0, 0.0)
        world = forward(skeleton, angles)
        for leg in LEGS:
            want = np.array(foot_offset(float(ph), leg, 1.0, 0.0))
            got = world[f"claw_{leg}"][0] - base[f"claw_{leg}"][0]
            worst = max(worst, float(np.linalg.norm(got - want)))
    print(f"worst foot placement error over a gait cycle: {worst:.4f} "
          f"({worst/0.2619*100:.1f}% of body length)")
    zs = []
    for ph in np.linspace(0, 2 * np.pi, 32, endpoint=False):
        w = forward(skeleton, pose_from(inv, float(ph), 1.0, 0.0))
        zs.append(min(p[2] for p, _ in w.values()))
    print(f"ground clearance varies {min(zs):.4f} .. {max(zs):.4f}")
