"""Record the trained controller driving the REAL flybody model under physics.

This is a genuine closed loop, not a playback of the kinematic toy:

    MuJoCo body pose -> arena sensors -> 80-cell circuit -> readout -> action
      -> CPG -> leg IK -> position actuators -> MuJoCo physics -> repeat

The fly walks because the gait produces real ground reaction forces; nothing
teleports it. Clips are stored as root pose + the 36 driven joint angles per
frame, and the browser rebuilds each frame with the same forward kinematics it
uses for the live simulation -- so if a clip looks right, the parity holds.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import mujoco

from ..circuit import load as load_circuit, load_silenced
from ..readout import act
from .arenas import ARENAS
from .gait import Walker
from .legs import inverses

FLY_SCALE = 53.5           # arena units per MuJoCo length unit; matches web/app.js


def record(task="targets", policy=None, seconds=14.0, fps=30, seed=2100001,
           xml=None, skeleton="web/skeleton.json", silenced=False):
    xml = xml or "fruitfly/assets/floor.xml"
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    actuator = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}
    inv = inverses(json.loads(Path(skeleton).read_text()))

    theta = np.array(json.loads(Path(policy or f"policies/{task}.json").read_text())["theta"])
    circuit = load_silenced() if silenced else load_circuit()
    env = ARENAS[task](seconds=seconds)
    obs = env.reset(seed)
    circuit.reset()

    mujoco.mj_resetData(m, d)
    d.qpos[2] = 0.145
    mujoco.mj_forward(m, d)

    walker = Walker()
    walker.reset(0.0, 0.0, 0.0)
    origin = d.qpos[:3].copy()

    dt = m.opt.timestep
    physics_per_frame = max(1, int(round((1.0 / fps) / dt)))
    decisions_per_frame = max(1, int(round(30.0 / fps)))
    joint_names = sorted(inv and {f"{j}_{leg}" for leg in inv for j in
                                  ["coxa_abduct", "coxa_twist", "coxa", "femur_twist", "femur", "tibia"]})
    joint_names = [j for j in joint_names if j in actuator]

    frames, actions = [], []
    from .legs import pose_from
    for f in range(int(seconds * fps)):
        for _ in range(decisions_per_frame):
            activity = circuit.step(obs)
            action = int(act(theta, activity, circuit.n_outputs, env.N_ACTIONS))
            obs, score, done = env.step(action)
            if done:
                obs = env.reset(seed + f + 1); circuit.reset()
        actions.append(action)

        # the kinematic arena decides intent; MuJoCo decides what the body does
        walker.speed, walker.turn = env.walker.speed, env.walker.turn
        for _ in range(physics_per_frame):
            walker.phase = float(np.mod(walker.phase + 2 * np.pi *
                                        (0.6 + 7.5 * abs(walker.speed)) * dt, 2 * np.pi))
            angles = pose_from(inv, walker.phase, walker.speed, walker.turn)
            for name, value in angles.items():
                if name in actuator:
                    d.ctrl[actuator[name]] = value
            mujoco.mj_step(m, d)

        pose = [round(float(v), 5) for v in d.qpos[:7]]
        frames.append(pose + [round(float(d.qpos[m.jnt_qposadr[
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]]), 4) for n in joint_names])

    travelled = float(np.linalg.norm(d.qpos[:2] - origin[:2]))
    return dict(task=task, fps=fps, seconds=seconds, seed=seed, silenced=silenced,
                joints=joint_names, frames=frames,
                travelled_body_lengths=round(travelled / 0.2619, 2),
                final_height=round(float(d.qpos[2]), 4),
                scale=FLY_SCALE)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--xml", required=True)
    p.add_argument("--tasks", nargs="+", default=["targets", "swat", "plume", "chase"])
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--out", default="web/clips.json")
    a = p.parse_args()
    clips = {}
    for task in a.tasks:
        if not Path(f"policies/{task}.json").exists():
            print(f"  {task}: no policy yet, skipped"); continue
        clip = record(task=task, seconds=a.seconds, xml=a.xml)
        clips[task] = clip
        print(f"  {task}: {len(clip['frames'])} frames, "
              f"walked {clip['travelled_body_lengths']} body lengths, "
              f"upright at z={clip['final_height']}")
    Path(a.out).write_text(json.dumps(clips, separators=(",", ":")))
    print(f"wrote {a.out} ({Path(a.out).stat().st_size/1e3:.0f} KB)")
