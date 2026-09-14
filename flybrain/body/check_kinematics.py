"""Verify that skeleton.json + the JS kinematics rule reproduce MuJoCo exactly.

If this passes, a pose computed in the browser is the same pose MuJoCo would
compute. If it ever fails, the browser fly is lying about the body.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import mujoco


def quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def quat_rot(q, v):
    w, x, y, z = q
    u = np.array([x, y, z])
    return v + 2*np.cross(u, np.cross(u, v) + w*v)


def axis_angle(axis, angle):
    a = np.asarray(axis, float); a = a/np.linalg.norm(a)
    s = np.sin(angle/2)
    return np.array([np.cos(angle/2), a[0]*s, a[1]*s, a[2]*s])


def forward(skeleton, angles):
    """The exact rule web/kinematics.js implements."""
    by_name = {b["name"]: b for b in skeleton["bodies"]}
    world = {}
    for b in skeleton["bodies"]:
        if b["parent"] is None:
            world[b["name"]] = (np.zeros(3), np.array([1.0, 0, 0, 0]))
            continue
        pp, pq = world[b["parent"]]
        pos = pp + quat_rot(pq, np.array(b["pos"], float))
        quat = quat_mul(pq, np.array(b["quat"], float))
        for j in b["joints"]:                       # joints applied in order
            theta = float(angles.get(j["name"], 0.0))
            if theta == 0.0:
                continue
            anchor = np.array(j["anchor"], float)
            rot = axis_angle(j["axis"], theta)
            # rotate about the anchor, expressed in this body's frame
            pos = pos + quat_rot(quat, anchor - quat_rot(rot, anchor))
            quat = quat_mul(quat, rot)
        world[b["name"]] = (pos, quat)
    return world


def main():
    skeleton = json.loads(Path("web/skeleton.json").read_text())
    xml = sys.argv[1]
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)

    rng = np.random.default_rng(0)
    worst = 0.0
    for trial in range(6):
        angles = {}
        for b in skeleton["bodies"]:
            for j in b["joints"]:
                lo, hi = j["range"]
                if lo == hi == 0.0:
                    lo, hi = -0.4, 0.4
                angles[j["name"]] = float(rng.uniform(lo * 0.8, hi * 0.8))
        mujoco.mj_resetData(m, d)
        for jname, value in angles.items():
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
            d.qpos[m.jnt_qposadr[jid]] = value
        mujoco.mj_kinematics(m, d)

        ours = forward(skeleton, angles)
        for b in skeleton["bodies"]:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b["name"])
            if bid < 0:
                continue
            err = np.linalg.norm(ours[b["name"]][0] - d.xpos[bid])
            worst = max(worst, err)
    scale = float(np.abs(d.xpos).max())
    print(f"worst body-position error across 6 random poses: {worst:.3e} "
          f"(model extent {scale:.3f}) -> {worst/scale*100:.4f}% of body size")
    ok = worst < 1e-9
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
