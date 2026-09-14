"""Turn flybody's 151 MB of meshes into a ~1 MB browser asset.

Produces two files:
  web/fly.glb        decimated geometry, one mesh per MuJoCo body
  web/skeleton.json  the body tree, rest transforms, joint axes and anchors

The JS side rebuilds MuJoCo's forward kinematics from skeleton.json: each body
is a node at (body_pos, body_quat) containing a pivot that rotates by the joint
angle about (jnt_pos, jnt_axis). That is exactly what mj_kinematics does for
hinge joints, so a pose computed in Python and a pose computed in JS agree.
"""
from __future__ import annotations
import argparse, base64, json, struct
from pathlib import Path
import numpy as np
import mujoco

DRIVEN = ("coxa_abduct", "coxa", "femur", "tibia")


def decimate(V, F, keep):
    try:
        import fast_simplification as fs
    except ImportError:
        return V, F
    target = max(700, int(len(F) * keep))
    if len(F) <= target:
        return V, F
    V2, F2 = fs.simplify(V.astype(np.float32), F.astype(np.int32),
                         target_reduction=1.0 - target / len(F))
    return np.asarray(V2, np.float32), np.asarray(F2, np.uint32)


def build(xml_path: str, out_dir: Path, keep: float = 0.20):
    m = mujoco.MjModel.from_xml_path(xml_path)
    name = lambda t, i: mujoco.mj_id2name(m, t, i)

    # ---- geometry: merge each body's mesh geoms into one buffer -----------
    chunks, meshes = [], {}
    for b in range(m.nbody):
        bname = name(mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
        verts, faces, base = [], [], 0
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mi = m.geom_dataid[g]
            vn, fn = m.mesh_vertnum[mi], m.mesh_facenum[mi]
            V = m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi] + vn].reshape(-1, 3)
            F = m.mesh_face[m.mesh_faceadr[mi]:m.mesh_faceadr[mi] + fn].reshape(-1, 3)
            V, F = decimate(np.asarray(V, np.float32), np.asarray(F, np.int32), keep)
            # geom pose within the body
            q = m.geom_quat[g]
            R = np.zeros(9); mujoco.mju_quat2Mat(R, q); R = R.reshape(3, 3)
            V = V @ R.T + m.geom_pos[g]
            verts.append(V.astype(np.float32)); faces.append(F.astype(np.uint32) + base)
            base += len(V)
        if verts:
            meshes[bname] = (np.concatenate(verts), np.concatenate(faces))

    # ---- skeleton --------------------------------------------------------
    # a body can carry several hinge joints (coxa_abduct + coxa_twist + coxa all
    # sit on the same coxa body). MuJoCo applies them in declaration order, so
    # keep the full ordered list rather than one per body.
    from .gait import LEGS
    driven_names = {f"{d}_{leg}" for d in DRIVEN for leg in LEGS}
    joints_of = {}
    for j in range(m.njnt):
        jname = name(mujoco.mjtObj.mjOBJ_JOINT, j)
        if jname and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            joints_of.setdefault(int(m.jnt_bodyid[j]), []).append(dict(
                name=jname,
                axis=[float(x) for x in m.jnt_axis[j]],
                anchor=[float(x) for x in m.jnt_pos[j]],
                range=[float(x) for x in m.jnt_range[j]],
                driven=jname in driven_names,
            ))

    bodies = []
    for b in range(m.nbody):
        bname = name(mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
        bodies.append(dict(
            name=bname,
            parent=name(mujoco.mjtObj.mjOBJ_BODY, m.body_parentid[b]) if b else None,
            pos=[float(x) for x in m.body_pos[b]],
            quat=[float(x) for x in m.body_quat[b]],     # w,x,y,z (MuJoCo order)
            joints=joints_of.get(b, []),
            mesh=bname in meshes,
        ))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "skeleton.json").write_text(json.dumps(
        dict(bodies=bodies, driven=list(DRIVEN),
             note="quaternions are MuJoCo order (w,x,y,z); convert for three.js"),
        separators=(",", ":")))

    write_mesh_json(out_dir / "fly.json", meshes)
    tri = sum(len(f) for _, f in meshes.values())
    size = (out_dir / "fly.json").stat().st_size
    print(f"{len(meshes)} meshes, {tri:,} triangles -> {size/1e6:.2f} MB  ({out_dir/'fly.json'})")
    driven = sum(1 for x in bodies for j in x["joints"] if j["driven"])
    print(f"skeleton: {len(bodies)} bodies, "
          f"{sum(len(x['joints']) for x in bodies)} hinge joints, {driven} driven")


def write_mesh_json(path: Path, meshes: dict):
    """One entry per body: base64 float32 positions and uint32 indices.

    Plain JSON rather than glTF because the hierarchy already lives in
    skeleton.json -- all the browser needs is raw geometry per body, and a
    servable text format travels anywhere.
    """
    out = {"format": "flybrain-mesh-1", "bodies": {}}
    for name, (V, F) in meshes.items():
        out["bodies"][name] = {
            "positions": base64.b64encode(V.astype(np.float32).tobytes()).decode(),
            "indices": base64.b64encode(F.astype(np.uint32).tobytes()).decode(),
            "vertices": int(len(V)), "triangles": int(len(F)),
        }
    path.write_text(json.dumps(out, separators=(",", ":")))


def _unused_write_glb(path: Path, meshes: dict):
    """Kept for reference: the same data as a minimal glTF 2.0 binary."""
    buf, views, accessors, prims, nodes = bytearray(), [], [], [], []
    for bname, (V, F) in meshes.items():
        F = F.astype(np.uint32)
        while len(buf) % 4: buf.append(0)
        v_off = len(buf); buf += V.astype(np.float32).tobytes()
        while len(buf) % 4: buf.append(0)
        f_off = len(buf); buf += F.tobytes()

        views.append(dict(buffer=0, byteOffset=v_off, byteLength=V.nbytes, target=34962))
        views.append(dict(buffer=0, byteOffset=f_off, byteLength=F.nbytes, target=34963))
        vi, fi = len(views) - 2, len(views) - 1
        accessors.append(dict(bufferView=vi, componentType=5126, count=len(V), type="VEC3",
                              min=V.min(axis=0).tolist(), max=V.max(axis=0).tolist()))
        accessors.append(dict(bufferView=fi, componentType=5125, count=F.size, type="SCALAR"))
        prims.append(dict(name=bname, primitives=[dict(
            attributes=dict(POSITION=len(accessors) - 2), indices=len(accessors) - 1)]))
        nodes.append(dict(name=bname, mesh=len(prims) - 1))

    gltf = dict(asset=dict(version="2.0", generator="flybrain-starter"),
                scene=0, scenes=[dict(nodes=list(range(len(nodes))))],
                nodes=nodes, meshes=prims, accessors=accessors,
                bufferViews=views, buffers=[dict(byteLength=len(buf))])
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    while len(buf) % 4: buf.append(0)
    glb = (struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(buf))
           + struct.pack("<II", len(js), 0x4E4F534A) + js
           + struct.pack("<II", len(buf), 0x004E4942) + bytes(buf))
    path.write_bytes(glb)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("xml", help="path to flybody's fruitfly.xml")
    p.add_argument("--out", default="web")
    p.add_argument("--keep", type=float, default=0.20)
    a = p.parse_args()
    build(a.xml, Path(a.out), a.keep)
