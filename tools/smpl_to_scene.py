"""Convert ProtoMotions' Isaac-Gym SMPL humanoid MJCF into a MuJoCo scene for get-up training.

Changes: free root joint, joint springs -> position actuators (kp = stiffness, kv = damping),
human-ish joint ranges (the source has +-180 deg everywhere), floor + light, standing keyframe.

python tools/smpl_to_scene.py  ->  assets/smpl_scene.xml
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "assets", "smpl_humanoid_raw.xml")
DST = os.path.join(ROOT, "assets", "smpl_scene.xml")

D = math.radians
# (lo, hi) in radians per joint; frame: x forward, y left, z up
RANGES = {
    "Hip_x": (-0.8, 0.8), "Hip_y": (-2.2, 0.6), "Hip_z": (-0.8, 0.8),
    "Knee_x": (-0.1, 0.1), "Knee_y": (-0.05, 2.6), "Knee_z": (-0.1, 0.1),
    "Ankle_x": (-0.5, 0.5), "Ankle_y": (-0.8, 0.8), "Ankle_z": (-0.5, 0.5),
    "Toe_x": (-0.3, 0.3), "Toe_y": (-0.6, 0.6), "Toe_z": (-0.3, 0.3),
    "Torso": (-0.6, 0.6), "Spine": (-0.6, 0.6), "Chest": (-0.5, 0.5),
    "Neck": (-0.6, 0.6), "Head": (-0.5, 0.5),
    "Thorax": (-0.4, 0.4),
    "Shoulder": (-2.0, 2.0),
    "Elbow_x": (-0.3, 0.3), "Elbow_y": (-0.6, 0.6),
    "Wrist": (-0.8, 0.8), "Hand": (-0.6, 0.6),
}  # fmt: skip
ELBOW_Z = {"L": (-2.6, 0.05), "R": (-0.05, 2.6)}  # left arm points +y, flexing forward is -z rotation
# standing pose: arms down (shoulder x), everything else zero
STAND = {"L_Shoulder_x": -1.3, "R_Shoulder_x": 1.3}


def joint_range(name: str):
    head, axis = name.rsplit("_", 1)
    side, part = head.split("_", 1) if head[:2] in ("L_", "R_") else ("", head)
    if part == "Elbow" and axis == "z":
        return ELBOW_Z[side]
    for key in (f"{part}_{axis}", part):
        if key in RANGES:
            return RANGES[key]
    raise KeyError(name)


def main():
    tree = ET.parse(SRC)
    root = tree.getroot()
    for c in root.findall("compiler"):
        root.remove(c)
    root.insert(0, ET.Element("compiler", angle="radian"))
    root.insert(1, ET.Element("option", timestep="0.005", integrator="implicitfast", iterations="10", ls_iterations="10"))
    for tag in ("size", "actuator"):
        for e in root.findall(tag):
            root.remove(e)
    root.append(ET.Element("size", memory="20M"))

    asset = root.find("asset")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker", rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3",
                  width="300", height="300", mark="edge", markrgb="0.8 0.8 0.8")  # fmt: skip
    ET.SubElement(asset, "material", name="grid", texture="grid", texrepeat="5 5", texuniform="true", reflectance="0.2")

    wb = root.find("worldbody")
    wb.insert(0, ET.Element("geom", name="floor", type="plane", size="0 0 0.05", material="grid", condim="3", contype="1", conaffinity="1"))
    pelvis = wb.find("body")
    pelvis.set("pos", "0 0 1")

    act = ET.SubElement(root, "actuator")
    names = []
    for j in root.iter("joint"):
        n = j.get("name")
        if j.get("type") == "free":
            continue
        lo, hi = joint_range(n)
        kp, kd = j.attrib.pop("stiffness"), j.attrib.pop("damping")
        j.set("range", f"{lo} {hi}")
        j.set("actuatorfrcrange", "-500 500")
        ET.SubElement(act, "position", name=n, joint=n, kp=kp, kv=kd, ctrlrange=f"{lo} {hi}")
        names.append(n)

    # standing keyframe: place the lowest geom on the floor
    tmp = DST + ".tmp.xml"
    tree.write(tmp)
    m = mujoco.MjModel.from_xml_path(tmp)
    d = mujoco.MjData(m)
    q = np.zeros(len(names))
    for k, v in STAND.items():
        q[names.index(k)] = v
    d.qpos[:3], d.qpos[3:7], d.qpos[7:] = (0, 0, 1), (1, 0, 0, 0), q
    mujoco.mj_forward(m, d)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    zmin = min(d.geom_xpos[g, 2] - m.geom_rbound[g] for g in range(m.ngeom) if g != floor and m.body_parentid[m.geom_bodyid[g]] >= 0
               and m.geom_bodyid[g] in [m.body(f"{s}_{p}").id for s in "LR" for p in ("Ankle", "Toe")])  # fmt: skip
    # rbound is conservative for boxes; refine with the actual lowest contact after settling below
    h = 1.0 - zmin
    os.remove(tmp)

    kf = ET.SubElement(root, "keyframe")
    qpos = " ".join(f"{x:.4f}" for x in [0, 0, h, 1, 0, 0, 0, *q])
    ET.SubElement(kf, "key", name="stand", qpos=qpos, ctrl=" ".join(f"{x:.4f}" for x in q))
    ET.indent(tree)
    tree.write(DST)

    # settle the standing pose for 1 s and report the resting pelvis height
    m = mujoco.MjModel.from_xml_path(DST)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for _ in range(200):
        mujoco.mj_step(m, d)
    print(f"wrote {DST}: nq {m.nq} nu {m.nu} mass {m.body_subtreemass[1]:.1f} kg | key height {h:.3f} -> settled {d.qpos[2]:.3f}, up {1 - 2 * (d.qpos[4]**2 + d.qpos[5]**2):.3f}")


if __name__ == "__main__":
    main()
