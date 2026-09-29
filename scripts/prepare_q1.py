"""Prepare the supplied Q1 asset and calibrate its LAFAN1 IK configuration."""

import argparse
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from general_motion_retargeting.utils.lafan1 import load_bvh_file


def prepare(model_dir: Path, reference_bvh: Path):
    source = model_dir / "q1_local_mesh.xml"
    output = model_dir / "q1_gmr.xml"
    tree = ET.parse(source)
    xml = tree.getroot()
    xml.set("model", "q1_gmr_22dof")
    world = xml.find("worldbody")
    for body in list(world.findall("body")):
        if body.get("name") != "pelvis":
            world.remove(body)
    bodies = {body.get("name"): body for body in xml.iter("body")}
    bodies["pelvis"].set("pos", "0 0 0.42")
    for side in ("left", "right"):
        # Forearm mesh extends to x=0.204 m; use the centre of its distal end.
        ET.SubElement(bodies[f"{side}_elbow_link"], "body", {
            "name": f"{side}_hand_mocap", "pos": "0.195 0 0",
        })
    ET.indent(tree, space="  ")
    tree.write(output, encoding="unicode")
    model = mj.MjModel.from_xml_path(str(output))
    if (model.nq, model.nv, model.nu) != (29, 28, 22):
        raise ValueError(f"Expected free root + 22 hinges, got {model.nq, model.nv, model.nu}")

    frames, height = load_bvh_file(str(reference_bvh))
    reference = frames[0]
    # The first frame of walk1_subject1 is the LAFAN reference T pose.
    source_root = R.from_quat(reference["Hips"][1], scalar_first=True)
    baseline = source_root * R.from_quat([0.5, 0.5, 0.5, 0.5], scalar_first=True)
    yaw = np.arctan2(baseline.as_matrix()[1, 0], baseline.as_matrix()[0, 0])
    root_rotation = R.from_euler("z", yaw)
    data = mj.MjData(model)
    for side, sign in (("left", 1), ("right", -1)):
        data.qpos[model.joint(f"{side}_shoulder_roll_joint").qposadr[0]] = sign * np.pi / 2
        data.qpos[model.joint(f"{side}_elbow_joint").qposadr[0]] = np.pi / 2
    data.qpos[3:7] = root_rotation.as_quat(scalar_first=True)
    data.qpos[:2] = reference["Hips"][0][:2] * 0.45
    mj.mj_forward(model, data)
    sole_ids = [i for i in range(model.ngeom) if model.body(int(model.geom_bodyid[i])).name in
                ("left_ankle_roll_link", "right_ankle_roll_link") and model.geom_type[i] == mj.mjtGeom.mjGEOM_SPHERE]
    sole_bottom = min(data.geom_xpos[i, 2] - model.geom_size[i, 0] for i in sole_ids)
    data.qpos[2] -= sole_bottom
    mj.mj_forward(model, data)

    mapping = {
        "pelvis": ("Hips", 0.45, 0, 10, 100, 5),
        "torso_link": ("Spine2", 0.45, 0, 60, 0, 10),
    }
    for side, human_side in (("left", "Left"), ("right", "Right")):
        mapping.update({
            f"{side}_hip_yaw_link": (f"{human_side}UpLeg", 0.45, 0, 10, 10, 5),
            f"{side}_knee_link": (f"{human_side}Leg", 0.45, 0, 10, 10, 5),
            f"{side}_ankle_roll_link": (f"{human_side}FootMod", 0.45, 80, 20, 80, 20),
            f"{side}_shoulder_yaw_link": (f"{human_side}Arm", 0.55, 0, 40, 5, 20),
            f"{side}_elbow_link": (f"{human_side}ForeArm", 0.55, 0, 10, 10, 5),
            f"{side}_hand_mocap": (f"{human_side}Hand", 0.55, 30, 0, 30, 0),
        })
    cfg = {
        "robot_root_name": "pelvis", "human_root_name": "Hips",
        "ground_height": 0.0, "human_height_assumption": height,
        "use_ik_match_table1": True, "use_ik_match_table2": True,
        "human_scale_table": {}, "ik_match_table1": {}, "ik_match_table2": {},
    }
    human_root = reference["Hips"][0]
    scaled_root = human_root * 0.45
    for robot_body, (human_body, scale, pos1, rot1, pos2, rot2) in mapping.items():
        cfg["human_scale_table"][human_body] = scale
        human_pos, human_quat = reference[human_body]
        body_id = model.body(robot_body).id
        robot_rot = R.from_quat(data.xquat[body_id], scalar_first=True)
        human_rot = R.from_quat(human_quat, scalar_first=True)
        rot_offset = (human_rot.inv() * robot_rot).as_quat(scalar_first=True)
        target_pos = scaled_root + (human_pos - human_root) * scale
        pos_offset = robot_rot.inv().apply(data.xpos[body_id] - target_pos)
        for table, pos_cost, rot_cost in (("ik_match_table1", pos1, rot1), ("ik_match_table2", pos2, rot2)):
            cfg[table][robot_body] = [human_body, pos_cost, rot_cost,
                                     pos_offset.round(9).tolist(), rot_offset.round(9).tolist()]
    config_path = Path(__file__).resolve().parents[1] / "general_motion_retargeting/ik_configs/bvh_lafan1_to_q1.json"
    config_path.write_text(json.dumps(cfg, indent=2) + "\n")
    metadata = {
        "source_model": str(source.resolve()), "model": str(output.resolve()),
        "reference_bvh": str(reference_bvh.resolve()), "reference_frame": 0,
        "joint_names": [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] == mj.mjtJoint.mjJNT_HINGE],
        "root_quaternion_order": "xyzw in saved motion; wxyz in MuJoCo",
        "hand_point_local": [0.195, 0, 0], "reference_root_height_m": float(data.qpos[2]),
    }
    (model_dir / "q1_gmr_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Prepared {output}: nq=29, nv=28, nu=22; config={config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-bvh", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.model_dir, args.reference_bvh)
