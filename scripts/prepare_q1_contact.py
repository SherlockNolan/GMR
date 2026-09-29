"""Measure the supplied Q1 mechanism and create a separate contact IK asset."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np


def prepare(folder):
    source = folder / "q1_local_mesh.xml"
    tree = ET.parse(source)
    xml = tree.getroot()
    xml.set("model", "q1_contact_v2")
    world = xml.find("worldbody")
    for body in list(world.findall("body")):
        if body.get("name") != "pelvis":
            world.remove(body)
    bodies = {b.get("name"): b for b in xml.iter("body")}
    bodies["pelvis"].set("pos", "0 0 0.42")
    # Compile in the original directory so relative mesh paths stay valid.
    model = mj.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"),
        assets={"meshes/" + p.name: p.read_bytes() for p in (folder / "meshes").glob("*.STL")})
    hands = {}
    for side in ("left", "right"):
        body = model.body(f"{side}_elbow_link").id
        geom = next(i for i in range(model.ngeom) if model.geom_bodyid[i] == body
                    and model.geom_type[i] == mj.mjtGeom.mjGEOM_MESH)
        mid = model.geom_dataid[geom]
        vertices = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]]
        rotation = np.empty(9)
        mj.mju_quat2Mat(rotation, model.geom_quat[geom])
        vertices = vertices @ rotation.reshape(3, 3).T + model.geom_pos[geom]
        distal = vertices[vertices[:, 0] >= vertices[:, 0].max() - .002]
        tip = np.array([vertices[:, 0].max(), np.median(distal[:, 1]), np.median(distal[:, 2])])
        hands[side] = {"local_point_m": tip.tolist(), "mesh_bounds_m": [vertices.min(0).tolist(), vertices.max(0).tolist()],
                       "definition": "distal mesh x endpoint; median y/z of final 2 mm; a proxy, not a wrist joint"}
        ET.SubElement(bodies[f"{side}_elbow_link"], "body", {
            "name": f"{side}_hand_mocap", "pos": " ".join(map(str, tip))})
        foot = bodies[f"{side}_ankle_roll_link"]
        for index, geom_xml in enumerate(g for g in foot.findall("geom") if g.get("type") == "sphere"):
            # Sites are at sphere CENTRES. Ground contact is centre.z == radius.
            ET.SubElement(foot, "site", {"name": f"{side}_contact_{index}", "pos": geom_xml.get("pos"),
                "size": ".003", "rgba": "0 0 0 0"})
    output = folder / "q1_contact_v2.xml"
    ET.indent(tree, space="  ")
    tree.write(output, encoding="unicode")
    model = mj.MjModel.from_xml_path(str(output))
    data = mj.MjData(model)
    for side, sign in (("left", 1), ("right", -1)):
        data.qpos[model.joint(f"{side}_shoulder_roll_joint").qposadr[0]] = sign*np.pi/2
        data.qpos[model.joint(f"{side}_elbow_joint").qposadr[0]] = np.pi/2
    mj.mj_forward(model, data)
    feet = [i for i in range(model.ngeom) if "ankle_roll" in model.body(int(model.geom_bodyid[i])).name
            and model.geom_type[i] == mj.mjtGeom.mjGEOM_SPHERE]
    data.qpos[2] -= min(data.geom_xpos[i, 2]-model.geom_size[i, 0] for i in feet)
    mj.mj_forward(model, data)
    positions = {model.body(i).name: (data.xpos[i]-data.qpos[:3]).tolist() for i in range(1, model.nbody)}
    edges = []
    for i in range(2, model.nbody):
        parent = int(model.body_parentid[i])
        edges.append({"parent": model.body(parent).name, "child": model.body(i).name,
                      "local_offset_m": model.body_pos[i].tolist(),
                      "distance_m": float(np.linalg.norm(data.xpos[i]-data.xpos[parent]))})
    def distance(a, b):
        return float(np.linalg.norm(np.array(positions[a])-positions[b]))
    lengths = {"pelvis_to_torso": distance("pelvis", "torso_link"),
               "torso_to_head": distance("torso_link", "head_link")}
    for side in ("left", "right"):
        for key, a, b in (("thigh", "hip_pitch_link", "knee_link"), ("shank", "knee_link", "ankle_pitch_link"),
                          ("upperarm", "shoulder_roll_link", "elbow_link"), ("forearm", "elbow_link", "hand_mocap")):
            lengths[f"{side}_{key}"] = distance(f"{side}_{a}", f"{side}_{b}")
    limits = {joint.get("name"): {k: float(joint.find("limit").get(k)) for k in ("velocity", "effort", "lower", "upper")}
              for joint in ET.parse(folder/"q1_22dof_box.urdf").getroot().findall("joint") if joint.find("limit") is not None}
    metadata = {"source_model": str(source), "model": str(output), "nq": model.nq, "nv": model.nv, "nu": model.nu,
                "reference_pose": "zero joints except shoulder_roll +/- pi/2 and elbows pi/2 (T pose)",
                "reference_qpos": data.qpos.tolist(), "reference_root_height_m": float(data.qpos[2]),
                "body_positions_relative_pelvis_m": positions, "mechanism_edges": edges, "lengths_m": lengths,
                "hand_proxies": hands, "urdf_joint_limits": limits,
                "foot_spheres": [{"side": "left" if "left" in model.body(int(model.geom_bodyid[i])).name else "right",
                                  "local_center_m": model.geom_pos[i].tolist(), "radius_m": float(model.geom_size[i, 0])} for i in feet],
                "caveats": ["Hip and shoulder axes are not coincident. Effective segment lengths are measured in the reference pose; IK uses the full mechanism.",
                            "No wrist joint exists. Human Hand maps to the measured distal forearm mesh proxy.",
                            "The supplied XML and URDF are preserved; decorative free-floating ball is excluded."]}
    report = folder/"q1_contact_geometry.json"
    report.write_text(json.dumps(metadata, indent=2)+"\n")
    lines = ["# Q1 实测机构与人体映射", "", "单位：m。尺寸来自提供的 XML/STL；速度上限来自同目录 URDF。", "",
             "## 有效段长（参考 T 姿态）", "", "|段|长度 m|", "|---|---:|"]
    lines += [f"|{key}|{value:.9f}|" for key, value in lengths.items()]
    lines += ["", "## 逐级机构偏移", "", "|父 body|子 body|局部偏移 xyz m|距离 m|", "|---|---|---|---:|"]
    lines += [f"|{e['parent']}|{e['child']}|{e['local_offset_m']}|{e['distance_m']:.9f}|" for e in edges]
    lines += ["", "## 定义与限制", "", "- 髋和肩的转轴不重合；有效段长只是参考姿态定义，求解使用完整 MuJoCo 机构。",
              "- Hand 对应前臂 STL 的末端代理点，不是模型中不存在的腕关节。",
              "- 足部使用原 XML 的四个碰撞球，每个半径 0.008 m。支撑目标是球心 z=0.008 m。",
              "- 支撑约束允许骨盆六自由度移动；不对整段数据追加统一抬高。", "",
              f"站立参考骨盆高度：{data.qpos[2]:.9f} m。完整坐标、左右差异和速度限制见 q1_contact_geometry.json。"]
    (folder/"q1_contact_geometry.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"model": str(output), "geometry": str(report), "lengths_m": lengths}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("/root/zy/code/smp/artifacts/models/q1"))
    prepare(parser.parse_args().model_dir)
