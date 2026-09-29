"""Materialize audited kinematic candidate intervals; no dynamics certification."""

import argparse
import json
from pathlib import Path
import pickle

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/zy/code/smp/datasets/lafan1/q1_gmr_v3"))
    args = parser.parse_args()
    root = args.root
    recipe = json.loads((root/"kinematic_candidate_intervals.json").read_text())
    output = root/"candidates"
    for folder in ("pkl", "csv"):
        (output/folder).mkdir(parents=True, exist_ok=True)
    current_name, motion = None, None
    entries = []
    for item in recipe["intervals"]:
        if current_name != item["clip"]:
            current_name = item["clip"]
            with Path(item["pkl"]).open("rb") as stream:
                motion = pickle.load(stream)
        start, end = item["first_frame"], item["end_frame_exclusive"]
        count = len(motion["root_pos"])
        name = f"{current_name}__f{start:06d}_{end:06d}"
        cropped = {key: value[start:end].copy() if isinstance(value, np.ndarray) and len(value.shape) and value.shape[0] == count else value
                   for key, value in motion.items()}
        cropped["metadata"] = {"source_clip": current_name, "source_motion": item["pkl"],
                               "source_frame_range": [start, end], "frames": end-start, "fps": motion["fps"],
                               "model": motion["metadata"]["model"], "motion_recipe_fingerprint": motion["metadata"]["fingerprint"],
                               "dynamic_validation": False, "body_patch_order": motion["metadata"].get("body_patch_order", []),
                               "needs_environment_geometry": False, "quality_summary": str(root/"quality_summary.json")}
        pkl_path, csv_path = output/"pkl"/(name+".pkl"), output/"csv"/(name+".csv")
        with pkl_path.open("wb") as stream:
            pickle.dump(cropped, stream, protocol=pickle.HIGHEST_PROTOCOL)
        np.savetxt(csv_path, np.column_stack((cropped["root_pos"], cropped["root_rot"], cropped["dof_pos"])), delimiter=",", fmt="%.9g")
        entries.append({**item, "frames": end-start, "candidate_pkl": str(pkl_path), "candidate_csv": str(csv_path)})
    summary = {"status": "complete", "segments": len(entries), "frames": sum(e["frames"] for e in entries),
               "dynamic_validation": False, "source_manifest": str(root/"kinematic_candidate_intervals.json"), "segments_metadata": entries}
    (output/"manifest.json").write_text(json.dumps(summary, indent=2)+"\n")
    (output/"README.md").write_text(f"# Q1 运动学候选片段\n\n共 {summary['segments']} 段、{summary['frames']} 帧，原始 30 FPS。\n\n"
        "来自 q1_gmr_v3 的数值/保真筛选区间，已排除推断障碍支撑、身体支撑或低姿态、求解恢复/接触延迟，以及大骨盆修正和单帧关节突变区间。\n\n"
        "每段原始来源和起止帧记录在 manifest.json；CSV 为根 XYZ、四元数 XYZW、22 个关节，共 29 列。\n\n"
        "片段是运动学候选数据，尚未通过动力学跟踪验证；SMP 仍需适配 Q1 的特征维度和机器人配置。\n")
    print(json.dumps({k:v for k,v in summary.items() if k != "segments_metadata"}, indent=2))


if __name__ == "__main__":
    main()
