"""Audit expanded Q1 trajectories and export conservative interval manifests."""

import argparse
import html
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.ndimage import maximum_filter1d

from summarize_q1_contact import intervals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/zy/code/smp/datasets/lafan1/q1_gmr_v3"))
    args = parser.parse_args()
    root = args.root
    state = json.loads((root/"status.json").read_text())
    reports = [json.loads(p.read_text()) for p in sorted((root/"reports").glob("*.json"))]
    expected = len(list(Path(reports[0]["source"]).parent.glob("*.bvh")))
    # Partial reruns update selected clips; aggregate the complete dataset again.
    if len(reports) == expected and all(r["frames"] == r["source_frames"] and r["max_frames"] == 0 for r in reports):
        (root/"status_last_batch.json").write_text(json.dumps(state, indent=2)+"\n")
        state = {"status": "complete", "total": expected, "completed": len(reports), "failed": 0,
                 "frames": sum(r["frames"] for r in reports), "clips": reports, "errors": [], "last_batch_total": state["total"]}
        (root/"status.json").write_text(json.dumps(state, indent=2)+"\n")
    geometry = json.loads(Path(reports[0]["model"]).with_name("q1_contact_geometry.json").read_text())
    pelvis_threshold = .25*geometry["reference_root_height_m"]
    result = {"status": state["status"], "expected_clips": expected, "completed_clips": len(reports),
              "source_frames": sum(r["source_frames"] for r in reports), "exported_frames": sum(r["frames"] for r in reports),
              "failed_clips": state["errors"], "families": {}, "clips": [], "heuristics": {
                  "contact_error_m": .001, "floor_penetration_m": .001, "body_surface_error_m": .02,
                  "single_frame_joint_change_rad": .35, "pelvis_adjustment_m": pelvis_threshold,
                  "candidate_min_frames": 90, "candidate_bad_frame_padding": 2},
              "dynamic_validation": False, "obstacle_geometry_available": False,
              "candidate_intervals_are_not_training_certification": True}
    candidates = []
    for r in reports:
        name, fps = r["clip"], r["fps"]
        data = np.load(root/"contacts"/(name+".npz"))
        with (root/"pkl"/(name+".pkl")).open("rb") as stream:
            motion = pickle.load(stream)
        n = r["frames"]
        error = np.nan_to_num(np.abs(data["actual"]-data["anchors"]), nan=0.).max(axis=(1, 2, 3))
        collision_z = data["collision_bottom_z_m"].min(axis=1)
        numerical = data["relaxed"] | (error > .001) | (data["collision_floor_penetration_m"] > .001)
        recovery = np.zeros(n, dtype=bool)
        for field in ("solver_recovery", "frozen_output", "soft_foot_contact"):
            if field in data:
                recovery |= data[field]
        numerical |= recovery
        correction = np.linalg.norm(data["pelvis_adjustment_m"], axis=1)
        large_correction = correction > pelvis_threshold
        joint_step = np.max(np.abs(np.diff(motion["dof_pos"], axis=0)), axis=1)
        abrupt = np.r_[False, joint_step > .35]
        body = data["body_labels"].any(axis=1)
        body_miss = np.any(data["body_labels"] & (np.abs(data["body_bottom_z_m"]) > .02), axis=1)
        source_low = data["body_source_height_m"][:, 4] < .68*data["body_source_height_m"][0, 4]
        body_review = body | source_low
        reasons = []
        if numerical.any():
            reasons.append("contact_or_floor_constraint")
        if recovery.any():
            reasons.append("recorded_solver_recovery_or_deferred_contact")
        if large_correction.any():
            reasons.append("large_pelvis_adjustment")
        if abrupt.any():
            reasons.append("single_frame_joint_change_over_20deg")
        if body_review.any():
            reasons.append("body_contact_or_low_pose")
        if r["needs_environment_geometry"]:
            reasons.append("missing_obstacle_geometry")
        bad = numerical | large_correction | abrupt | body_review
        bad = maximum_filter1d(bad.astype(np.uint8), size=5, mode="nearest").astype(bool)
        good = ~bad
        good[:min(90, n)] = False
        if r["needs_environment_geometry"]:
            good[:] = False
        ranges = [entry for entry in intervals(good, fps) if entry["end_frame_exclusive"]-entry["first_frame"] >= 90]
        candidates.extend({"clip": name, "pkl": str(root/"pkl"/(name+".pkl")), **entry} for entry in ranges)
        entry = {"clip": name, "family": r["motion_family"], "frames": n, "review_reasons": reasons,
                 "constraint_failed_frames": int(numerical.sum()), "body_contact_failed_frames": int(body_miss.sum()),
                 "large_pelvis_adjustment_frames": int(large_correction.sum()), "abrupt_joint_frames": int(abrupt.sum()),
                 "body_contact_frames": int(body.sum()), "no_foot_support_fraction": float((~data["labels"].any(axis=(1, 2))).mean()),
                 "recovery_or_deferred_frames": int(recovery.sum()),
                 "flight_candidate_fraction": float((~data["labels"].any(axis=(1, 2)) & ~body & ~source_low).mean()),
                 "constraint_review_intervals": intervals(numerical, fps),
                 "motion_review_intervals": intervals(large_correction | abrupt | body_review, fps),
                 "kinematic_candidate_intervals": ranges, "kinematic_candidate_frames": sum(e["end_frame_exclusive"]-e["first_frame"] for e in ranges),
                 "foot_height_mean_mm": r["v3_contact_metrics"]["support_height_abs_m"].get("mean", 0.)*1000,
                 "foot_sliding_mean_mm_s": r["v3_contact_metrics"]["support_xy_speed_m_s"].get("mean", 0.)*1000,
                 "min_collision_surface_z_m": float(collision_z.min()), "max_joint_velocity_rad_s": r["max_joint_velocity_rad_s"]}
        # Select a representative active interval rather than only the initial T pose.
        if r["needs_environment_geometry"]:
            interesting = np.flatnonzero(data["inferred_plane_height_m"].max(axis=(1, 2)) > .02)
        elif body_review.any():
            interesting = np.flatnonzero(body_review)
        elif abrupt.any():
            interesting = np.flatnonzero(abrupt)
        else:
            interesting = np.array([min(12*fps, max(0, n-600))])
        centre = int(interesting[len(interesting)//2]) if len(interesting) else min(360, max(0, n-600))
        start = max(0, min(centre-90, n-600))
        entry["preview_start_seconds"] = start/fps
        entry["preview_frames"] = min(600, n-start)
        result["clips"].append(entry)
        family = result["families"].setdefault(r["motion_family"], {"clips": 0, "frames": 0, "constraint_failed_frames": 0, "review_clips": 0,
                                                                   "kinematic_candidate_frames": 0})
        family["clips"] += 1
        family["frames"] += n
        family["constraint_failed_frames"] += entry["constraint_failed_frames"]
        family["review_clips"] += int(bool(reasons))
        family["kinematic_candidate_frames"] += entry["kinematic_candidate_frames"]
    result["constraint_failed_frames"] = sum(e["constraint_failed_frames"] for e in result["clips"])
    result["body_contact_failed_frames"] = sum(e["body_contact_failed_frames"] for e in result["clips"])
    result["recovery_or_deferred_frames"] = sum(e["recovery_or_deferred_frames"] for e in result["clips"])
    result["review_clips"] = sum(bool(e["review_reasons"]) for e in result["clips"])
    result["kinematic_candidate_frames"] = sum(e["kinematic_candidate_frames"] for e in result["clips"])
    for version in ("v1", "v3"):
        valid = [r[f"{version}_contact_metrics"] for r in reports if r[f"{version}_contact_metrics"]]
        for key in ("support_height_abs_m", "support_xy_speed_m_s"):
            count = sum(m[key]["count"] for m in valid)
            result[f"{version}_weighted_{key}"] = sum(m[key].get("mean", 0.)*m[key]["count"] for m in valid)/max(1, count)
    (root/"quality_summary.json").write_text(json.dumps(result, indent=2)+"\n")
    (root/"kinematic_candidate_intervals.json").write_text(json.dumps({"dynamic_validation": False,
        "description": "Conservative numerical/fidelity candidate intervals; review before diffusion or RL. Ground/body contacts and all obstacle clips are excluded.",
        "intervals": candidates}, indent=2)+"\n")
    lines = ["# Q1 扩展动作重定向（v3）", "", f"完成 {len(reports)}/{expected} 段、{result['exported_frames']} 帧，30 FPS。", "",
             "## 动作覆盖", "", "|动作类型|段数|帧数|约束异常或恢复帧|需要复查的段数|候选帧数|", "|---|---:|---:|---:|---:|---:|"]
    lines += [f"|{name}|{value['clips']}|{value['frames']}|{value['constraint_failed_frames']}|{value['review_clips']}|{value['kinematic_candidate_frames']}|"
              for name, value in sorted(result["families"].items())]
    lines += ["", "## 处理方法", "", "- 按 Q1 实测分段长度映射人体；参考模型仍是 q1_contact_v2.xml，运动版本为 v3。",
              "- 站立帧校准脚部高度，并用脚底朝向门控，避免躺倒时把脚踝低位误当成平放脚底。",
              "- 支撑脚固定球心接触点；跳跃期间释放支撑约束，骨盆保持可调整。",
              "- 原模型所有碰撞 box/sphere 的最低点参与地面约束，覆盖手臂、躯干、头、腿。",
              "- 手、膝、骨盆、躯干和头的支撑期采用软接触高度目标，允许滚动及水平移动；与脚部固定锚点不同。",
              "- 障碍序列由慢速稳定的高位脚掌推断支撑平面；没有从 BVH 恢复障碍物实体几何。", "",
              "## 质量与复查", "", f"脚部/地面约束异常或求解恢复、接触延迟帧：{result['constraint_failed_frames']}。",
              f"人体估计身体支撑期内，Q1 接触面高度偏差超过 2 cm 的帧：{result['body_contact_failed_frames']}。",
              "接触目标和碰撞几何是运动学近似；身体接触误差会保留并标记，不通过统一抬高整段掩盖。",
              "solver_recovery、frozen_output、soft_foot_contact 明确记录求解恢复、回退帧和延迟的接触目标；这些帧全部排除出候选区间。",
              "单帧关节变化超过 0.35 rad（约 20°）、骨盆修正超过参考站立高度 25% 的区间也单独标记；这些是工程复查阈值。",
              "脚部无支撑并不自动等于身体腾空，quality_summary.json 分别记录脚部离开支撑率和腾空候选率。", "",
              "## 文件", "", "- pkl/、csv/：完整原始时间轴，每段全部帧；根四元数 XYZW，22 个关节。",
              "- contacts/：左右脚、身体支撑标签，推断地形高度，全身碰撞最低点，骨盆修正。",
              "- quality_summary.json：逐段复查原因和时间区间。",
              "- kinematic_candidate_intervals.json：保守筛选的连续区间，至少 3 秒，异常前后各排除 2 帧，排除首 3 秒。",
              "- candidates/：运行 scripts/export_q1_candidates.py 后生成候选 PKL/CSV；manifest.json 记录来源和起止帧。",
              "- previews/index.html：每段代表动作的 20 秒预览；不是整段视频。",
              "候选区间尚未经动力学验证；所有障碍序列、身体支撑/低姿态区间暂不进入候选集。"]
    (root/"README.md").write_text("\n".join(lines)+"\n")
    preview = root/"previews"
    preview.mkdir(exist_ok=True)
    sections = []
    for family in sorted(result["families"]):
        cards = []
        for entry in result["clips"]:
            if entry["family"] != family:
                continue
            name = html.escape(entry["clip"])
            reason = html.escape(", ".join(entry["review_reasons"]) or "numerical checks passed")
            cards.append(f'<article><h3>{name}</h3><video controls preload="none" src="{name}.mp4"></video>'
                         f'<p>从 {entry["preview_start_seconds"]:.2f}s 起；脚部高度误差 {entry["foot_height_mean_mm"]:.3f} mm</p>'
                         f'<p>复查：{reason}</p><a href="../reports/{name}.json">报告</a> · '
                         f'<a href="../contacts/{name}_timeline.png">完整支撑时间线</a></article>')
        sections.append(f'<h2>{html.escape(family)}</h2><div class="grid">'+''.join(cards)+'</div>')
    (preview/"index.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Q1 expanded actions</title>'
        '<style>body{max-width:1450px;margin:25px auto;font:16px sans-serif;background:#edf1f5}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}article{background:white;padding:16px;border-radius:8px}video{width:100%}p{overflow-wrap:anywhere}</style>'
        '<h1>Q1 v3：完整动作覆盖</h1><p>代表动作片段与逐段复查原因。支撑高度与标签是估计；推断高位平面以橙色显示。运动学回放，尚未验证动力学。</p>'+''.join(sections)+'</html>')
    print(json.dumps({k:v for k,v in result.items() if k not in ("clips", "failed_clips")}, indent=2))


if __name__ == "__main__":
    main()
