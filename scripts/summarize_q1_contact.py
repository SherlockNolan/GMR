"""Write length mappings, contact comparisons and review intervals for Q1 v2."""

import argparse
import html
import json
from pathlib import Path

import numpy as np


def intervals(mask, fps):
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])+1
    return [{"first_frame": int(a), "end_frame_exclusive": int(b), "start_s": float(a/fps), "end_s": float(b/fps)}
            for a, b in zip(starts, ends)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/root/zy/code/smp/datasets/lafan1/q1_gmr_v2"))
    args = parser.parse_args()
    folder = args.output
    reports = [json.loads(p.read_text()) for p in sorted((folder/"reports").glob("*.json"))]
    if not reports:
        parser.error("No completed reports")
    state = json.loads((folder/"status.json").read_text())
    summary = {"status": state["status"], "clips": len(reports), "frames": sum(r["frames"] for r in reports),
               "constraint_review": [r["clip"] for r in reports if r["requires_review"]],
               "motion_fidelity_review": [], "motion_fidelity_review_intervals": {},
               "max_foot_penetration_m": max(r["max_foot_penetration_m"] for r in reports),
               "max_velocity_limit_violation_rad_s": max(r["max_velocity_limit_violation_rad_s"] for r in reports),
               "max_joint_limit_violation_rad": max(r["max_joint_limit_violation_rad"] for r in reports),
               "contact_failed_frames": sum(r["contact_failed_frames"] for r in reports),
               "relaxed_contact_frames": sum(r["relaxed_contact_frames"] for r in reports),
               "review_intervals": {}, "subjects": {}}
    for r in reports:
        data = np.load(folder/"contacts"/(r["clip"]+".npz"))
        error = np.nan_to_num(np.abs(data["actual"]-data["anchors"]), nan=0.).max(axis=(1, 2, 3))
        bad = data["relaxed"] | (error > .001) | (data["foot_penetration_m"] > .001)
        summary["review_intervals"][r["clip"]] = intervals(bad, r["fps"])
        # Explicit review heuristic, separate from hard contact/velocity failures.
        geometry = json.loads(Path(r["model"]).with_name("q1_contact_geometry.json").read_text())
        height = geometry["reference_root_height_m"]
        correction = np.linalg.norm(data["pelvis_adjustment_m"], axis=1)
        fidelity_flag = r["pelvis_adjustment_norm_m"]["p99"] > .25*height or correction.max() > .5*height
        if fidelity_flag:
            summary["motion_fidelity_review"].append(r["clip"])
            summary["motion_fidelity_review_intervals"][r["clip"]] = intervals(correction > .25*height, r["fps"])
        subject = r["clip"].split("_")[-1]
        summary["subjects"].setdefault(subject, []).append({"clip": r["clip"], **r["mapping"]})
    for version in ("v1", "v2"):
        metrics = [r[f"{version}_contact_metrics"] for r in reports if r[f"{version}_contact_metrics"]]
        summary[f"{version}_weighted_mean_support_height_abs_m"] = sum(m["support_height_abs_m"]["mean"]*m["support_height_abs_m"]["count"] for m in metrics)/sum(m["support_height_abs_m"]["count"] for m in metrics)
        summary[f"{version}_weighted_mean_support_xy_speed_m_s"] = sum(m["support_xy_speed_m_s"]["mean"]*m["support_xy_speed_m_s"]["count"] for m in metrics)/sum(m["support_xy_speed_m_s"]["count"] for m in metrics)
    summary["requires_review"] = sorted(set(summary["constraint_review"]+summary["motion_fidelity_review"]))
    summary["foot_dimensions_m"] = {}
    for side in ("left", "right"):
        spheres = [s for s in geometry["foot_spheres"] if s["side"] == side]
        centers = np.array([s["local_center_m"] for s in spheres])
        radii = np.array([s["radius_m"] for s in spheres])
        summary["foot_dimensions_m"][side] = {"heel_to_forefoot_center_span": float(np.ptp(centers[:, 0])),
            "collision_longitudinal_extent": float(np.ptp(np.r_[centers[:, 0]-radii, centers[:, 0]+radii])),
            "heel_center_width": float(np.ptp(centers[centers[:, 0] < 0, 1])),
            "forefoot_center_width": float(np.ptp(centers[centers[:, 0] > 0, 1])), "sphere_radius": float(radii[0])}
    (folder/"quality_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    lines = ["# Q1 第二版：逐段映射与支撑约束", "", f"已完成 {len(reports)} 段、{summary['frames']} 帧，30 FPS。", "",
             "## Q1 尺寸与映射定义", "",
             "完整机构偏移见 ../../../artifacts/models/q1/q1_contact_geometry.md（绝对路径见各段报告）。", "",
             "|人体 landmark|Q1 对应|处理|", "|---|---|---|",
             "|Hips|pelvis|按左右腿总长比缩放轨迹，IK 允许骨盆调整|",
             "|UpLeg → Leg|hip_pitch → knee|参考姿态下有效大腿长度；髋各轴偏移保留在模型中|",
             "|Leg → Foot|knee → ankle_pitch|Q1 小腿实测长度|",
             "|Hips → Spine2|pelvis → torso|方向映射到 Q1 腰部高度，胸部按模型刚体几何连接|",
             "|Arm → ForeArm|shoulder_roll → elbow|Q1 有效上臂长度，肩部三个偏置转轴由 IK 处理|",
             "|ForeArm → Hand|elbow → hand_mocap|前臂 STL 末端代理点|",
             "|Foot / Toe|脚跟代理 / 前掌碰撞球|识别支撑期，固定球心接触锚点|",
             "|Spine2 → Head|torso → head|只记录比例；Q1 头部固定，无独立 IK 目标|", "",
             "逐段人体长度为 FK landmark 距离中位数；腿和手臂分段缩放。躯干距离是 landmark 距离，不是脊柱骨链总长。", "",
             "Q1 脚跟/前掌球心纵向间距 0.130 m，碰撞球外缘总长 0.146 m；脚跟球心宽 0.050 m、前掌球心宽 0.040 m，球半径 0.008 m。", "",
             "人体 Foot→Toe 约 0.173 m；它与 Q1 球心间距的参考比为 0.751445，但 landmark 定义不同。足部直接使用 Q1 刚体几何和源姿态/支撑标签，不按此比拉伸足底。", "",
             "## 各人物的长度比例", "", "|人物|片段数|root 缩放|左大腿|左小腿|左上臂|左前臂|", "|---|---:|---:|---:|---:|---:|---:|"]
    for subject, records in sorted(summary["subjects"].items()):
        values = [np.median([m["segment_ratios"][f"left_{part}"] for m in records]) for part in ("thigh", "shank", "upperarm", "forearm")]
        root = np.median([m["root_translation_scale"] for m in records])
        lines.append(f"|{subject}|{len(records)}|{root:.6f}|"+"|".join(f"{v:.6f}" for v in values)+"|")
    lines += ["", "这 16 段 BVH 中五位人物使用相同的标准骨架段长，分别测量后的四肢比例一致（只有浮点误差）。", "",
              "逐段、双侧比例完整记录在 reports/*.json 的 mapping；汇总保留全部记录。", "",
              "## 接触质量比较", "", "统计仅使用人体估计的支撑期；跑步腾空帧不计入支撑高度误差。", "",
              "|片段|v1 高度均值 mm|v2 高度均值 mm|v1 支撑滑速 m/s|v2 支撑滑速 m/s|失败帧|复查|", "|---|---:|---:|---:|---:|---:|---|"]
    for r in reports:
        old, new = r["v1_contact_metrics"], r["v2_contact_metrics"]
        lines.append(f"|{r['clip']}|{old['support_height_abs_m']['mean']*1000:.3f}|{new['support_height_abs_m']['mean']*1000:.3f}|"
                     f"{old['support_xy_speed_m_s']['mean']:.6f}|{new['support_xy_speed_m_s']['mean']:.6f}|{r['contact_failed_frames']}|{r['clip'] in summary['requires_review']}|")
    lines += ["", "## 输出和使用", "", "- pkl/、csv/：30 FPS、22 关节，根四元数 XYZW；保持原关节顺序。",
              "- contacts/*.npz：逐帧左右脚跟/前掌标签、源高度与速度、固定锚点、实际球心、骨盆修正、约束放宽标记。",
              "- contacts/*_timeline.png：每段完整支撑期时间线。",
              "- quality_summary.json：失败区间和按人物汇总的长度比例。",
              "- previews/：片段预览，文件名注明截取区间。", "",
              "动作保真复查启发式：骨盆修正 P99 超过参考站立骨盆高度的 25%，或峰值超过 50%。这与接触约束失败分开记录。", "",
              "接触标签是平地 mocap 的估计，Foot 只是脚跟代理。未使用力传感器标签。",
              "约束允许骨盆六自由度调整；每次输出帧关节速度以 URDF 上限约束，不依赖内部 IK 迭代步长。",
              "仍需检查动作保真、碰撞、动力学跟踪；未据此启动 Q1 prior/RL。失败区间应修复或排除后再训练。"]
    (folder/"README.md").write_text("\n".join(lines)+"\n")
    cards = []
    for r in reports:
        clip = html.escape(r["clip"])
        review_note = "；骨盆调整较大，仍需复查转身段" if r["clip"] in summary["motion_fidelity_review"] else ""
        cards.append(f'<section><h2>{clip}</h2><video controls preload="metadata" src="{clip}_12s_32s.mp4"></video>'
                     f'<p>原版支撑高度误差 {r["v1_contact_metrics"]["support_height_abs_m"]["mean"]*1000:.2f} mm → '
                     f'{r["v2_contact_metrics"]["support_height_abs_m"]["mean"]*1000:.2f} mm；失败帧 {r["contact_failed_frames"]}{review_note}</p>'
                     f'<a href="../reports/{clip}.json">逐段报告和长度映射</a>'
                     f'<img src="../contacts/{clip}_timeline.png" alt="支撑期时间线"></section>')
    previews = folder/"previews"
    previews.mkdir(exist_ok=True)
    (previews/"index.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Q1 contact v2</title>'
        '<style>body{font:16px sans-serif;max-width:1400px;margin:30px auto;background:#edf1f5}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}section{background:white;padding:16px;border-radius:10px}video,img{width:100%}h2{font-size:20px}</style>'
        '<h1>Q1 contact v2：walk/run</h1><p>视频截取 12–32 秒，并显示人体估计的左右脚支撑状态。时间线覆盖每段完整录像。这里只回放运动学轨迹，尚未验证动力学平衡。</p>'
        '<p><a href="comparisons.html">查看 walk/run 的 v1/v2 同步对比</a> · '
        '<a href="walk2_subject3_183s_203s.mp4">复查 walk2_subject3 转身段（183–203 秒）</a></p><main>' + ''.join(cards)+'</main></html>')
    print(json.dumps({k:v for k,v in summary.items() if k not in ("subjects", "review_intervals", "motion_fidelity_review_intervals")}, indent=2))


if __name__ == "__main__":
    main()
