"""Retarget flat-ground LAFAN1 walk/run to Q1 with contact-aware segment IK."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

os.environ.setdefault("MUJOCO_GL", "disable")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw
from general_motion_retargeting.q1_contact import Q1ContactRetargeter, measure_human, detect_support
from general_motion_retargeting.utils.lafan1 import load_bvh_file


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def statistics(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {"count": int(len(values)), "mean": float(values.mean()), "p50": float(np.quantile(values, .5)),
            "p95": float(np.quantile(values, .95)), "p99": float(np.quantile(values, .99)), "max": float(values.max())}


def contact_metrics(points, labels, fps, anchors=None, reference_z=.008):
    supported = labels
    height = np.abs(points[..., 2]-reference_z)[supported]
    consecutive = labels[1:] & labels[:-1]
    speed = np.linalg.norm(np.diff(points[..., :2], axis=0), axis=-1)*fps
    drift = []
    for side in range(2):
        for patch in range(2):
            previous = False
            origin = None
            for i in range(len(points)):
                if supported[i, side, patch]:
                    if not previous:
                        origin = points[i, side, patch, :2].copy()
                    drift.append(np.linalg.norm(points[i, side, patch, :2]-origin))
                previous = supported[i, side, patch]
    metrics = {"support_height_abs_m": statistics(height), "support_xy_speed_m_s": statistics(speed[consecutive]),
               "within_patch_phase_xy_drift_m": statistics(drift)}
    if anchors is not None:
        metrics["anchor_position_error_m"] = statistics(np.linalg.norm(points-anchors, axis=-1)[supported])
    return metrics


def process(job):
    source, target, model_folder, limit, override, *options = job
    expanded = bool(options and options[0] == "expanded")
    body_contact_cost = float(options[1]) if len(options) > 1 else 100.
    recovery = bool(options[2]) if len(options) > 2 else False
    source, target, model_folder = Path(source), Path(target), Path(model_folder)
    name = source.stem
    report_path = target/"reports"/(name+".json")
    orientation_config = Path(__file__).resolve().parents[1]/"general_motion_retargeting/ik_configs/bvh_lafan1_to_q1.json"
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    extension = (Path(__file__).resolve().parents[1]/"general_motion_retargeting/q1_multi_contact.py").read_bytes() if expanded else b""
    if recovery:
        extension += (Path(__file__).resolve().parents[1]/"general_motion_retargeting/q1_contact_recovery.py").read_bytes()
    fingerprint = hashlib.sha256((model_folder/"q1_contact_v2.xml").read_bytes()+
        (model_folder/"q1_contact_geometry.json").read_bytes()+
        orientation_config.read_bytes()+
        (Path(__file__).resolve().parents[1]/"general_motion_retargeting/q1_contact.py").read_bytes()+
        Path(__file__).read_bytes()+extension+(f"{body_contact_cost}:{recovery}".encode() if expanded else b"")).hexdigest()
    if report_path.exists() and not override:
        existing = json.loads(report_path.read_text())
        if existing["fingerprint"] == fingerprint and existing["max_frames"] == limit and existing["source_sha256"] == source_sha256:
            return existing
        raise ValueError(f"Different existing settings for {name}; use --override")
    started = time.monotonic()
    metadata = json.loads((model_folder/"q1_contact_geometry.json").read_text())
    frames, _ = load_bvh_file(str(source))
    positions, mapping = measure_human(frames, metadata)
    extra = None
    family = "walk" if name.startswith("walk") else "run"
    if expanded:
        from general_motion_retargeting.q1_multi_contact import Q1MultiContactRetargeter, detect_contacts, motion_family, BODY_PATCHES
        family = motion_family(name)
        cls = Q1MultiContactRetargeter
        if recovery:
            from general_motion_retargeting.q1_contact_recovery import Q1RecoverableRetargeter
            cls = Q1RecoverableRetargeter
        retargeter = cls(metadata, mapping, positions, body_contact_cost=body_contact_cost)
        labels, heights, speeds, support, extra = detect_contacts(frames, positions, mapping, family, retargeter.offsets)
    else:
        labels, heights, speeds, support = detect_support(positions, mapping)
        retargeter = Q1ContactRetargeter(metadata, mapping, positions)
    total_frames = len(frames)
    if limit:
        frames = frames[:limit]
        labels, heights, speeds = labels[:limit], heights[:limit], speeds[:limit]
        if extra is not None:
            extra = {key: value[:limit] for key, value in extra.items()}
    n, fps = len(frames), 30
    model = retargeter.model
    qpos = np.empty((n, model.nq))
    anchors = np.full((n, 2, 2, 3), np.nan)
    actual = anchors.copy()
    selected = np.full((n, 2, 2), -1, dtype=np.int32)
    floor = np.empty(n)
    relaxed = np.empty(n, dtype=bool)
    tolerances = np.empty(n)
    contact_error = np.empty(n)
    adjustment = np.empty((n, 3))
    body_bottom = np.empty((n, len(BODY_PATCHES))) if expanded else None
    collision_bottom = np.empty((n, len(retargeter.contacts.geoms))) if expanded else None
    foot_floor = np.empty(n)
    solver_recovery = np.zeros(n, dtype=bool)
    frozen_output = np.zeros(n, dtype=bool)
    soft_foot_contact = np.zeros(n, dtype=bool)
    body_ids = np.arange(1, model.nbody)
    local_body_pos = np.empty((n, len(body_ids), 3), dtype=np.float32)
    for i, frame in enumerate(frames):
        kwargs = {"levels": extra["levels"][i], "body_labels": extra["body_labels"][i]} if expanded else {}
        qpos[i], result = retargeter.retarget(frame, labels[i], first=i == 0, **kwargs)
        anchors[i], actual[i], selected[i] = result["anchors"], result["actual"], result["site_ids"]
        floor[i], relaxed[i], tolerances[i] = result["penetration"], result["relaxed"], result["tolerance"]
        adjustment[i], contact_error[i] = result["pelvis_adjustment"], result["contact_error"]
        foot_floor[i] = result.get("foot_penetration", result["penetration"])
        solver_recovery[i], frozen_output[i], soft_foot_contact[i] = result.get("solver_recovery", False), result.get("frozen", False), result.get("soft_foot_contact", False)
        if expanded:
            body_bottom[i], collision_bottom[i] = result["body_bottom_z"], result["collision_bottom_z"]
        data = retargeter.configuration.data
        rotation = data.xmat[model.body("pelvis").id].reshape(3, 3)
        local_body_pos[i] = (data.xpos[body_ids]-qpos[i, :3]) @ rotation
        if i and i % 3000 == 0:
            print(f"PROGRESS {name} {i}/{n}", flush=True)
    hinge_ids = [i for i in range(model.njnt) if model.jnt_type[i] == mj.mjtJoint.mjJNT_HINGE]
    joint_names = [model.joint(i).name for i in hinge_ids]
    vmax = retargeter.frame_bounds.vmax
    velocity = np.abs(np.diff(qpos[:, 7:], axis=0)*fps)
    root_rot = qpos[:, [4, 5, 6, 3]].copy()
    flips = np.sum(root_rot[1:]*root_rot[:-1], axis=1) < 0
    root_rot *= np.r_[1, np.cumprod(np.where(flips, -1, 1))][:, None]
    reference_z = anchors[..., 2] if expanded else .008
    new_metrics = contact_metrics(actual, labels, fps, anchors, reference_z)
    baseline_path = target.parent/"q1_gmr/pkl"/(name+".pkl")
    old_metrics = None
    if baseline_path.exists():
        with baseline_path.open("rb") as f:
            old = pickle.load(f)
        data = mj.MjData(model)
        old_points = np.full_like(actual, np.nan)
        for i in range(n):
            data.qpos[:3] = old["root_pos"][i]
            data.qpos[3:7] = old["root_rot"][i, [3, 0, 1, 2]]
            data.qpos[7:] = old["dof_pos"][i]
            mj.mj_forward(model, data)
            for side in range(2):
                for patch in range(2):
                    if selected[i, side, patch] >= 0:
                        old_points[i, side, patch] = data.site_xpos[selected[i, side, patch]]
        old_metrics = contact_metrics(old_points, labels, fps, reference_z=reference_z)
    velocity_violation = float(np.maximum(velocity-vmax, 0).max())
    ranges = model.jnt_range[hinge_ids]
    range_violation = float(np.maximum(np.maximum(ranges[:, 0]-qpos[:, 7:], qpos[:, 7:]-ranges[:, 1]), 0).max())
    report = {"clip": name, "source": str(source), "source_frames": total_frames, "frames": n, "fps": fps,
              "contact_mode": "expanded" if expanded else "flat", "motion_family": family,
              "source_sha256": source_sha256, "orientation_config": str(orientation_config),
              "orientation_config_sha256": hashlib.sha256(orientation_config.read_bytes()).hexdigest(),
              "model": metadata["model"], "fingerprint": fingerprint, "max_frames": limit,
              "mapping": mapping, "source_support": support, "v2_contact_metrics": new_metrics,
              "v1_contact_metrics": old_metrics, "max_foot_penetration_m": float(foot_floor.max()),
              "max_collision_floor_penetration_m": float(floor.max()),
              "max_joint_velocity_rad_s": float(velocity.max()), "urdf_velocity_limits_rad_s": dict(zip(joint_names, vmax.tolist())),
              "max_velocity_limit_violation_rad_s": velocity_violation, "max_joint_limit_violation_rad": range_violation,
              "relaxed_contact_frames": int(relaxed.sum()), "contact_failed_frames": int(np.sum(contact_error > .001)),
              "pelvis_adjustment_norm_m": statistics(np.linalg.norm(adjustment, axis=1)),
              "pelvis_vertical_adjustment_m": {"min": float(adjustment[:, 2].min()), "max": float(adjustment[:, 2].max())},
              "ground_shift_m": 0., "seconds": time.monotonic()-started,
              "contact_box_tolerance_m": .0004, "foot_floor_tolerance_m": .00005,
              "requires_review": bool(relaxed.any() or floor.max() > .001 or contact_error.max() > .001 or range_violation > 1e-5 or velocity_violation > 1e-5),
              "limitations": ["Contact labels are estimates from flat-ground human mocap, not force measurements.",
                              "Kinematic foot constraints and URDF velocity bounds do not prove dynamic feasibility or balance.",
                              "Any relaxed or failed contact frames are reported and must not be silently included in training."]}
    if expanded:
        report["v3_contact_metrics"] = new_metrics
        report["body_surface_height_abs_m"] = statistics(np.abs(body_bottom)[extra["body_labels"]])
        report["body_contact_failed_frames"] = int(np.sum(np.any(extra["body_labels"] & (np.abs(body_bottom) > .02), axis=1)))
        report["collision_geom_ids"] = retargeter.contacts.geoms
        report["body_patch_order"] = [p[0] for p in BODY_PATCHES]
        report["body_contact_cost"] = retargeter.body_contact_cost
        report["contact_recovery_enabled"] = recovery
        report["solver_recovery_frames"] = int(solver_recovery.sum())
        report["frozen_output_frames"] = int(frozen_output.sum())
        report["soft_foot_contact_frames"] = int(soft_foot_contact.sum())
        report["needs_environment_geometry"] = family == "obstacles"
        report["needs_body_contact_review"] = bool(extra["body_labels"].any() or support["low_body_fraction"] > .01)
        report["requires_review"] |= report["needs_environment_geometry"] or report["needs_body_contact_review"]
        report["limitations"][0] = "Contact labels and elevated support planes are estimates from BVH markers; no force/environment geometry labels."
        report["limitations"].append("Non-foot contacts use soft body-surface height goals; collision primitive nonpenetration is enforced for the whole body.")
    motion = {"root_pos": qpos[:, :3].astype(np.float32), "root_rot": root_rot.astype(np.float32),
              "dof_pos": qpos[:, 7:].astype(np.float32), "local_body_pos": local_body_pos,
              "fps": fps, "joint_names": joint_names, "link_body_list": [model.body(i).name for i in body_ids],
              "quaternion_order": "xyzw", "contact_labels": labels, "metadata": report}
    if expanded:
        motion["body_contact_labels"] = extra["body_labels"]
        motion["contact_plane_height_m"] = np.nan_to_num(anchors[..., 2]-.008, nan=0.).astype(np.float32)
        motion["contact_recovery_flags"] = solver_recovery | frozen_output | soft_foot_contact
    for directory in ("pkl", "csv", "contacts", "reports"):
        (target/directory).mkdir(parents=True, exist_ok=True)
    with (target/"pkl"/(name+".pkl")).open("wb") as f:
        pickle.dump(motion, f, protocol=pickle.HIGHEST_PROTOCOL)
    np.savetxt(target/"csv"/(name+".csv"), np.column_stack((motion["root_pos"], root_rot, motion["dof_pos"])), delimiter=",", fmt="%.9g")
    contact_extra = {"body_labels": extra["body_labels"], "body_source_height_m": extra["body_heights"],
                     "body_bottom_z_m": body_bottom, "collision_bottom_z_m": collision_bottom,
                     "collision_floor_penetration_m": floor, "body_patch_order": np.array(report["body_patch_order"]),
                     "inferred_plane_height_m": extra["levels"], "solver_recovery": solver_recovery,
                     "frozen_output": frozen_output, "soft_foot_contact": soft_foot_contact} if expanded else {}
    np.savez_compressed(target/"contacts"/(name+".npz"), labels=labels, source_height_m=heights,
                        source_horizontal_speed_m_s=speeds, anchors=anchors, actual=actual, site_ids=selected,
                        pelvis_adjustment_m=adjustment, foot_penetration_m=foot_floor, relaxed=relaxed,
                        contact_tolerance_m=tolerances, fps=fps, patch_order=np.array(["heel_proxy", "toe"]),
                        side_order=np.array(["left", "right"]), **contact_extra)
    save_json(report_path, report)
    # A readable support timeline; one column bins the entire clip, preserving flight.
    image = Image.new("RGB", (1200, 210), "white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 8), f"{name}: estimated human support phases, {n/fps:.1f}s; green=support, grey=flight", fill="black")
    for row, title in enumerate(("Left heel proxy", "Left toe", "Right heel proxy", "Right toe")):
        draw.text((10, 45+row*35), title, fill="black")
        bits = labels[:, row//2, row%2]
        for x in range(1000):
            start, end = int(x*n/1000), max(int((x+1)*n/1000), int(x*n/1000)+1)
            fraction = bits[start:end].mean()
            colour = tuple(int((1-fraction)*c+fraction*d) for c, d in zip((225, 225, 225), (30, 145, 70)))
            draw.line((190+x, 43+row*35, 190+x, 65+row*35), fill=colour)
    image.save(target/"contacts"/(name+"_timeline.png"))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/root/zy/code/smp/datasets/lafan1/raw"))
    parser.add_argument("--output", type=Path, default=Path("/root/zy/code/smp/datasets/lafan1/q1_gmr_v2"))
    parser.add_argument("--model-dir", type=Path, default=Path("/root/zy/code/smp/artifacts/models/q1"))
    parser.add_argument("--clips", nargs="*")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--contact-mode", choices=("flat", "expanded"), default="flat")
    parser.add_argument("--all", action="store_true", help="Process all LAFAN1 motion families")
    parser.add_argument("--body-contact-cost", type=float, default=100.)
    parser.add_argument("--recover-contacts", action="store_true", help="Record contact deferral/pose rollback when estimated constraints are infeasible")
    args = parser.parse_args()
    if args.max_frames < 0 or args.workers < 1 or args.body_contact_cost <= 0:
        parser.error("max-frames must be nonnegative and workers positive")
    paths = sorted(set(args.source.glob("walk*.bvh")) | set(args.source.glob("run*.bvh")))
    if args.all:
        paths = sorted(args.source.glob("*.bvh"))
    if args.clips:
        paths = [args.source/(name.removesuffix(".bvh")+".bvh") for name in args.clips]
    if not paths:
        parser.error("No walk/run clips")
    args.output.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "total": len(paths), "completed": 0, "failed": 0, "clips": [], "errors": []}
    save_json(args.output/"status.json", state)
    started = time.monotonic()
    jobs = [(str(p), str(args.output), str(args.model_dir), args.max_frames, args.override, args.contact_mode, args.body_contact_cost, args.recover_contacts) for p in paths]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(process, job): Path(job[0]).stem for job in jobs}
        for future in as_completed(pending):
            try:
                report = future.result()
                state["clips"].append(report)
                state["completed"] += 1
                print(f"DONE {report['clip']} frames={report['frames']} penetration={report['max_foot_penetration_m']:.6f} "
                      f"contact_failed={report['contact_failed_frames']} relaxed={report['relaxed_contact_frames']} "
                      f"seconds={report['seconds']:.1f}", flush=True)
            except Exception:
                state["failed"] += 1
                error = traceback.format_exc()
                state["errors"].append({"clip": pending[future], "error": error})
                print(error, flush=True)
            state["elapsed_seconds"] = time.monotonic()-started
            save_json(args.output/"status.json", state)
    state["status"] = "complete" if not state["failed"] else "failed"
    save_json(args.output/"status.json", state)
    return int(bool(state["failed"]))


if __name__ == "__main__":
    raise SystemExit(main())
