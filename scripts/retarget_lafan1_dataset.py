"""Retarget LAFAN1 BVH files without a viewer or GPU; export PKL and CSV."""

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
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def collision_bottom(model, data, geom_ids):
    """World z of the lowest point of robot sphere/box/capsule collision geoms."""
    lowest = np.inf
    for i in geom_ids:
        size, kind = model.geom_size[i], model.geom_type[i]
        z_axis = data.geom_xmat[i].reshape(3, 3)[2]
        if kind == mj.mjtGeom.mjGEOM_SPHERE:
            extent = size[0]
        elif kind == mj.mjtGeom.mjGEOM_BOX:
            extent = np.abs(z_axis) @ size
        elif kind == mj.mjtGeom.mjGEOM_CAPSULE:
            extent = size[0] + abs(z_axis[2]) * size[1]
        else:
            continue
        lowest = min(lowest, data.geom_xpos[i, 2] - extent)
    return float(lowest)


def retarget_file(job):
    source, relative, target, robot, override, max_frames, ground_align = job
    source, target, relative = Path(source), Path(target), Path(relative)
    pkl_path = target / "pkl" / relative.with_suffix(".pkl")
    csv_path = target / "csv" / relative.with_suffix(".csv")
    report_path = target / "reports" / relative.with_suffix(".json")
    model_path = Path(ROBOT_XML_DICT[robot])
    config_path = Path(IK_CONFIG_DICT["bvh_lafan1"][robot])
    fingerprint = hashlib.sha256(model_path.read_bytes() + config_path.read_bytes()).hexdigest()
    if all(p.exists() for p in (pkl_path, csv_path, report_path)) and not override:
        report = json.loads(report_path.read_text())
        if (report.get("model_config_sha256") == fingerprint and
                report.get("max_frames") == max_frames and report.get("ground_align") == ground_align):
            return {**report, "skipped": True}
        raise ValueError(f"Existing output for {relative} uses different settings; use --override")
    started = time.monotonic()
    frames, height = load_bvh_file(str(source), format="lafan1")
    source_frames = len(frames)
    if max_frames:
        frames = frames[:max_frames]
    with source.open() as f:
        for line in f:
            if line.strip().startswith("Frame Time:"):
                fps = 1.0 / float(line.split(":", 1)[1])
                break
        else:
            raise ValueError(f"Missing BVH frame time: {source}")
    if abs(fps - 30) < 1e-3:
        fps = 30.0
    retargeter = GMR(src_human="bvh_lafan1", tgt_robot=robot,
                     actual_human_height=height, verbose=False)
    model = retargeter.model
    hinge_ids = [i for i in range(model.njnt) if model.jnt_type[i] == mj.mjtJoint.mjJNT_HINGE]
    if model.nq != 7 + len(hinge_ids) or model.nv != 6 + len(hinge_ids):
        raise ValueError("The motion model must contain one free root and otherwise only hinges")
    joint_names = [model.joint(i).name for i in hinge_ids]
    body_ids = list(range(1, model.nbody))
    body_names = [model.body(i).name for i in body_ids]
    collision_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] != 0 and
                     (model.geom_contype[i] or model.geom_conaffinity[i])]
    foot_ids = [i for i in collision_ids if "ankle_roll" in model.body(int(model.geom_bodyid[i])).name or
                "foot" in model.body(int(model.geom_bodyid[i])).name]
    if not collision_ids or not foot_ids:
        raise ValueError("Expected robot collision geometry and foot collision geometry")
    qpos = np.empty((len(frames), model.nq), dtype=np.float64)
    local_body_pos = np.empty((len(frames), len(body_ids), 3), dtype=np.float32)
    floor_z = np.empty(len(frames))
    foot_z = np.empty(len(frames))
    # Converge the initial pose before recording, preserving every source frame.
    for _ in range(3):
        retargeter.retarget(frames[0])
    for index, frame in enumerate(frames):
        qpos[index] = retargeter.retarget(frame)
        data = retargeter.configuration.data
        root_mat = data.xmat[model.body(retargeter.robot_root_name).id].reshape(3, 3)
        local_body_pos[index] = (data.xpos[body_ids] - qpos[index, :3]) @ root_mat
        floor_z[index] = collision_bottom(model, data, collision_ids)
        foot_z[index] = collision_bottom(model, data, foot_ids)
    if not np.isfinite(qpos).all():
        raise ValueError(f"Nonfinite IK output in {source}")
    # A constant translation preserves velocities; it is not a dynamics/contact fix.
    ground_shift = max(0.0, -float(floor_z.min())) if ground_align else 0.0
    qpos[:, 2] += ground_shift
    root_rot = qpos[:, [4, 5, 6, 3]].copy()  # MuJoCo wxyz -> output xyzw.
    flips = np.sum(root_rot[1:] * root_rot[:-1], axis=1) < 0
    signs = np.concatenate(([1], np.cumprod(np.where(flips, -1, 1))))
    root_rot *= signs[:, None]
    dof_pos = qpos[:, 7:]
    joint_ranges = model.jnt_range[hinge_ids]
    violations = np.maximum(joint_ranges[:, 0] - dof_pos, dof_pos - joint_ranges[:, 1])
    max_violation = float(max(0, violations.max()))
    if max_violation > 1e-5:
        raise ValueError(f"IK joint limit violation {max_violation} rad in {source}")
    joint_velocity = np.diff(dof_pos, axis=0) * fps
    max_velocity = float(np.abs(joint_velocity).max()) if len(joint_velocity) else 0.0
    elapsed = time.monotonic() - started
    report = {
        "source": str(source), "clip": relative.with_suffix("").as_posix(),
        "source_frames": source_frames, "frames": len(frames), "fps": fps,
        "max_frames": max_frames, "ground_align": ground_align,
        "model_config_sha256": fingerprint, "model": str(model_path.resolve()),
        "ik_config": str(config_path.resolve()), "joint_names": joint_names,
        "quaternion_order": "xyzw", "csv_columns": 7 + len(joint_names),
        "ground_shift_m": ground_shift, "raw_min_collision_z_m": float(floor_z.min()),
        "min_collision_z_m": float(floor_z.min() + ground_shift),
        "foot_bottom_z_quantiles_m": np.quantile(foot_z + ground_shift, [0, .01, .5, .99, 1]).tolist(),
        "max_joint_limit_violation_rad": max_violation,
        "max_joint_velocity_rad_s": max_velocity,
        "joint_velocity_abs_p99_rad_s": float(np.quantile(np.abs(joint_velocity), .99)) if len(joint_velocity) else 0.0,
        "requires_review": ground_shift > .05 or max_velocity > 20,
        "seconds": elapsed, "frames_per_second": len(frames) / elapsed,
        "pkl": str(pkl_path), "csv": str(csv_path),
    }
    motion = {
        "root_pos": qpos[:, :3].astype(np.float32),
        "root_rot": root_rot.astype(np.float32), "dof_pos": dof_pos.astype(np.float32),
        "local_body_pos": local_body_pos, "fps": fps,
        "link_body_list": body_names, "joint_names": joint_names,
        "quaternion_order": "xyzw", "metadata": report,
    }
    for path in (pkl_path, csv_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    pkl_temp = pkl_path.with_suffix(".pkl.tmp")
    with pkl_temp.open("wb") as f:
        pickle.dump(motion, f, protocol=pickle.HIGHEST_PROTOCOL)
    pkl_temp.replace(pkl_path)
    csv_temp = csv_path.with_suffix(".csv.tmp")
    np.savetxt(csv_temp, np.column_stack((motion["root_pos"], motion["root_rot"], motion["dof_pos"])),
               delimiter=",", fmt="%.9g")
    csv_temp.replace(csv_path)
    atomic_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src_folder", type=Path, required=True)
    parser.add_argument("--tgt_folder", type=Path, required=True)
    parser.add_argument("--robot", default="unitree_g1", choices=IK_CONFIG_DICT["bvh_lafan1"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pattern", default="*.bvh")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--target_fps", type=int, default=30)
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--no-ground-align", action="store_true")
    args = parser.parse_args()
    if args.target_fps != 30:
        parser.error("LAFAN1 is exported at its original 30 FPS; resample downstream")
    if args.workers < 1 or args.max_frames < 0:
        parser.error("workers must be positive; max-frames must be nonnegative")
    source, target = args.src_folder.resolve(), args.tgt_folder.resolve()
    paths = sorted(source.rglob(args.pattern))
    if not paths:
        parser.error(f"No BVH files in {source} matching {args.pattern}")
    target.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "robot": args.robot, "total": len(paths), "completed": 0,
             "failed": 0, "frames": 0, "workers": args.workers, "clips": [], "errors": []}
    status_path = target / "status.json"
    started = time.monotonic()
    atomic_json(status_path, state)
    jobs = [(str(path), str(path.relative_to(source)), str(target), args.robot,
             args.override, args.max_frames, not args.no_ground_align) for path in paths]
    print(f"START robot={args.robot} clips={len(jobs)} workers={args.workers} output={target}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(retarget_file, job): job[1] for job in jobs}
        for future in as_completed(pending):
            clip = pending[future]
            try:
                report = future.result()
                state["completed"] += 1
                state["frames"] += report["frames"]
                state["clips"].append(report)
                print(f"DONE {state['completed']}/{state['total']} {clip} frames={report['frames']} "
                      f"fps={report['frames_per_second']:.1f} shift={report['ground_shift_m']:.3f}m "
                      f"review={report['requires_review']}", flush=True)
            except Exception:
                state["failed"] += 1
                error = traceback.format_exc()
                state["errors"].append({"clip": clip, "error": error})
                print(f"FAILED {clip}\n{error}", flush=True)
            state["elapsed_seconds"] = time.monotonic() - started
            atomic_json(status_path, state)
    state["status"] = "complete" if not state["failed"] else "failed"
    state["elapsed_seconds"] = time.monotonic() - started
    atomic_json(status_path, state)
    print(f"FINISHED completed={state['completed']} failed={state['failed']} frames={state['frames']}", flush=True)
    return 1 if state["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
