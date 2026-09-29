"""Reprocess elevated support, ground contact and jump takeoff intervals."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import shutil
import time
import traceback

from retarget_q1_contact import process, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/zy/code/smp/datasets/lafan1/q1_gmr_v3"))
    parser.add_argument("--source", type=Path, default=Path("/root/zy/code/smp/datasets/lafan1/raw"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--body-contact-cost", type=float, default=100.)
    args = parser.parse_args()
    jobs = []
    archive = args.root/"reports_initial"
    archive.mkdir(exist_ok=True)
    for path in sorted(args.source.glob("*.bvh")):
        report_path = args.root/"reports"/(path.stem+".json")
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        selected = not report or report.get("needs_environment_geometry") or report.get("needs_body_contact_review") or report.get("motion_family") == "jumps"
        if not selected:
            continue
        if report and not (archive/report_path.name).exists():
            shutil.copy2(report_path, archive/report_path.name)
        jobs.append((str(path), str(args.root), "/root/zy/code/smp/artifacts/models/q1", 0, True, "expanded", args.body_contact_cost))
    state = {"status": "running", "total": len(jobs), "completed": 0, "failed": 0, "clips": [], "errors": [],
             "reason": "consistent single-foot support planes, stronger body contact goals, vertical jump takeoff gate"}
    started = time.monotonic()
    save_json(args.root/"refinement_status.json", state)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(process, job): Path(job[0]).stem for job in jobs}
        for future in as_completed(futures):
            try:
                r = future.result()
                state["clips"].append(r)
                state["completed"] += 1
                print(f"DONE {state['completed']}/{len(jobs)} {r['clip']} foot_failed={r['contact_failed_frames']} "
                      f"body_failed={r['body_contact_failed_frames']} seconds={r['seconds']:.1f}", flush=True)
            except Exception:
                state["failed"] += 1
                error = traceback.format_exc()
                state["errors"].append({"clip": futures[future], "error": error})
                print(error, flush=True)
            state["elapsed_seconds"] = time.monotonic()-started
            save_json(args.root/"refinement_status.json", state)
    state["status"] = "complete" if not state["failed"] else "failed"
    save_json(args.root/"refinement_status.json", state)
    return int(bool(state["failed"]))


if __name__ == "__main__":
    raise SystemExit(main())
