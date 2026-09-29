"""Render the audited representative interval of every expanded Q1 sequence."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from render_q1_review_previews import render_one, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/zy/code/smp/datasets/lafan1/q1_gmr_v3"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--unchanged-only", action="store_true", help="Render clips outside the current refinement selection")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    summary = json.loads((args.root/"quality_summary.json").read_text())
    output = args.root/"previews"
    jobs = []
    refining = {p.stem for p in (args.root/"reports_initial").glob("*.json")}
    manifest_path = output/"render_manifest.json"
    old_entries = {}
    if manifest_path.exists():
        old_entries = {e["clip"]: e for e in json.loads(manifest_path.read_text())["clips"] if e.get("status") == "complete"}
    cached = []
    for entry in summary["clips"]:
        if args.unchanged_only and entry["clip"] in refining:
            continue
        report = json.loads((args.root/"reports"/(entry["clip"]+".json")).read_text())
        job = {"clip": entry["clip"], "group": f"Q1 v3 | {entry['family']}", "video": entry["clip"]+".mp4",
                     "frames": entry["preview_frames"], "duration_seconds": entry["preview_frames"]/30,
                     "start_seconds": entry["preview_start_seconds"], "full": False, "model_file": report["model"],
                     "report_fingerprint": report["fingerprint"]}
        previous = old_entries.get(entry["clip"])
        if args.skip_existing and previous and previous.get("report_fingerprint") == report["fingerprint"] and previous["start_seconds"] == job["start_seconds"] and (output/job["video"]).exists():
            cached.append(previous)
        else:
            jobs.append(job)
    total = len(jobs)+len(cached)
    state = {"status": "running", "total": total, "completed": len(cached), "failed": 0, "clips": cached, "partial_selection": args.unchanged_only}
    write_json(output/"render_manifest.json", state)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(render_one, entry, args.root, output, i%2): entry for i, entry in enumerate(jobs)}
        for future in as_completed(futures):
            try:
                state["clips"].append(future.result())
                state["completed"] += 1
                print(f"DONE {state['completed']}/{total} {futures[future]['clip']}", flush=True)
            except Exception as error:
                state["failed"] += 1
                state["clips"].append({**futures[future], "status": "failed", "error": str(error)})
                print(str(error), flush=True)
            write_json(output/"render_manifest.json", state)
    state["status"] = "complete" if not state["failed"] else "failed"
    write_json(output/"render_manifest.json", state)
    return int(bool(state["failed"]))


if __name__ == "__main__":
    raise SystemExit(main())
