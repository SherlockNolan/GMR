"""Render all sixteen contact-v2 excerpts plus walk/run comparisons."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess

from render_q1_review_previews import render_one, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/zy/code/smp/datasets/lafan1/q1_gmr_v2"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.root
    output = root/"previews"
    output.mkdir(exist_ok=True)
    reports = [json.loads(p.read_text()) for p in sorted((root/"reports").glob("*.json"))]
    jobs = []
    for r in reports:
        entry = {"clip": r["clip"], "group": "Q1 contact v2", "video": r["clip"]+"_12s_32s.mp4",
                 "frames": 600, "duration_seconds": 20, "start_seconds": 12, "full": False, "model_file": r["model"]}
        jobs.append((entry, root))
    for clip, start in (("run1_subject2", 219), ("walk1_subject5", 111), ("walk2_subject3", 183)):
        r = next(r for r in reports if r["clip"] == clip)
        frames = min(600, r["frames"]-start*30)
        jobs.append(({"clip": clip, "group": "Q1 contact v2 - transition review", "video": f"{clip}_{start}s_{start+20}s.mp4",
                     "frames": frames, "duration_seconds": 20, "start_seconds": start, "full": False, "model_file": r["model"]}, root))
    for clip in ("walk1_subject1", "run1_subject2"):
        jobs.append(({"clip": clip, "group": "Q1 original v1", "video": f"v1_{clip}_12s_32s.mp4",
                     "frames": 600, "duration_seconds": 20, "start_seconds": 12, "full": False}, root.parent/"q1_gmr"))
    state = {"status": "running", "total": len(jobs), "completed": 0, "failed": 0, "clips": []}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(render_one, entry, input_root, output, index%2): entry for index, (entry, input_root) in enumerate(jobs)}
        for future in as_completed(futures):
            try:
                entry = future.result()
                state["completed"] += 1
                state["clips"].append(entry)
                print(f"DONE {state['completed']}/{len(jobs)} {entry['video']}", flush=True)
            except Exception as error:
                state["failed"] += 1
                state["clips"].append({**futures[future], "status": "failed", "error": str(error)})
                print(str(error), flush=True)
            write_json(output/"render_manifest.json", state)
    if state["failed"]:
        state["status"] = "failed"
        write_json(output/"render_manifest.json", state)
        return 1
    comparisons = []
    for clip in ("walk1_subject1", "run1_subject2"):
        video = output/f"compare_{clip}_12s_32s.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "2",
                        "-i", str(output/f"v1_{clip}_12s_32s.mp4"), "-i", str(output/f"{clip}_12s_32s.mp4"),
                        "-filter_complex", "[0:v][1:v]hstack=inputs=2", "-filter_complex_threads", "1",
                        "-c:v", "libx264", "-threads", "2", "-crf", "22", str(video)], check=True)
        comparisons.append(f'<h2>{clip}：左 v1，右接触 v2</h2><video controls preload="metadata" src="{video.name}"></video>')
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>Q1 v1/v2 comparison</title><style>body{max-width:1300px;margin:24px auto;font:16px sans-serif}video{width:100%}</style><h1>Q1 walk/run 重定向对比</h1><p>同一源动作，同一帧率和截取区间；运动学回放。右侧显示人体估计的支撑标签。</p>'+''.join(comparisons)+'</html>'
    (output/"comparisons.html").write_text(page)
    state["status"] = "complete"
    state["comparisons"] = [f"compare_{clip}_12s_32s.mp4" for clip in ("walk1_subject1", "run1_subject2")]
    write_json(output/"render_manifest.json", state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
