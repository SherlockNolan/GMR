"""Render ten diverse flagged Q1 clips and every unflagged clip in full."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REVIEW_SELECTION = [
    "walk1_subject1", "run1_subject2", "sprint1_subject2", "dance2_subject2",
    "fallAndGetUp1_subject5", "ground2_subject2", "fight1_subject3",
    "jumps1_subject2", "obstacles4_subject3", "pushAndStumble1_subject3",
]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def render_one(entry, root, output, device):
    video = output / entry["video"]
    log = output / "render_logs" / (video.stem + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    temporary = video.with_name(video.stem + ".partial.mp4")
    environment = dict(os.environ, MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID=str(device),
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    script = Path(__file__).with_name("render_robot_motion.py")
    command = [sys.executable, str(script), "--robot", "q1", "--motion-file",
               str(root / "pkl" / (entry["clip"] + ".pkl")), "--output", str(temporary),
               "--fps", "30", "--label", f"{entry['group']} | {entry['clip']}"]
    if entry.get("model_file"):
        command.extend(["--model-file", entry["model_file"]])
    if entry.get("full", True):
        command.append("--full")
    else:
        command.extend(["--start-seconds", str(entry["start_seconds"]),
                        "--duration", str(entry["duration_seconds"])])
    started = time.monotonic()
    with log.open("w") as stream:
        subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=nb_frames,width,height,avg_frame_rate,duration",
                            "-of", "json", str(temporary)], capture_output=True, text=True, check=True)
    metadata = json.loads(probe.stdout)["streams"][0]
    if int(metadata["nb_frames"]) != entry["frames"]:
        raise ValueError(f"Wrong frame count for {entry['clip']}: {metadata}")
    temporary.replace(video)
    return {**entry, "status": "complete", "render_seconds": time.monotonic() - started,
            "bytes": video.stat().st_size, "video_metadata": metadata}


def write_index(output, entries):
    sections = []
    for group, title in (("review", "需要复查：选取 10 段"), ("unflagged", "未触发复查阈值：全部 4 段")):
        cards = []
        for entry in entries:
            if entry["group"] != group:
                continue
            name = html.escape(entry["clip"])
            video = html.escape(entry["video"], quote=True)
            poster = html.escape(Path(entry["video"]).with_suffix(".jpg").name, quote=True)
            cards.append(f'''<article><h3>{name}</h3>
<video controls preload="none" src="{video}" poster="{poster}"></video>
<p>时长 {entry['duration_seconds'] / 60:.2f} 分钟 · 30 FPS · 高度修正 {entry['ground_shift_m'] * 100:.1f} cm · 最大关节速度 {entry['max_joint_velocity_rad_s']:.1f} rad/s</p>
<a href="{video}" download>下载视频</a></article>''')
        sections.append(f"<section><h2>{title}</h2><div class='grid'>{''.join(cards)}</div></section>")
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Q1 动作复查预览</title>
<style>body{{margin:24px;background:#10151c;color:#e9edf2;font:16px system-ui,sans-serif}}h1{{font-size:26px}}h2{{font-size:21px;margin-top:32px}}h3{{font-size:16px;overflow-wrap:anywhere}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}}article{{background:#1c2530;border-radius:10px;padding:16px}}video{{width:100%;aspect-ratio:4/3;background:#000}}p{{line-height:1.6;color:#b7c4d2}}a{{color:#81baff}}</style>
<h1>Q1 重定向动作预览</h1><p>14 段完整动作，按原始 30 FPS 播放。未触发阈值不代表已验证动力学可执行性。这些视频回放的是 IK 姿态。</p>
{''.join(sections)}</html>'''
    (output / "index.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] / "smp/datasets/lafan1/q1_gmr")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "previews"
    output.mkdir(parents=True, exist_ok=True)
    quality = json.loads((root / "quality_summary.json").read_text())
    by_name = {entry["clip"]: entry for entry in quality["clips"]}
    names = REVIEW_SELECTION + [entry["clip"] for entry in quality["clips"] if not entry["requires_review"]]
    entries = []
    for name in names:
        report = by_name[name]
        group = "review" if report["requires_review"] else "unflagged"
        if name in REVIEW_SELECTION and group != "review":
            raise ValueError(f"Selected review clip is no longer flagged: {name}")
        entries.append({"clip": name, "group": group, "video": f"{group}_{name}.mp4",
                        "frames": report["frames"], "duration_seconds": report["frames"] / 30,
                        "ground_shift_m": report["ground_shift_m"],
                        "max_joint_velocity_rad_s": report["max_joint_velocity_rad_s"], "status": "pending"})
    status = {"status": "running", "total": len(entries), "completed": 0,
              "failed": 0, "fps": 30, "full_length": True,
              "selection_reason": "Ten flagged clips cover walking, running, sprinting, dance, getting up, crawling, fighting, jumping, obstacles and stumbling; all four unflagged clips are included.",
              "clips": entries}
    manifest = output / "selection_manifest.json"
    write_json(manifest, status)
    started = time.monotonic()
    print(f"START previews={len(entries)} full_motion_frames={sum(e['frames'] for e in entries)} workers={args.workers}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(render_one, entry, root, output, args.devices[i % len(args.devices)]): i
                   for i, entry in enumerate(entries)}
        for future in as_completed(futures):
            index = futures[future]
            try:
                entries[index] = future.result()
                status["completed"] += 1
                print(f"DONE {status['completed']}/{status['total']} {entries[index]['video']}", flush=True)
            except Exception as error:
                entries[index] = {**entries[index], "status": "failed", "error": str(error)}
                status["failed"] += 1
                print(f"FAILED {entries[index]['clip']}: {error}", flush=True)
            status["clips"] = entries
            status["elapsed_seconds"] = time.monotonic() - started
            write_json(manifest, status)
    status["status"] = "complete" if not status["failed"] else "failed"
    write_json(manifest, status)
    write_index(output, entries)
    print(f"FINISHED completed={status['completed']} failed={status['failed']} seconds={status['elapsed_seconds']:.1f}", flush=True)
    return bool(status["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
