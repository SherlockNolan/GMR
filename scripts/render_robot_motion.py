"""Render a saved GMR motion to video on a headless server."""

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from general_motion_retargeting.data_loader import load_robot_motion
from general_motion_retargeting.params import ROBOT_XML_DICT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="q1", choices=ROBOT_XML_DICT)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, help="Override the registered robot asset (e.g. contact v2)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seconds", type=float, default=0)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--full", action="store_true", help="Render the full remaining motion")
    parser.add_argument("--label", default="", help="Display a clip/review label")
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    if args.start_seconds < 0 or args.duration <= 0 or args.fps < 1:
        parser.error("start-seconds must be nonnegative; duration and fps must be positive")
    motion, fps, root_pos, root_rot, joints, _, _ = load_robot_motion(args.motion_file)
    model = mj.MjModel.from_xml_path(str(args.model_file or ROBOT_XML_DICT[args.robot]))
    if joints.shape[1] != model.nq - 7:
        raise ValueError("Motion joint dimension does not match robot")
    if motion.get("joint_names"):
        names = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] == mj.mjtJoint.mjJNT_HINGE]
        if names != motion["joint_names"]:
            raise ValueError("Motion joint names/order do not match robot")
    end_seconds = len(joints) / fps if args.full else min(len(joints) / fps,
                                                       args.start_seconds + args.duration)
    video_frames = int(np.ceil((end_seconds - args.start_seconds) * args.fps - 1e-8))
    indexes = np.floor((args.start_seconds + np.arange(max(0, video_frames)) / args.fps)
                       * fps + 1e-8).astype(int)
    indexes = np.minimum(indexes, len(joints) - 1)
    if not len(indexes):
        raise ValueError("Requested preview starts after the motion ends")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = mj.MjData(model)
    camera = mj.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 1.7, 135, -15
    font = ImageFont.load_default(size=16)
    with mj.Renderer(model, height=480, width=640) as renderer:
        with imageio.get_writer(str(args.output), fps=args.fps, codec="libx264",
                               ffmpeg_params=["-threads", "2"]) as writer:
            for index in indexes:
                data.qpos[:3], data.qpos[3:7], data.qpos[7:] = root_pos[index], root_rot[index], joints[index]
                mj.mj_forward(model, data)
                camera.lookat[:] = [root_pos[index, 0], root_pos[index, 1], max(.35, root_pos[index, 2]) + .1]
                renderer.update_scene(data, camera=camera)
                if "contact_plane_height_m" in motion:
                    # Visualize only inferred support patches, not a recovered
                    # obstacle scene. Orange pads are explicitly labelled below.
                    levels = motion["contact_plane_height_m"][index]
                    for side, name in enumerate(("left", "right")):
                        if not motion["contact_labels"][index, side].any() or levels[side].max() < .01:
                            continue
                        foot = data.xpos[model.body(f"{name}_ankle_roll_link").id]
                        geom = renderer.scene.geoms[renderer.scene.ngeom]
                        mj.mjv_initGeom(geom, mj.mjtGeom.mjGEOM_BOX,
                            np.array([.12, .07, .006]), np.array([foot[0], foot[1], levels[side].max()-.006]),
                            np.eye(3).ravel(), np.array([1., .55, .10, .65], dtype=np.float32))
                        renderer.scene.ngeom += 1
                frame = renderer.render()
                if args.label:
                    panel = Image.fromarray(frame)
                    draw = ImageDraw.Draw(panel, "RGBA")
                    panel_height = 103 if "body_contact_labels" in motion else (80 if "contact_labels" in motion else 57)
                    draw.rectangle((0, 0, 640, panel_height), fill=(0, 0, 0, 170))
                    draw.text((9, 6), args.label, font=font, fill=(255, 255, 255, 255))
                    draw.text((9, 30), f"t={index / fps:.2f}s  frame={index + 1}/{len(joints)}",
                              font=font, fill=(220, 220, 220, 255))
                    if "contact_labels" in motion:
                        contact = motion["contact_labels"][index].any(axis=1)
                        draw.text((9, 53), f"Estimated support: L={'ON' if contact[0] else 'OFF'} R={'ON' if contact[1] else 'OFF'}",
                                  font=font, fill=(180, 255, 190, 255))
                    if "body_contact_labels" in motion:
                        patches = motion["metadata"]["body_patch_order"]
                        active = [name for name, bit in zip(patches, motion["body_contact_labels"][index]) if bit]
                        note = "Body: " + (", ".join(active) if active else "none")
                        if motion.get("contact_recovery_flags", np.zeros(len(joints), dtype=bool))[index]:
                            note = "REVIEW: recovered/deferred contact | " + note
                        if motion["metadata"].get("needs_environment_geometry"):
                            note = "Orange pads = inferred support; terrain unavailable"
                        draw.text((9, 76), note, font=font, fill=(255, 210, 145, 255))
                    frame = np.asarray(panel)
                writer.append_data(frame)
    print(f"Saved {args.output}: {len(indexes)} frames, {args.fps} FPS")


if __name__ == "__main__":
    main()
