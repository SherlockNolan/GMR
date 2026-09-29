"""Segment-aware Q1 IK with estimated human support phases and fixed contacts.

This extends the kinematic, two-stage IK approach used by GMR. It does not
establish dynamic balance or supply ground-truth contact labels.
"""

import json
from pathlib import Path

import mink
import mujoco as mj
import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation as R


SIDES = (("left", "Left"), ("right", "Right"))


def measure_human(frames, geometry):
    positions = {bone: np.array([frame[bone][0] for frame in frames]) for bone in frames[0]}
    pairs = {"pelvis_to_torso": ("Hips", "Spine2"), "torso_to_head": ("Spine2", "Head")}
    for side, human in SIDES:
        pairs.update({f"{side}_thigh": (human+"UpLeg", human+"Leg"),
                      f"{side}_shank": (human+"Leg", human+"Foot"),
                      f"{side}_upperarm": (human+"Arm", human+"ForeArm"),
                      f"{side}_forearm": (human+"ForeArm", human+"Hand"),
                      f"{side}_foot_to_toe": (human+"Foot", human+"Toe")})
    lengths = {key: float(np.median(np.linalg.norm(positions[b]-positions[a], axis=1)))
               for key, (a, b) in pairs.items()}
    ratios = {key: geometry["lengths_m"][key]/lengths[key] for key in geometry["lengths_m"]}
    root_scale = sum(geometry["lengths_m"][f"{side}_{part}"] for side, _ in SIDES for part in ("thigh", "shank")) / sum(
        lengths[f"{side}_{part}"] for side, _ in SIDES for part in ("thigh", "shank"))
    return positions, {"source_lengths_m": lengths, "robot_lengths_m": geometry["lengths_m"],
                       "segment_ratios": ratios, "root_translation_scale": root_scale,
                       "source_landmarks": {k: list(v) for k, v in pairs.items()},
                       "definition": "median FK landmark distances over this clip; separate segment lengths, not global arm/leg constants"}


def detect_support(positions, mapping, fps=30):
    n = len(positions["Hips"])
    labels = np.zeros((n, 2, 2), dtype=bool)
    heights = np.zeros((n, 2, 2))
    speeds = np.zeros_like(heights)
    baselines = np.zeros((2, 2))
    parameters = []
    for side_index, (side, human) in enumerate(SIDES):
        leg = mapping["source_lengths_m"][f"{side}_thigh"] + mapping["source_lengths_m"][f"{side}_shank"]
        enter_h, exit_h = .04*leg, .065*leg
        enter_v, exit_v = .20*leg/.86, .40*leg/.86
        parameters.append({"side": side, "enter_height_m": enter_h, "exit_height_m": exit_h,
                           "enter_horizontal_speed_m_s": enter_v, "exit_horizontal_speed_m_s": exit_v})
        for patch, bone in enumerate((human+"Foot", human+"Toe")):
            points = positions[bone]
            baseline = float(np.quantile(points[:, 2], .02))
            height = median_filter(points[:, 2]-baseline, size=3, mode="nearest")
            velocity = np.linalg.norm(np.gradient(points, 1/fps, axis=0)[:, :2], axis=1)
            velocity = median_filter(velocity, size=3, mode="nearest")
            current = False
            for i in range(n):
                if current:
                    current = height[i] < exit_h and velocity[i] < exit_v
                else:
                    current = height[i] < enter_h and velocity[i] < enter_v
                labels[i, side_index, patch] = current
            # Only remove isolated contact pulses; keep short running flight phases.
            bits = labels[:, side_index, patch]
            singleton = bits[1:-1] & ~bits[:-2] & ~bits[2:]
            bits[np.flatnonzero(singleton)+1] = False
            heights[:, side_index, patch], speeds[:, side_index, patch] = height, velocity
            baselines[side_index, patch] = baseline
    foot = labels.any(axis=2)
    report = {"method": "Foot ankle-marker heel proxy + Toe; per-marker 2% height baseline; horizontal speed/height hysteresis; 3-frame median",
              "parameters": parameters, "marker_baselines_m": baselines.tolist(),
              "heel_is_proxy": True, "flat_ground_only": True,
              "left_support_fraction": float(foot[:, 0].mean()), "right_support_fraction": float(foot[:, 1].mean()),
              "double_support_fraction": float(foot.all(axis=1).mean()), "flight_fraction": float((~foot.any(axis=1)).mean()),
              "left_support_phases": int(np.sum(foot[:, 0] & ~np.r_[False, foot[:-1, 0]])),
              "right_support_phases": int(np.sum(foot[:, 1] & ~np.r_[False, foot[:-1, 1]]))}
    return labels, heights, speeds, report


class WorldPoint(mink.Task):
    def __init__(self, model, name, cost, site=False):
        super().__init__(np.broadcast_to(cost, (3,)).copy(), lm_damping=.01)
        self.site = site
        self.index = model.site(name).id if site else model.body(name).id
        self.target = np.zeros(3)
        self.jac = np.empty((3, model.nv))

    def compute_error(self, configuration):
        data = configuration.data
        return (data.site_xpos if self.site else data.xpos)[self.index] - self.target

    def compute_jacobian(self, configuration):
        fn = mj.mj_jacSite if self.site else mj.mj_jacBody
        fn(configuration.model, configuration.data, self.jac, None, self.index)
        return self.jac


class FrameBounds(mink.Limit):
    """Bounds remain anchored to the previous OUTPUT frame during all IK iterations."""
    def __init__(self, model, metadata, fps):
        hinges = [i for i in range(model.njnt) if model.jnt_type[i] == mj.mjtJoint.mjJNT_HINGE]
        self.qids = model.jnt_qposadr[hinges]
        self.projection = np.eye(model.nv)[model.jnt_dofadr[hinges]]
        self.ranges = model.jnt_range[hinges]
        self.vmax = np.array([metadata["urdf_joint_limits"][model.joint(i).name]["velocity"] for i in hinges])
        self.fps = fps
        self.previous = None
        self.trust = np.r_[np.full(3, .06), np.full(3, .20), np.full(model.nv-6, .25)]
        self.eye = np.eye(model.nv)

    def compute_qp_inequalities(self, configuration, dt):
        low, high = self.ranges[:, 0].copy(), self.ranges[:, 1].copy()
        if self.previous is not None:
            low = np.maximum(low, self.previous-self.vmax/self.fps)
            high = np.minimum(high, self.previous+self.vmax/self.fps)
        current = configuration.q[self.qids]
        return mink.Constraint(G=np.vstack((self.projection, -self.projection, self.eye, -self.eye)),
                               h=np.r_[high-current, current-low, self.trust, self.trust])


class ContactBounds(mink.Limit):
    def __init__(self, model, sites, radii):
        self.model, self.sites, self.radii = model, sites, radii
        self.anchors = {}
        self.tolerance = .0004
        self.floor_tolerance = .00005
        self.jac = np.empty((3, model.nv))

    def compute_qp_inequalities(self, configuration, dt):
        rows, bounds = [], []
        for site, radius in zip(self.sites, self.radii):
            mj.mj_jacSite(self.model, configuration.data, self.jac, None, site)
            rows.append(-self.jac[2].copy())
            bounds.append(configuration.data.site_xpos[site, 2]-radius+self.floor_tolerance)
            if site in self.anchors:
                delta = self.anchors[site]-configuration.data.site_xpos[site]
                rows.extend(self.jac.copy())
                bounds.extend(delta+self.tolerance)
                rows.extend(-self.jac.copy())
                bounds.extend(-delta+self.tolerance)
        return mink.Constraint(G=np.array(rows), h=np.array(bounds))


class Q1ContactRetargeter:
    def __init__(self, metadata, mapping, positions, fps=30):
        self.metadata, self.mapping, self.fps = metadata, mapping, fps
        self.model = mj.MjModel.from_xml_path(metadata["model"])
        self.configuration = mink.Configuration(self.model)
        self.configuration.update(np.array(metadata["reference_qpos"]))
        old_cfg = json.loads((Path(__file__).parent/"ik_configs/bvh_lafan1_to_q1.json").read_text())
        self.offsets = {body: (human, R.from_quat(item[4], scalar_first=True))
                        for body, item in old_cfg["ik_match_table2"].items() for human in [item[0]]}
        self.point_tasks = {}
        self.orientation_tasks = {}
        self.stage1, self.stage2 = [], []
        def point(body, cost, first=False):
            task = WorldPoint(self.model, body, cost)
            self.point_tasks[body] = task
            self.stage2.append(task)
            if first:
                self.stage1.append(task)
        point("pelvis", [4, 4, 1], True)
        point("torso_link", 2)
        for side, _ in SIDES:
            point(f"{side}_hip_pitch_link", 1)
            point(f"{side}_knee_link", 4)
            point(f"{side}_ankle_pitch_link", 5, True)
            point(f"{side}_shoulder_roll_link", 1)
            point(f"{side}_elbow_link", 3)
            point(f"{side}_hand_mocap", 3, True)
        for body in self.offsets:
            if "hand" in body:
                continue
            cost = 5 if body in ("pelvis", "torso_link") else 2
            task = mink.FrameTask(body, "body", position_cost=0, orientation_cost=cost, lm_damping=.01)
            self.orientation_tasks[body] = task
            self.stage1.append(task)
            self.stage2.append(task)
        self.posture = mink.PostureTask(self.model, cost=1.5)
        self.stage1.append(self.posture)
        self.stage2.append(self.posture)
        self.frame_bounds = FrameBounds(self.model, metadata, fps)
        self.sites = [[self.model.site(f"{side}_contact_{i}").id for i in range(4)] for side, _ in SIDES]
        self.radius = .008
        self.contacts = ContactBounds(self.model, sum(self.sites, []), [self.radius]*8)
        self.contact_points = {site: WorldPoint(self.model, site, 40., site=True) for site in sum(self.sites, [])}
        self.polish_posture = mink.PostureTask(self.model, cost=.2)
        self.selected = {}
        self.anchor_origin = {}
        self.z_offset = metadata["reference_root_height_m"]-positions["Hips"][0, 2]*mapping["root_translation_scale"]
        self.previous = None
        self.relaxed_frames = 0

    def targets(self, frame):
        rotations = {body: R.from_quat(frame[human][1], scalar_first=True)*offset
                     for body, (human, offset) in self.offsets.items()}
        root = frame["Hips"][0]*self.mapping["root_translation_scale"]
        root = root.copy()
        root[2] += self.z_offset
        desired = {"pelvis": root}
        reference = self.metadata["body_positions_relative_pelvis_m"]
        lengths = self.metadata["lengths_m"]
        def vector(a, b, length):
            delta = frame[b][0]-frame[a][0]
            return delta/max(np.linalg.norm(delta), 1e-9)*length
        desired["torso_link"] = root+vector("Hips", "Spine2", lengths["pelvis_to_torso"])
        for side, human in SIDES:
            hip = f"{side}_hip_pitch_link"
            desired[hip] = root+rotations["pelvis"].apply(reference[hip])
            knee, ankle = f"{side}_knee_link", f"{side}_ankle_pitch_link"
            desired[knee] = desired[hip]+vector(human+"UpLeg", human+"Leg", lengths[f"{side}_thigh"])
            desired[ankle] = desired[knee]+vector(human+"Leg", human+"Foot", lengths[f"{side}_shank"])
            shoulder, elbow, hand = (f"{side}_{part}" for part in ("shoulder_roll_link", "elbow_link", "hand_mocap"))
            offset = np.array(reference[shoulder])-reference["torso_link"]
            desired[shoulder] = desired["torso_link"]+rotations["torso_link"].apply(offset)
            desired[elbow] = desired[shoulder]+vector(human+"Arm", human+"ForeArm", lengths[f"{side}_upperarm"])
            desired[hand] = desired[elbow]+vector(human+"ForeArm", human+"Hand", lengths[f"{side}_forearm"])
        for body, task in self.point_tasks.items():
            task.target = desired[body]
        for body, task in self.orientation_tasks.items():
            task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rotations[body].as_quat(scalar_first=True)), np.zeros(3)))
        return desired, rotations

    def update_contacts(self, labels):
        for key in list(self.selected):
            if not labels[key]:
                self.contacts.anchors.pop(self.selected.pop(key))
                self.anchor_origin.pop(key)
        data = self.configuration.data
        for side in range(2):
            for patch in range(2):
                key = (side, patch)
                if not labels[key] or key in self.selected:
                    continue
                candidates = self.sites[side][patch*2:patch*2+2]
                site = min(candidates, key=lambda i: data.site_xpos[i, 2])
                anchor = data.site_xpos[site].copy()
                # Freeze touchdown XY, constrain the selected sphere centre to the floor.
                # When the other patch already supports, create a consistent flat-foot anchor.
                other = self.selected.get((side, 1-patch))
                if other is not None:
                    rotation = data.xmat[self.model.body(f"{SIDES[side][0]}_ankle_roll_link").id].reshape(3, 3)
                    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
                    offset = R.from_euler("z", yaw).apply(self.model.site_pos[site]-self.model.site_pos[other])
                    anchor = self.contacts.anchors[other]+offset
                anchor[2] = self.radius
                self.selected[key] = site
                self.contacts.anchors[site] = anchor
                self.anchor_origin[key] = anchor.copy()

    def residuals(self):
        data = self.configuration.data
        penetration = max(0., self.radius-min(data.site_xpos[site, 2] for site in sum(self.sites, [])))
        errors = [np.max(np.abs(data.site_xpos[site]-target)) for site, target in self.contacts.anchors.items()]
        return penetration, max(errors, default=0.)

    def retarget(self, frame, labels, first=False):
        desired, rotations = self.targets(frame)
        if first:
            q = self.configuration.q.copy()
            q[:3] = desired["pelvis"]
            q[3:7] = rotations["pelvis"].as_quat(scalar_first=True)
            self.configuration.update(q)
        self.posture.set_target(self.configuration.q.copy())
        self.frame_bounds.previous = self.previous
        self.update_contacts(labels)
        self.contacts.tolerance = .0004
        relaxed = False
        iteration_count = 0
        for tasks, count in ((self.stage1, 4 if not first else 10), (self.stage2, 5 if not first else 14)):
            for _ in range(count):
                try:
                    velocity = mink.solve_ik(self.configuration, tasks, 1/self.fps, "daqp", damping=.005,
                                             limits=[self.frame_bounds, self.contacts])
                except mink.NoSolutionFound:
                    # Preserve temporal/joint/floor bounds; explicitly mark relaxed contact frames.
                    self.contacts.tolerance = max(.002, self.contacts.tolerance*2)
                    if self.contacts.tolerance > .04:
                        raise
                    continue
                self.configuration.integrate_inplace(velocity, 1/self.fps)
                # Two points on a rigid foot cannot exactly reach a newly flattened
                # target in one linearization. Tighten a continuation relaxation
                # after each step, rather than retaining it for the output frame.
                self.contacts.tolerance = max(.0004, self.contacts.tolerance/2)
                iteration_count += 1
                penetration, contact_error = self.residuals()
                if np.linalg.norm(velocity)/self.fps < 1e-4 and penetration < .0001 and contact_error < self.contacts.tolerance+.0001:
                    break
        # Re-linearize contact and floor constraints until geometric residuals settle.
        self.polish_posture.set_target(self.configuration.q.copy())
        polish_tasks = [self.polish_posture]
        for site, target in self.contacts.anchors.items():
            self.contact_points[site].target = target
            polish_tasks.append(self.contact_points[site])
        old_trust = self.frame_bounds.trust.copy()
        self.frame_bounds.trust = np.minimum(old_trust, .06)
        for _ in range(24):
            penetration, contact_error = self.residuals()
            if penetration < .00012 and contact_error < .00052 and self.contacts.tolerance <= .0004:
                break
            try:
                velocity = mink.solve_ik(self.configuration, polish_tasks, 1/self.fps, "daqp", damping=.01,
                                         limits=[self.frame_bounds, self.contacts])
            except mink.NoSolutionFound:
                self.contacts.tolerance = max(.002, self.contacts.tolerance*2)
                continue
            self.configuration.integrate_inplace(velocity, 1/self.fps)
            self.contacts.tolerance = max(.0004, self.contacts.tolerance/2)
            iteration_count += 1
        self.frame_bounds.trust = old_trust
        q = self.configuration.q.copy()
        self.previous = q[self.frame_bounds.qids].copy()
        relaxed = self.contacts.tolerance > .0004 or self.residuals()[1] > .0006
        self.relaxed_frames += int(relaxed)
        anchors = np.full((2, 2, 3), np.nan)
        actual = anchors.copy()
        site_ids = np.full((2, 2), -1, dtype=int)
        for key, site in self.selected.items():
            anchors[key] = self.contacts.anchors[site]
            actual[key] = self.configuration.data.site_xpos[site]
            site_ids[key] = site
        return q, {"anchors": anchors, "actual": actual, "site_ids": site_ids,
                   "relaxed": relaxed, "tolerance": self.contacts.tolerance,
                   "iterations": iteration_count, "pelvis_adjustment": q[:3]-desired["pelvis"],
                   "penetration": self.residuals()[0], "contact_error": self.residuals()[1]}
