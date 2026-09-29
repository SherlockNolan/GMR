"""Q1 contact extension for jumping, elevated support, and ground motions.

BVH supplies no environment mesh or force labels. Elevated support is inferred
and body support uses soft surface-height goals, allowing rolling/sliding.
All supplied robot collision primitives must remain above the ground.
"""

import re

import mink
import mujoco as mj
import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation as R

from .q1_contact import SIDES, Q1ContactRetargeter, ContactBounds


BODY_PATCHES = (
    ("left_hand", "LeftHand", "left_elbow_link", .10),
    ("right_hand", "RightHand", "right_elbow_link", .10),
    ("left_knee", "LeftLeg", "left_knee_link", .11),
    ("right_knee", "RightLeg", "right_knee_link", .11),
    ("pelvis", "Hips", "pelvis", .20),
    ("torso", "Spine2", "torso_link", .18),
    ("head", "Head", "head_link", .16),
)


def motion_family(name):
    return re.sub(r"\d+$", "", name.split("_")[0])


def phases(bits):
    starts = np.flatnonzero(bits & ~np.r_[False, bits[:-1]])
    ends = np.flatnonzero(bits & ~np.r_[bits[1:], False])+1
    return zip(starts, ends)


def hysteresis(height, speed, enter_h, exit_h, enter_v=.20, exit_v=.40):
    bits = np.zeros(len(height), dtype=bool)
    current = False
    for i in range(len(bits)):
        current = (height[i] < exit_h and speed[i] < exit_v) if current else (height[i] < enter_h and speed[i] < enter_v)
        bits[i] = current
    for start, end in list(phases(bits)):
        if end-start < 2:
            bits[start:end] = False
    return bits


def detect_contacts(frames, positions, mapping, family, offsets, fps=30):
    n = len(frames)
    foot_labels = np.zeros((n, 2, 2), dtype=bool)
    height = np.zeros((n, 2, 2))
    speed = np.zeros_like(height)
    levels = np.zeros_like(height)
    root_z = positions["Hips"][:, 2]
    standing_height = float(root_z[0])
    upright = root_z > .72*standing_height
    if np.count_nonzero(upright) < 30:
        upright[:min(n, 30)] = True
    baselines = np.zeros((2, 2))
    elevated_phases = 0
    for side_index, (side, human) in enumerate(SIDES):
        _, offset = offsets[f"{side}_ankle_roll_link"]
        quat = np.array([frame[human+"FootMod"][1] for frame in frames])
        normals = (R.from_quat(quat, scalar_first=True)*offset).as_matrix()[:, 2, 2]
        # Do not flatten a vertical or inverted foot during crawling/falling.
        valid_orientation = normals > .50
        for patch, bone in enumerate((human+"Foot", human+"Toe")):
            points = positions[bone]
            baseline = float(np.quantile(points[upright, 2], .02))
            z = median_filter(points[:, 2]-baseline, size=3, mode="nearest")
            horizontal = median_filter(np.linalg.norm(np.gradient(points, 1/fps, axis=0)[:, :2], axis=1), size=3, mode="nearest")
            vertical = median_filter(np.abs(np.gradient(points[:, 2], 1/fps)), size=3, mode="nearest")
            leg = mapping["source_lengths_m"][f"{side}_thigh"]+mapping["source_lengths_m"][f"{side}_shank"]
            bits = hysteresis(z, horizontal, .04*leg, .065*leg)
            bits &= valid_orientation
            if family == "jumps":
                # A vertical takeoff can have almost zero horizontal speed.
                # Release the support estimate before the height band is crossed.
                bits &= vertical < .55
            # Obstacles have no surface geometry in BVH. Only stable, elevated
            # marker plateaus are proposed as support; every such clip is reviewed.
            if family == "obstacles":
                stable = (horizontal < .16) & (vertical < .12) & (z > .08) & valid_orientation
                for start, end in list(phases(stable)):
                    if end-start >= 5:
                        bits[start:end] = True
                        levels[start:end, side_index, patch] = max(0., float(np.median(z[start:end])))*mapping["root_translation_scale"]
                        elevated_phases += 1
            foot_labels[:, side_index, patch] = bits
            height[:, side_index, patch], speed[:, side_index, patch] = z, horizontal
            baselines[side_index, patch] = baseline
    # Ground height in source coordinates is estimated from upright toe markers.
    floor = float(np.mean(baselines[:, 1]))
    low_body = root_z-floor < .68*(standing_height-floor)
    body_labels = np.zeros((n, len(BODY_PATCHES)), dtype=bool)
    body_heights = np.empty_like(body_labels, dtype=float)
    for index, (_, bone, _, threshold) in enumerate(BODY_PATCHES):
        points = positions[bone]
        z = median_filter(points[:, 2]-floor, size=3, mode="nearest")
        velocity = median_filter(np.linalg.norm(np.gradient(points, 1/fps, axis=0), axis=1), size=3, mode="nearest")
        labels = hysteresis(z, velocity, threshold, threshold+.05, .25, .50) & low_body
        body_labels[:, index], body_heights[:, index] = labels, z
    foot = foot_labels.any(axis=2)
    report = {"method": "upright-calibrated marker height/speed hysteresis + foot-normal gate; estimated elevated plateaus for obstacles",
              "marker_baselines_m": baselines.tolist(), "source_floor_proxy_m": floor,
              "heel_is_proxy": True, "flat_ground_only": family != "obstacles",
              "left_support_fraction": float(foot[:, 0].mean()), "right_support_fraction": float(foot[:, 1].mean()),
              "double_support_fraction": float(foot.all(axis=1).mean()), "flight_fraction": float((~foot.any(axis=1)).mean()),
              "no_foot_support_fraction": float((~foot.any(axis=1)).mean()),
              "flight_candidate_fraction": float((~foot.any(axis=1) & ~body_labels.any(axis=1) & ~low_body).mean()),
              "vertical_takeoff_gate_m_s": .55 if family == "jumps" else None,
              "left_support_phases": len(list(phases(foot[:, 0]))), "right_support_phases": len(list(phases(foot[:, 1]))),
              "low_body_fraction": float(low_body.mean()), "inferred_elevated_patch_phases": elevated_phases,
              "body_patch_order": [p[0] for p in BODY_PATCHES],
              "body_support_fraction": {p[0]: float(body_labels[:, i].mean()) for i, p in enumerate(BODY_PATCHES)},
              "limitations": ["Foot is an ankle-marker heel proxy.", "BVH has no force or obstacle geometry labels.",
                              "Non-foot contact is estimated from markers and simplified supplied collision boxes."]}
    return foot_labels, height, speed, report, {"levels": levels, "body_labels": body_labels, "body_heights": body_heights}


def surface_point(model, data, geom):
    """Exact lowest support point for supplied box/sphere/capsule primitives."""
    kind = model.geom_type[geom]
    rotation = data.geom_xmat[geom].reshape(3, 3)
    center = data.geom_xpos[geom]
    size = model.geom_size[geom]
    if kind == mj.mjtGeom.mjGEOM_BOX:
        return center-rotation @ (np.sign(rotation[2])*size)
    if kind == mj.mjtGeom.mjGEOM_SPHERE:
        return center-np.array([0., 0., size[0]])
    if kind == mj.mjtGeom.mjGEOM_CAPSULE:
        return center-rotation[:, 2]*np.sign(rotation[2, 2])*size[1]-np.array([0., 0., size[0]])
    raise ValueError(f"Unsupported collision primitive {kind}")


class SurfaceHeight(mink.Task):
    """Soft body-surface height target; rolling and horizontal motion remain free."""
    def __init__(self, model, geom, cost=100.):
        super().__init__(np.array([cost]), lm_damping=.01)
        self.model, self.geom = model, geom
        self.jac = np.empty((3, model.nv))

    def compute_error(self, configuration):
        return np.array([surface_point(self.model, configuration.data, self.geom)[2]])

    def compute_jacobian(self, configuration):
        point = surface_point(self.model, configuration.data, self.geom)
        mj.mj_jac(self.model, configuration.data, self.jac, None, point, int(self.model.geom_bodyid[self.geom]))
        return self.jac[2:3].copy()


class WholeBodyBounds(ContactBounds):
    def __init__(self, model, sites, radii):
        super().__init__(model, sites, radii)
        self.geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] and (model.geom_contype[i] or model.geom_conaffinity[i])]
        self.surface_levels = {}

    def compute_qp_inequalities(self, configuration, dt):
        foot = super().compute_qp_inequalities(configuration, dt)
        rows, bounds = [foot.G], [foot.h]
        data = configuration.data
        for geom in self.geoms:
            point = surface_point(self.model, data, geom)
            mj.mj_jac(self.model, data, self.jac, None, point, int(self.model.geom_bodyid[geom]))
            level = self.surface_levels.get(int(self.model.geom_bodyid[geom]), 0.)
            rows.append(-self.jac[2:3].copy())
            bounds.append(np.array([point[2]-level+self.floor_tolerance]))
        return mink.Constraint(G=np.vstack(rows), h=np.concatenate(bounds))


class Q1MultiContactRetargeter(Q1ContactRetargeter):
    def __init__(self, metadata, mapping, positions, fps=30, body_contact_cost=100.):
        super().__init__(metadata, mapping, positions, fps)
        self.contacts = WholeBodyBounds(self.model, sum(self.sites, []), [.008]*8)
        self.base_stage1, self.base_stage2 = self.stage1.copy(), self.stage2.copy()
        self.body_geoms = []
        self.body_tasks = []
        self.body_contact_cost = body_contact_cost
        for _, _, body, _ in BODY_PATCHES:
            geom = next(i for i in self.contacts.geoms if self.model.geom_bodyid[i] == self.model.body(body).id)
            self.body_geoms.append(geom)
            self.body_tasks.append(SurfaceHeight(self.model, geom, body_contact_cost))
        self.levels = np.zeros((2, 2))

    def update_contacts(self, labels):
        previous = set(self.selected)
        super().update_contacts(labels)
        self.contacts.surface_levels = {}
        for side, (name, _) in enumerate(SIDES):
            keys = [key for key in self.selected if key[0] == side]
            if not keys:
                continue
            existing = [key for key in keys if key in previous]
            if existing:
                plane = self.contacts.anchors[self.selected[existing[0]]][2]-.008
            else:
                # One rigid foot uses one plane for the entire support phase.
                # Prefer a ground estimate over an uncertain elevated ankle proxy.
                plane = min(float(self.levels[key]) for key in keys)
            for key in keys:
                self.contacts.anchors[self.selected[key]][2] = .008+plane
            self.contacts.surface_levels[self.model.body(f"{name}_ankle_roll_link").id] = plane

    def residuals(self):
        _, error = super().residuals()
        data = self.configuration.data
        bottom = [surface_point(self.model, data, geom)[2]-self.contacts.surface_levels.get(int(self.model.geom_bodyid[geom]), 0.)
                  for geom in self.contacts.geoms]
        return max(0., -min(bottom)), error

    def retarget(self, frame, labels, first=False, levels=None, body_labels=None):
        self.levels = np.zeros((2, 2)) if levels is None else levels
        if body_labels is None:
            body_labels = np.zeros(len(self.body_tasks), dtype=bool)
        active = [task for task, supported in zip(self.body_tasks, body_labels) if supported]
        self.stage1, self.stage2 = self.base_stage1+active, self.base_stage2+active
        q, result = super().retarget(frame, labels, first)
        data = self.configuration.data
        result["body_bottom_z"] = np.array([surface_point(self.model, data, geom)[2] for geom in self.body_geoms])
        result["collision_bottom_z"] = np.array([surface_point(self.model, data, geom)[2] for geom in self.contacts.geoms])
        result["foot_penetration"] = max(0., .008-min(data.site_xpos[site, 2] for site in sum(self.sites, [])))
        return q, result
