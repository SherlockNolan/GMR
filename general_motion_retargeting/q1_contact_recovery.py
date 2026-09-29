"""Recorded recovery for infeasible estimated contacts; never certify them."""

import mink
import numpy as np

from .q1_multi_contact import Q1MultiContactRetargeter, WholeBodyBounds, surface_point


class RecoverableBounds(WholeBodyBounds):
    def __init__(self, model, sites, radii):
        super().__init__(model, sites, radii)
        self.soft_sites = set()

    def compute_qp_inequalities(self, configuration, dt):
        original = self.anchors
        self.anchors = {site: value for site, value in original.items() if site not in self.soft_sites}
        try:
            return super().compute_qp_inequalities(configuration, dt)
        finally:
            self.anchors = original


class Q1RecoverableRetargeter(Q1MultiContactRetargeter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contacts = RecoverableBounds(self.model, sum(self.sites, []), [.008]*8)
        self.recovered_frames = 0
        self.frozen_frames = 0
        self.terrain_soft = False

    def update_contacts(self, labels):
        super().update_contacts(labels)
        self.contacts.soft_sites.intersection_update(self.contacts.anchors)
        if self.terrain_soft:
            self.contacts.soft_sites.update(site for site, target in self.contacts.anchors.items() if target[2] > .018)
        for site in list(self.contacts.soft_sites):
            if not self.terrain_soft and np.max(np.abs(self.configuration.data.site_xpos[site]-self.contacts.anchors[site])) < .0004:
                self.contacts.soft_sites.remove(site)
        # A BVH-inferred plane is not supplied obstacle collision geometry.
        # If its support target is deferred/soft, only the actual world floor
        # remains a hard collision constraint for that foot.
        for key, site in self.selected.items():
            if site in self.contacts.soft_sites:
                body = self.model.site_bodyid[site]
                self.contacts.surface_levels[int(body)] = 0.
                task = self.contact_points[site]
                task.target = self.contacts.anchors[site]
                self.stage1.append(task)
                self.stage2.append(task)

    def fallback_result(self, q, desired):
        data = self.configuration.data
        anchors = np.full((2, 2, 3), np.nan)
        actual = anchors.copy()
        ids = np.full((2, 2), -1, dtype=int)
        for key, site in self.selected.items():
            anchors[key], actual[key], ids[key] = self.contacts.anchors[site], data.site_xpos[site], site
        self.previous = q[self.frame_bounds.qids].copy()
        return {"anchors": anchors, "actual": actual, "site_ids": ids, "relaxed": True,
                "tolerance": self.contacts.tolerance, "iterations": 0, "pelvis_adjustment": q[:3]-desired["pelvis"],
                "penetration": self.residuals()[0], "contact_error": self.residuals()[1],
                "body_bottom_z": np.array([surface_point(self.model, data, geom)[2] for geom in self.body_geoms]),
                "collision_bottom_z": np.array([surface_point(self.model, data, geom)[2] for geom in self.contacts.geoms]),
                "foot_penetration": max(0., .008-min(data.site_xpos[site, 2] for site in sum(self.sites, [])))}

    def retarget(self, frame, labels, first=False, levels=None, body_labels=None):
        before = self.configuration.q.copy()
        old_sites = set(self.contacts.anchors)
        old_trust = self.frame_bounds.trust.copy()
        recovered = frozen = False
        try:
            q, result = super().retarget(frame, labels, first, levels, body_labels)
        except mink.NoSolutionFound:
            recovered = True
            self.recovered_frames += 1
            self.configuration.update(before)
            new_sites = set(self.contacts.anchors)-old_sites
            self.contacts.soft_sites.update(new_sites or set(self.contacts.anchors))
            self.frame_bounds.trust = np.r_[np.full(3, .02), np.full(3, .06), np.full(self.model.nv-6, .08)]
            try:
                q, result = super().retarget(frame, labels, first, levels, body_labels)
            except mink.NoSolutionFound:
                # Reuse the preceding feasible pose and report the rejected frame.
                # All source frames stay on their original timeline.
                self.configuration.update(before)
                q = before.copy()
                desired, _ = self.targets(frame)
                result = self.fallback_result(q, desired)
                frozen = True
                self.frozen_frames += 1
        finally:
            self.frame_bounds.trust = old_trust
        result["solver_recovery"] = recovered
        result["frozen"] = frozen
        result["soft_foot_contact"] = bool(self.contacts.soft_sites)
        if self.contacts.soft_sites:
            result["relaxed"] = True
        return q, result
