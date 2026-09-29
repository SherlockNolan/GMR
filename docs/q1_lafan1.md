# LAFAN1 to Q1

The Q1 source model lives in the adjacent SMP repository at
`../smp/artifacts/models/q1/q1_local_mesh.xml`.

`scripts/prepare_q1.py` creates `q1_gmr.xml`, removing the scene's free-floating
ball and adding fixed hand reference bodies 0.195 m along each forearm. The
robot still has exactly 22 actuated hinges and one free root: `nq=29`, `nv=28`.
The source model files are preserved.

The IK tables are calibrated against the initial T pose of
`walk1_subject1.bvh`. Root/leg/torso motion is scaled by 0.45, and upper limb
motion by 0.55. Position offsets account for the different link origins;
rotation offsets align the human and robot reference poses. Hand orientation
is not constrained separately, because Q1 has no wrists. These coefficients
are an initial configuration, not a claim of dynamic tracking feasibility.

## Environment and preparation

```bash
cd /root/zy/code/GMR
uv sync
MUJOCO_GL=disable uv run scripts/prepare_q1.py \
  --model-dir /root/zy/code/smp/artifacts/models/q1 \
  --reference-bvh /root/zy/code/smp/datasets/lafan1/raw/walk1_subject1.bvh
```

BVH retargeting uses CPU MuJoCo and DAQP. The uv environment does not require
Torch, CUDA or body model assets. Torch-based FK/SMPL workflows have separate
optional dependencies: `uv sync --extra body-models`.

## Batch conversion

```bash
bash scripts/run_lafan1_q1.sh
```

The launcher processes all 77 BVH clips with 16 CPU workers, preserving their
original 30 FPS and frame counts. Individual files are written atomically;
rerunning skips completed outputs with matching model/configuration settings.
Use `bash scripts/run_lafan1_q1.sh --override` after changing the calibration.

Output in `../smp/datasets/lafan1/q1_gmr/`:

* `pkl/`: root position, root quaternion, joint positions, local body positions,
  joint/body names, frame rate and processing metadata.
* `csv/`: headerless rows containing root XYZ, root quaternion **XYZW**, then
  the 22 Q1 joint angles, in MuJoCo joint order (29 columns total).
* `reports/`: per-clip ground shift, joint limit/velocity statistics and review
  flags.
* `status.json`: current totals and detailed completed/failed clip records.
* `retarget.log`: batch progress and errors.

For this model, waist **roll precedes yaw** in the output; the G1 column order
must not be reused. The complete order is saved in every PKL/report and in
`q1_gmr_metadata.json` next to the derived model.

The optional ground alignment applies one constant upward translation per
clip to remove collision-geometry penetration. This preserves velocities but
can leave other frames floating; it does not fix sliding, self-collision,
actuator limits or dynamic balance. Clips with a ground shift over 5 cm or
joint speeds over 20 rad/s are marked `requires_review`. Floor-contact motions
such as getting up need particular care before training. `--no-ground-align`
exports the unshifted IK trajectories instead.

## Progress

```bash
tail -f /root/zy/code/smp/datasets/lafan1/q1_gmr/retarget.log
uv run python -c 'import json; s=json.load(open("../smp/datasets/lafan1/q1_gmr/status.json")); print(s["status"], s["completed"], "/", s["total"], "failed:", s["failed"])'
```

This step produces Q1 motion data. SMP's current G1-only CSV feature converter,
feature slicing, GSI and RL robot configuration still need adaptation before
these files can be used to train a Q1 diffusion prior or policy.

## Headless video preview

```bash
cd /root/zy/code/GMR
MUJOCO_GL=egl EGL_DEVICE_ID=0 uv run scripts/render_robot_motion.py \
  --robot q1 \
  --motion-file ../smp/datasets/lafan1/q1_gmr/pkl/walk1_subject1.pkl \
  --output ../smp/datasets/lafan1/q1_gmr/previews/walk1_subject1.mp4 \
  --start-seconds 3 --duration 10
```

The preview replays kinematic poses; it is not a simulation of a trained policy.

To render the selected ten flagged clips and all four unflagged clips in full:

```bash
uv run scripts/render_q1_review_previews.py
```

The output uses `review_` and `unflagged_` filename prefixes, preserves every
source frame at 30 FPS, and includes `previews/index.html`, a selection list,
and `selection_manifest.json` with video frame counts and completion status.

## Contact-aware Q1 v2 (flat-ground walk/run)

```bash
cd /root/zy/code/GMR
uv run scripts/prepare_q1_contact.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run scripts/retarget_q1_contact.py --workers 8
uv run scripts/summarize_q1_contact.py
```

This creates a separate `q1_contact_v2.xml` from the supplied Q1 source, removes
the decorative ball, measures every mechanism offset, and derives hand proxies
from the distal forearm STL geometry. The original source and v1 outputs are
preserved. Geometry, reference-pose lengths, foot collision spheres and URDF
joint speed limits are recorded in `../smp/artifacts/models/q1/q1_contact_geometry.json`
and the accompanying Markdown table.

The new path measures human landmark distances per clip and maps thighs,
shanks, upper arms and forearms individually. Shoulder/hip offsets come from
Q1's mechanism; the IK solver retains all non-coincident axes. The fixed head
length is reported but has no independent IK target. Root translation uses
the combined leg-length ratio rather than a fixed 0.45. The orientation offsets
retain the v1 reference calibration from `walk1_subject1`.
The sixteen source BVH files use the same canonical limb lengths for all five
subjects; per-clip measurements confirm this rather than assuming different
anthropometrics from the subject names.

Estimated support labels use each human Foot/Toe marker's robust low-height
baseline and horizontal speed hysteresis. Foot is a heel proxy, not an actual
heel marker. Running flight phases remain unconstrained by support anchors.
Labels assume flat ground and are not intended for obstacles/getup/crawling.

During a support patch, its Q1 sphere-centre XY anchor is fixed and its target
height equals the sphere's 8 mm radius. Hard QP bounds enforce contact position
within 0.4 mm per axis and all foot spheres above the floor within 0.05 mm.
The pelvis has a soft position target and can translate/rotate. Joint bounds
combine XML ranges with **previous output-frame** URDF speed limits; successive
internal IK iterations cannot multiply the allowed output displacement.
QP limits are passed explicitly by keyword for the current Mink API.
The previous output pose is also a soft posture target (cost 1.5); this reduces
branch changes but is not a hard acceleration or jerk bound.

Rigid-foot touchdown uses continuation within an output frame to reach
compatible heel/toe anchors. Frames that still require relaxed constraints
or exceed the contact error threshold are reported, with rejection intervals
in `quality_summary.json`. No constant whole-clip ground shift is applied.

Output: `../smp/datasets/lafan1/q1_gmr_v2/`, initially all 12 walk + 4 run clips.
PKL/CSV formats match v1; `contacts/*.npz` also records per-frame support labels,
source marker heights/speeds, contact anchors, actual sphere centres, selected
sites, pelvis adjustments and relaxation flags. Reports compare both versions
under the **same** estimated human support phases. Check `README.md`,
`quality_summary.json`, and `previews/index.html` for actual results.

Preview a v2 clip with the correct asset:

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 uv run scripts/render_robot_motion.py \
  --robot q1 --model-file ../smp/artifacts/models/q1/q1_contact_v2.xml \
  --motion-file ../smp/datasets/lafan1/q1_gmr_v2/pkl/walk1_subject1.pkl \
  --output ../smp/datasets/lafan1/q1_gmr_v2/previews/walk1_subject1.mp4 \
  --start-seconds 12 --duration 20 --fps 30 --label "Contact v2 - walk1_subject1"
```

Contact and velocity constraints are kinematic checks, not a proof of dynamic
balance, torque feasibility, self-collision avoidance or policy tracking.
SMP's Q1 feature/prior/RL adaptation remains a subsequent stage.

Render all sixteen 12–32 s excerpts and representative v1/v2 side-by-side videos:

```bash
uv run scripts/render_q1_contact_previews.py --workers 4
```

The render manifest records actual video frame counts. Shorter end-of-clip
review excerpts use the available remaining frames. `previews/comparisons.html`
shows the representative walk/run comparisons; `previews/index.html` links all
sixteen clips and their complete human support timelines.

## Expanded actions (v3)

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run scripts/retarget_q1_contact.py --contact-mode expanded --all --workers 8 \
  --recover-contacts --body-contact-cost 100 \
  --output ../smp/datasets/lafan1/q1_gmr_v3
uv run scripts/summarize_q1_expanded.py
uv run scripts/export_q1_candidates.py
uv run scripts/render_q1_expanded_previews.py --workers 4
```

The expanded path retains the v2 Q1 asset and measured segment definitions.
It covers all source recording families, gates foot support by sole orientation,
and calibrates marker baselines using upright frames. It constrains all supplied
robot collision primitives above the ground, including forearms, torso, head,
hips and shanks. Low-pose human hand, knee, pelvis, torso and head support
estimates add soft body-surface height goals; horizontal motion and rolling are
allowed. These estimates use marker proximity and speed, not force labels.

Obstacle recordings contain skeleton trajectories without obstacle meshes.
Stable elevated foot marker plateaus provide inferred support heights, which
are visualized as labelled orange patches. This does not reconstruct walls,
platform edges or collision geometry. All obstacle clips require environment
review and are excluded from the conservative candidate manifest.
Each foot retains one support plane throughout a support phase, preferring
a ground estimate when marker estimates disagree. Heel/toe anchors cannot
impose different elevations on the same flat rigid foot. Jump support also
uses a vertical-speed release gate (0.55 m/s) for near-vertical takeoffs.
The soft body-surface height cost defaults to 100 and can be set with
`--body-contact-cost`; hardware speed/range limits remain unchanged.

After a first expanded pass, reprocess clips affected by these contact rules:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run scripts/refine_q1_expanded.py --workers 8 --body-contact-cost 100
uv run scripts/summarize_q1_expanded.py
uv run scripts/render_q1_expanded_previews.py --workers 4 --skip-existing
```

Initial reports are retained in `reports_initial/`. `refinement_status.json`
tracks the selected rerun; the summarizer restores a full-dataset `status.json`
only when all source clips have complete frame counts. Cached previews are
reused only when the motion recipe fingerprint and selected interval match.
`--recover-contacts` records rollback/retry, temporarily deferred contact targets,
and any preceding-pose fallback when an estimated contact is unreachable.
Retries reduce the internal trust region while retaining joint range, output
velocity and real-ground collision bounds. Those frames are explicitly
excluded from candidates, even if a retry subsequently reaches the target.
Inspect `solver_recovery_frames`, `frozen_output_frames`, `soft_foot_contact_frames`
and the corresponding NPZ arrays. This keeps an individual bad estimate from
discarding an entire recording without silently accepting the failed interval.

`quality_summary.json` records contact/floor failures, large pelvis corrections,
single-frame joint changes over 0.35 rad, body-contact review intervals and
missing environment geometry. `kinematic_candidate_intervals.json` selects
continuous intervals of at least 90 frames after excluding those cases and
the initial 90 frames. It is a numerical/fidelity filter, not a dynamics or
training certification. Full PKL/CSV files retain every original frame.

`scripts/export_q1_candidates.py` materializes these intervals in
`q1_gmr_v3/candidates/pkl/` and `candidates/csv/`. The candidate manifest records
each original clip and its inclusive start / exclusive end frame, so segments
remain traceable. Run the exporter again after changing the audit or motions;
the complete source sequences remain in the parent `pkl/` and `csv/` folders.
CSV columns are root XYZ, quaternion XYZW, then the same 22 Q1 hinge joints.
These segments still need dynamic tracking validation and SMP Q1 adaptation.

The expanded preview page groups every clip by motion family and shows one
representative 20-second interval selected from active motion, body contacts,
turns or elevated support. Full sequences can be rendered with
`render_robot_motion.py --full` and the model specified in the per-clip report.
