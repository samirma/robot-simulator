#!/usr/bin/env python
"""Generate model.xml from the upstream so101.xml, applying MolmoSpaces conventions.

Kept as a generator rather than a hand-edited copy so the delta from mujoco_menagerie
stays explicit and can be re-applied if the upstream model changes. The TCP frame is
*computed* from the model geometry rather than hardcoded, so it stays correct if the
jaw geometry is revised.

Edits applied:

1. **TCP site `tcp`.** MolmoSpaces' gripper convention is `+z` = approach direction with
   the fingers opening along `y` (see "Robot Conventions" in the MolmoSpaces README; the
   shipped Franka's `gripper/grasp_site` has +z along the approach). The stock
   `gripperframe` site instead uses `+x` as the approach axis, so we add a correctly
   oriented site placed at the true grasp centre (the midpoint between the jaw tips,
   ~12 mm from `gripperframe`).

   Both axes are measured from the model:
     - approach = gripper body origin -> jaw-tip midpoint
     - finger axis = fixed-jaw tip -> moving-jaw tip at a nominal grasp opening

   Note the SO-101 has a single *hinged* jaw, not a parallel-jaw gripper, so the finger
   axis is only exactly perpendicular to the approach at one opening. We measure it at
   NOMINAL_GRIP (~4 cm between the tips, a typical small-object grasp), where it is
   within ~8 degrees of perpendicular.

2. **Exo camera `exo_camera`.** Base-mounted third-person view of the workspace, so the
   robot has an exo view like the Franka configs do. The upstream `wrist_cam` is kept
   as the wrist camera.

3. **Gripper `forcerange`.** Upstream puts the gripper actuator on the `sts3215` class,
   which carries `forcerange="-2.94 2.94"` -- correct for the servo, wrong for grasping
   a light object. Measured on the sibling apple-on-plate rig at the equivalent +/-3.35:
   peak jaw force 96 N, mean 56 N, the apple slipping 397.9 mm and ending extruded onto
   the table at z 0.0199. At +/-0.30: peak 12.4 N, mean 4.8 N, slip 19.6 mm, apple held
   at z 0.1003. Below about +/-1.0 the commanded jaw value stops mattering at all, because
   the actuator saturates on force before it reaches its target -- which is the regime a
   compliant grasp wants. The arm's five other actuators are untouched.

4. **Task wrist camera `wrist`.** Added *beside* upstream's `wrist_cam` rather than
   replacing it. `wrist_cam` is menagerie's photographic camera (1920x1080, specified by
   `sensorsize`/`focal`), which is a fine thing to look through but is not what a policy
   was trained on: MolmoAct2's preprocessor stretches to 4:3 without preserving aspect,
   so a 16:9 frame arrives distorted in a way training data never was. The added camera
   is the geometry the apple-on-plate rig verified -- see the anchor below for the
   derivation of every number in it.

Run:  python robots/so101/make_model.py
"""

from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
# The generated model and its upstream source live in the shared spec dir, so every
# engine consumes the same MJCF. This adapter (make_model.py) stays engine-side.
import sys as _sys
_sys.path.insert(0, str(HERE.parents[2] / "shared"))
from robots_spec import spec_dir as _spec_dir  # noqa: E402

SPEC = _spec_dir("so101")
UPSTREAM = SPEC / "so101.xml"
OUT = SPEC / "model.xml"

# Gripper joint angle at which the finger axis is measured. ~0.04 m between jaw tips.
NOMINAL_GRIP = 0.3

FIXED_TIP = "fixed_jaw_sph_tip2"
MOVING_TIP = "moving_jaw_sph_tip2"
GRIPPER_BODY = "gripper"

GRIPPERFRAME = (
    '<site group="3" name="gripperframe" pos="0.012 -0.000218 -0.098127" quat="1 0 1 0"/>'
)
BASEFRAME = '<site group="3" name="baseframe" pos="0 0 0" quat="1 0 0 0"/>'

# Upstream drives the gripper off the sts3215 class, whose forcerange is right for the
# servo and far too strong for a compliant grasp. See edit 3 in the module docstring.
GRIPPER_ACTUATOR = (
    '<position class="sts3215" name="gripper" joint="gripper" ctrlrange="-0.17453 1.74533"/>'
)
GRIPPER_ACTUATOR_SOFT = (
    '<position class="sts3215" name="gripper" joint="gripper" ctrlrange="-0.17453 1.74533"\n'
    '      forcerange="-0.30 0.30"/>'
)

WRIST_CAM = (
    '<camera name="wrist_cam" mode="fixed" pos="0.0 0.055 -0.045" euler="-0.57 0 0" '
    'resolution="1920 1080"\n'
    '                    sensorsize="0.00576 0.00324" focal="0.0036 0.0036"/>'
)
TASK_WRIST_CAM = f"""{WRIST_CAM}
                  <!--
                    Eye-in-hand camera for the task pipeline. Every number here was
                    measured, not chosen:

                    pos    45 mm to the side on -y, NOT behind the jaws. The printed
                           wrist_roll_follower housing occludes every on-axis mount --
                           checked by ray-casting a 4 x 7 x 3 grid of candidate positions
                           at the grasp point and finding no clear line from any of them.
                           From here the working distance to the jaw centre is 127.1 mm
                           and the frame spans 195 mm.
                    xyaxes aims at the jaw centre, which projects to (0.00, -0.00) at
                           every waypoint of the scripted plan, with the apple inside
                           |u|,|v| < 0.15 through pre_grasp, descend, close and lift --
                           the whole phase where a wrist view decides anything.
                    fovy   75, matching robosuite's robot0_eye_in_hand, which is the FOV
                           the wrist views in LIBERO-derived training data were rendered
                           at. Free to match, so match it.
                    resolution MANDATORY. A <camera> with no resolution renders zero-sized
                           images and says nothing about it. 256x256 because consuming
                           policies resize to 224: publishing 256 makes that a downsample,
                           the same direction as training. It is also 65k px against the
                           307k px of a 640x480 scene camera, and render time is what
                           scales with camera count.

                    Rendering it costs real control rate (measured elsewhere: 4.1 Hz with
                    two cameras, ~2.1 Hz with four, against a 10 Hz loop), so the ROS
                    surface leaves it off unless asked. Declaring it here is free.
                  -->
                  <camera
                    name="wrist"
                    mode="fixed"
                    pos="0.000 -0.045 0.020"
                    xyaxes="0.99400 -0.03877 0.10227 -0.00000 0.93506 0.35448"
                    fovy="75"
                    resolution="256 256"
                  />"""

HEADER = """<!--
  GENERATED by make_model.py from so101.xml - do not edit by hand.

  so101.xml is mujoco_menagerie/robotstudio_so101 verbatim (Apache 2.0, see LICENSE).
  This file adds a MolmoSpaces-convention TCP site, an exo camera, a task-pipeline
  wrist camera, and a softened gripper forcerange; see make_model.py.
-->
"""

EXO_CAMERA = f"""{BASEFRAME}
      <!-- Third-person view of the arm's workspace -->
      <camera name="exo_camera" mode="fixed" pos="-0.25 -0.45 0.45"
        xyaxes="0.87 -0.5 0 0.22 0.38 0.9" fovy="58"/>"""


def compute_tcp_frame() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (pos, x_axis, y_axis, perpendicularity_error_deg) in the gripper body frame."""
    model = mujoco.MjModel.from_xml_path(str(UPSTREAM))
    data = mujoco.MjData(model)

    qadr = model.jnt_qposadr[model.joint("gripper").id]
    data.qpos[:] = 0
    data.qpos[qadr] = NOMINAL_GRIP
    mujoco.mj_forward(model, data)

    fixed_tip = data.geom_xpos[model.geom(FIXED_TIP).id]
    moving_tip = data.geom_xpos[model.geom(MOVING_TIP).id]
    body_id = model.body(GRIPPER_BODY).id
    body_rot = data.xmat[body_id].reshape(3, 3)
    body_pos = data.xpos[body_id]

    grasp_centre = (fixed_tip + moving_tip) / 2

    approach = grasp_centre - body_pos
    approach /= np.linalg.norm(approach)

    finger = fixed_tip - moving_tip
    finger /= np.linalg.norm(finger)
    error_deg = abs(90.0 - np.degrees(np.arccos(abs(float(approach @ finger)))))

    # z = approach; y = finger axis with any approach component projected out.
    z_axis = approach
    y_axis = finger - (finger @ z_axis) * z_axis
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)

    rot_world = np.column_stack([x_axis, y_axis, z_axis])
    assert np.isclose(np.linalg.det(rot_world), 1.0), "TCP frame is not right-handed"

    rot_local = body_rot.T @ rot_world
    pos_local = body_rot.T @ (grasp_centre - body_pos)
    return pos_local, rot_local[:, 0], rot_local[:, 1], error_deg


def main() -> int:
    pos, x_axis, y_axis, error_deg = compute_tcp_frame()
    fmt = lambda v: " ".join(f"{c:.6g}" for c in v)  # noqa: E731
    print(f"TCP at {fmt(pos)} (gripper frame); finger axis is {error_deg:.1f} deg off perpendicular")

    tcp_site = f"""{GRIPPERFRAME}
                <!-- MolmoSpaces TCP: +z = approach, +y = finger-opening axis, at the grasp centre.
                     Computed by make_model.py; see its docstring. -->
                <site group="3" name="tcp" pos="{fmt(pos)}"
                  xyaxes="{fmt(x_axis)} {fmt(y_axis)}"/>"""

    xml = UPSTREAM.read_text()
    for anchor, replacement, label in (
        (GRIPPERFRAME, tcp_site, "tcp site"),
        (BASEFRAME, EXO_CAMERA, "exo camera"),
        (WRIST_CAM, TASK_WRIST_CAM, "task wrist camera"),
        (GRIPPER_ACTUATOR, GRIPPER_ACTUATOR_SOFT, "gripper forcerange"),
    ):
        if anchor not in xml:
            raise SystemExit(f"anchor for {label} not found in {UPSTREAM}; upstream model changed?")
        xml = xml.replace(anchor, replacement, 1)

    idx = xml.index(">", xml.index("<mujoco")) + 1
    xml = xml[:idx] + "\n" + HEADER + xml[idx:]

    OUT.write_text(xml)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
