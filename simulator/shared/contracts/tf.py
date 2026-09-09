"""The transform tree: `/tf`, `/tf_static` and the description they are read against.

A real bringup does not only publish joint angles. `robot_state_publisher` runs beside
whatever produces `/joint_states`, reads the URDF out of the `robot_description`
parameter, and turns the two into a transform tree on `/tf` and `/tf_static`. That is
what lets a client place a laser scan in the base frame, ask where the gripper is, or
draw the robot at all. Without it a client gets `frame_id` strings -- `myagv/odom`,
`ainex/base_footprint` -- naming the nodes of a tree that was never published, which is a
promise this contract made and did not keep.

Stdlib only, for the same reason `namespace.py` is: the console's cross-project contract
test loads these files by path and can only import what the console itself can. The
MuJoCo half -- turning a compiled model into transforms -- is
`mujoco_bridge.TransformTree`; the loop that publishes them is `ros_surfaces/tf_stream.py`.

Three things here are contract decisions rather than implementation:

* **`/tf` takes the robot's namespace, like every other topic it presents.** In ROS the
  tf topic is a *relative* name, so a bringup inside `<group ns="myagv">` publishes
  `/myagv/tf` unless it is explicitly remapped, and the frames inside carry the matching
  `tf_prefix`. Both halves are what this contract already does with every other name, so
  tf needs no special case: `--ros-namespace ''` gives the bare `/tf` a single-robot
  bringup presents, which is the contract this simulator claims to be indistinguishable
  from.
* **`/tf_static` is a ROS 2 topic, and only the SO-101 has one.** The myAGV's stack is
  tf1: `myagv_active.launch` runs three `pkg="tf"` `static_transform_publisher` nodes,
  and that node re-publishes onto **`/tf`** on a period rather than onto `/tf_static`.
  Its `robot_state_publisher` has nothing to put there either -- the vendor URDF's one
  joint, `base_up`, is `continuous`, so the description has no fixed joint at all. So a
  ROS 1 robot here publishes its static transforms on `/tf` on a slow clock and never
  advertises `/tf_static`, which is what `rostopic list` on the real robot would show.
* **The static half is repeated, because rosbridge has no latching.** For the SO-101 that
  is a departure: on real hardware `/tf_static` is latched and a client connecting an hour
  later still receives it, and rosbridge relays a latched topic to each new subscriber.
  This bridge has no notion of latching, so a single publication would be missed by every
  client that arrived afterwards. `STATIC_PERIOD_S` is the smallest honest fix -- the same
  transforms, more often than a real one would send them. For the ROS 1 robots it is not a
  departure at all: repeating on a period is exactly what `tf`'s own node does, just at
  1 s rather than the vendor's 10-50 ms.
* **Both dialects, as everywhere else on this graph.** The myAGV and the AiNex are ROS 1
  stacks (`tf2_msgs/TFMessage`, a `secs`/`nsecs` stamp) and the SO-101 is a ROS 2 one
  (`tf2_msgs/msg/TFMessage`, `sec`/`nanosec`). A client reads the type off `rosapi`, as
  it must for every other topic here.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

#: tf's topic names. Relative in ROS, so a namespace applies to them like any other.
TOPIC_TF = "/tf"
TOPIC_TF_STATIC = "/tf_static"

TYPE_TF_MESSAGE = "tf2_msgs/TFMessage"           # ROS 1: the myAGV and the AiNex
TYPE_TF_MESSAGE_ROS2 = "tf2_msgs/msg/TFMessage"  # ROS 2: the SO-101

#: The parameter `robot_state_publisher` reads and every RViz-shaped client looks for.
#: A relative name, so it lands at `/myagv/robot_description` under a namespace and at
#: `/robot_description` bare -- exactly where `<group ns>` puts it.
PARAM_ROBOT_DESCRIPTION = "/robot_description"

#: How often `/tf_static` is repeated. 1 Hz is far below any control rate here, so it
#: costs nothing, and a client that connects mid-run has its tree within a second.
STATIC_PERIOD_S = 1.0


def transform(parent_frame: str, child_frame: str, pos, quat, *, stamp_s: float,
              seq: int = 0, ros2: bool = False) -> dict:
    """One geometry_msgs/TransformStamped.

    `quat` is `(w, x, y, z)` -- MuJoCo's order, which is what every caller here holds.
    The message names its components, so the order matters only at this boundary; it is
    spelled out because getting it wrong produces a robot rendered in a plausible wrong
    pose rather than an error.
    """
    if ros2:
        seconds = int(stamp_s)
        header = {
            "stamp": {"sec": seconds, "nanosec": int(round((stamp_s - seconds) * 1e9))},
            "frame_id": parent_frame,
        }
    else:
        header = {
            "seq": int(seq),
            "stamp": {"secs": int(stamp_s), "nsecs": int((stamp_s % 1) * 1e9)},
            "frame_id": parent_frame,
        }
    return {
        "header": header,
        "child_frame_id": child_frame,
        "transform": {
            "translation": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
            "rotation": {
                "x": float(quat[1]),
                "y": float(quat[2]),
                "z": float(quat[3]),
                "w": float(quat[0]),
            },
        },
    }


def tf_message(entries, *, stamp_s: float, seq: int = 0, ros2: bool = False) -> dict:
    """tf2_msgs/TFMessage from `(parent_frame, child_frame, pos, quat)` tuples."""
    return {
        "transforms": [
            transform(parent, child, pos, quat, stamp_s=stamp_s, seq=seq, ros2=ros2)
            for parent, child, pos, quat in entries
        ]
    }


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """URDF's `rpy` (fixed-axis XYZ) as `(w, x, y, z)`."""
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def urdf_links(urdf_text: str) -> list[str]:
    """Every `<link>` the description declares, in file order."""
    return [el.get("name", "") for el in ET.fromstring(urdf_text).findall("link")]


def urdf_fixed_joints(urdf_text: str) -> list[tuple[str, str, tuple, tuple]]:
    """The description's fixed joints, as `(parent, child, pos, quat)`.

    These are the static half of the tree, and they have to come from the URDF rather
    than from the compiled model: **MuJoCo merges a fixed-jointed link into its parent**,
    so no body for it ever exists. The AiNex's `camera_link` and `imu_link` are exactly
    that -- 28 links in the description, 26 bodies in the model -- and they are the two
    frames a client most wants, because they are what the camera and the IMU stamp their
    messages with. Reading them from the description is also what a real
    `robot_state_publisher` does with a fixed joint, so the numbers have one source.
    """
    out: list[tuple[str, str, tuple, tuple]] = []
    for joint in ET.fromstring(urdf_text).findall("joint"):
        if joint.get("type") != "fixed":
            continue
        parent, child = joint.find("parent"), joint.find("child")
        if parent is None or child is None:
            continue
        origin = joint.find("origin")
        xyz = (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()
        rpy = (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()
        out.append(
            (
                parent.get("link", ""),
                child.get("link", ""),
                tuple(float(v) for v in xyz),
                rpy_to_quat(*(float(v) for v in rpy)),
            )
        )
    return out
