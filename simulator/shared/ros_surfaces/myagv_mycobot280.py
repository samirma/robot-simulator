"""The myAGV + myCobot 280's ROS interface: `robots_specs/myagv_mycobot280/ros.yml`, transcribed.

That file `extends` the myAGV's: the composite's graph is the myAGV's boot launch -- served
here by the myAGV's own surface, `ros_surfaces/myagv.py`, unchanged -- plus
`myagv_navigation/launch/composite_robot_navigation_active.launch`: `map_server_for_test`,
`amcl` and `move_base`. What this module transcribes is only those additions. The
myCobot 280 Pi on the deck and its adaptive gripper are commanded over pymycobot's TCP
socket, not ROS, so they add nothing to the graph; the simulated arm holds its zero pose
(`REST_POSITIONS`) and no socket is served.

The navigation nodes, as far as this surface carries them:

* `map_server_for_test` serves the launch's map, `composite_robot_map.yaml`, from
  `robots_specs/myagv_mycobot280/myagv_navigation/` -- the vendor's demo field, not the
  simulated room: it is what the real robot loads wherever it stands. `/change_map` loads
  another map YAML by path.
* `amcl` starts at the launch's initial pose and follows the base's odometry: the pose it
  publishes is the odometry composed with its `map -> odom` correction, which
  `/initialpose` and `/set_map` reset. It does **not** match scans against the map (a gap,
  recorded in the report of the change that added it); it updates -- `/amcl_pose`,
  `/particlecloud` -- as amcl does, when the base has moved `update_min_d` or turned
  `update_min_a`, and re-sends `map -> odom` on every laser scan, future-dated by its
  `transform_tolerance`.
* `move_base` takes goals on its actionlib topics and `/move_base_simple/goal`, and drives
  the base the way TrajectoryPlannerROS is configured to (non-holonomic, `max_vel_x`,
  `min_in_place_vel_theta`, the goal tolerances), publishing on `/cmd_vel` at the
  controller frequency until the goal is reached, cancelled or stuck. It plans a straight
  line and does not avoid obstacles (a gap). Its costmaps publish the static map (the
  global one whole, the local one as its rolling window around the base), without the
  obstacle and inflation layers (a gap), at their publish frequencies; its status goes out
  at 5 Hz.

The stop command is two messages: cancel every goal on `/move_base/cancel`, then a zero
Twist on `/cmd_vel` (`STOP_COMMAND`).
"""

from __future__ import annotations

import math
import sys

from contracts.physical import Figure

from ros_surfaces import myagv as base_contract

# ------------------------------------------------------------------------------ nodes

NODE_MAP_SERVER = "map_server_for_test"
NODE_AMCL = "amcl"
NODE_MOVE_BASE = "move_base"

# ------------------------------------------------------------------------------ topics

TOPIC_MAP = "/map"
TOPIC_MAP_METADATA = "/map_metadata"
TOPIC_INITIALPOSE = "/initialpose"
TOPIC_AMCL_POSE = "/amcl_pose"
TOPIC_PARTICLECLOUD = "/particlecloud"
TOPIC_TF = "/tf"
TOPIC_GOAL = "/move_base/goal"
TOPIC_CANCEL = "/move_base/cancel"
TOPIC_FEEDBACK = "/move_base/feedback"
TOPIC_STATUS = "/move_base/status"
TOPIC_RESULT = "/move_base/result"
TOPIC_SIMPLE_GOAL = "/move_base_simple/goal"
TOPIC_CMD_VEL = "/cmd_vel"
TOPIC_CURRENT_GOAL = "/move_base/current_goal"
TOPIC_RECOVERY_STATUS = "/move_base/recovery_status"
TOPIC_GLOBAL_COSTMAP = "/move_base/global_costmap/costmap"
TOPIC_GLOBAL_UPDATES = "/move_base/global_costmap/costmap_updates"
TOPIC_GLOBAL_FOOTPRINT = "/move_base/global_costmap/footprint"
TOPIC_CLEARING_ENDPOINTS = "/move_base/global_costmap/obstacle_layer/clearing_endpoints"
TOPIC_LOCAL_COSTMAP = "/move_base/local_costmap/costmap"
TOPIC_LOCAL_UPDATES = "/move_base/local_costmap/costmap_updates"
TOPIC_LOCAL_FOOTPRINT = "/move_base/local_costmap/footprint"
TOPIC_PLAN = "/move_base/GlobalPlanner/plan"
TOPIC_POTENTIAL = "/move_base/GlobalPlanner/potential"
TOPIC_TP_GLOBAL_PLAN = "/move_base/TrajectoryPlannerROS/global_plan"
TOPIC_TP_LOCAL_PLAN = "/move_base/TrajectoryPlannerROS/local_plan"
TOPIC_COST_CLOUD = "/move_base/TrajectoryPlannerROS/cost_cloud"

TYPE_GRID = "nav_msgs/OccupancyGrid"
TYPE_MAP_METADATA = "nav_msgs/MapMetaData"
TYPE_POSE_COV = "geometry_msgs/PoseWithCovarianceStamped"
TYPE_POSE_ARRAY = "geometry_msgs/PoseArray"
TYPE_TF_MESSAGE = "tf2_msgs/TFMessage"
TYPE_ACTION_GOAL = "move_base_msgs/MoveBaseActionGoal"
TYPE_GOAL_ID = "actionlib_msgs/GoalID"
TYPE_ACTION_FEEDBACK = "move_base_msgs/MoveBaseActionFeedback"
TYPE_STATUS_ARRAY = "actionlib_msgs/GoalStatusArray"
TYPE_ACTION_RESULT = "move_base_msgs/MoveBaseActionResult"
TYPE_POSE_STAMPED = "geometry_msgs/PoseStamped"
TYPE_TWIST = "geometry_msgs/Twist"
TYPE_RECOVERY_STATUS = "move_base_msgs/RecoveryStatus"
TYPE_GRID_UPDATE = "map_msgs/OccupancyGridUpdate"
TYPE_POLYGON_STAMPED = "geometry_msgs/PolygonStamped"
TYPE_POLYGON = "geometry_msgs/Polygon"
TYPE_POINT_CLOUD = "sensor_msgs/PointCloud"
TYPE_PATH = "nav_msgs/Path"
TYPE_POINT_CLOUD2 = "sensor_msgs/PointCloud2"

EVENT = "event"
LATCHED = "latched"

#: What the composite adds to the myAGV's topics, as `(name, type, direction, node,
#: rate_hz)`, one row per the ROS file's.
TOPICS: tuple[tuple[str, str, str, str, float | str], ...] = (
    (TOPIC_MAP, TYPE_GRID, "out", NODE_MAP_SERVER, LATCHED),
    (TOPIC_MAP_METADATA, TYPE_MAP_METADATA, "out", NODE_MAP_SERVER, LATCHED),
    (TOPIC_INITIALPOSE, TYPE_POSE_COV, "in", NODE_AMCL, EVENT),
    (TOPIC_AMCL_POSE, TYPE_POSE_COV, "out", NODE_AMCL, EVENT),
    (TOPIC_PARTICLECLOUD, TYPE_POSE_ARRAY, "out", NODE_AMCL, EVENT),
    (TOPIC_TF, TYPE_TF_MESSAGE, "out", NODE_AMCL, 30.0),
    (TOPIC_GOAL, TYPE_ACTION_GOAL, "in", NODE_MOVE_BASE, EVENT),
    (TOPIC_CANCEL, TYPE_GOAL_ID, "in", NODE_MOVE_BASE, EVENT),
    (TOPIC_FEEDBACK, TYPE_ACTION_FEEDBACK, "out", NODE_MOVE_BASE, 5.0),
    (TOPIC_STATUS, TYPE_STATUS_ARRAY, "out", NODE_MOVE_BASE, 5.0),
    (TOPIC_RESULT, TYPE_ACTION_RESULT, "out", NODE_MOVE_BASE, EVENT),
    (TOPIC_SIMPLE_GOAL, TYPE_POSE_STAMPED, "in", NODE_MOVE_BASE, EVENT),
    (TOPIC_CMD_VEL, TYPE_TWIST, "out", NODE_MOVE_BASE, 5.0),
    (TOPIC_CURRENT_GOAL, TYPE_POSE_STAMPED, "out", NODE_MOVE_BASE, EVENT),
    (TOPIC_RECOVERY_STATUS, TYPE_RECOVERY_STATUS, "out", NODE_MOVE_BASE, EVENT),
    (TOPIC_GLOBAL_COSTMAP, TYPE_GRID, "out", NODE_MOVE_BASE, 0.3),
    (TOPIC_GLOBAL_UPDATES, TYPE_GRID_UPDATE, "out", NODE_MOVE_BASE, 0.3),
    (TOPIC_GLOBAL_FOOTPRINT, TYPE_POLYGON_STAMPED, "out", NODE_MOVE_BASE, 0.3),
    (TOPIC_GLOBAL_FOOTPRINT, TYPE_POLYGON, "in", NODE_MOVE_BASE, EVENT),
    (TOPIC_CLEARING_ENDPOINTS, TYPE_POINT_CLOUD, "out", NODE_MOVE_BASE, 0.3),
    (TOPIC_LOCAL_COSTMAP, TYPE_GRID, "out", NODE_MOVE_BASE, 2.0),
    (TOPIC_LOCAL_UPDATES, TYPE_GRID_UPDATE, "out", NODE_MOVE_BASE, 2.0),
    (TOPIC_LOCAL_FOOTPRINT, TYPE_POLYGON_STAMPED, "out", NODE_MOVE_BASE, 2.0),
    (TOPIC_LOCAL_FOOTPRINT, TYPE_POLYGON, "in", NODE_MOVE_BASE, EVENT),
    (TOPIC_PLAN, TYPE_PATH, "out", NODE_MOVE_BASE, 1.0),
    (TOPIC_POTENTIAL, TYPE_GRID, "out", NODE_MOVE_BASE, 1.0),
    (TOPIC_TP_GLOBAL_PLAN, TYPE_PATH, "out", NODE_MOVE_BASE, 5.0),
    (TOPIC_TP_LOCAL_PLAN, TYPE_PATH, "out", NODE_MOVE_BASE, 5.0),
    (TOPIC_COST_CLOUD, TYPE_POINT_CLOUD2, "out", NODE_MOVE_BASE, 5.0),
)

#: The rows whose rate holds only while something is so (the ROS file's `active_while`):
#: move_base's planning and control while a goal is active, and the costmaps' partial
#: updates, which `always_send_full_costmap` means are never sent.
WHILE_GOAL = frozenset({TOPIC_FEEDBACK, TOPIC_CMD_VEL, TOPIC_PLAN, TOPIC_POTENTIAL,
                        TOPIC_TP_GLOBAL_PLAN, TOPIC_TP_LOCAL_PLAN, TOPIC_COST_CLOUD})
NEVER_SENT = frozenset({TOPIC_GLOBAL_UPDATES, TOPIC_LOCAL_UPDATES})
CONDITIONAL = WHILE_GOAL | NEVER_SENT

# ---------------------------------------------------------------------------- services

SRV_GET_MAP = "nav_msgs/GetMap"
SRV_LOAD_MAP = "nav_msgs/LoadMap"
SRV_EMPTY = "std_srvs/Empty"
SRV_SET_MAP = "nav_msgs/SetMap"
SRV_RECONFIGURE = "dynamic_reconfigure/Reconfigure"
SRV_GET_PLAN = "nav_msgs/GetPlan"

SERVICES: tuple[tuple[str, str, str], ...] = (
    ("/static_map", SRV_GET_MAP, NODE_MAP_SERVER),
    ("/change_map", SRV_LOAD_MAP, NODE_MAP_SERVER),
    ("/global_localization", SRV_EMPTY, NODE_AMCL),
    ("/request_nomotion_update", SRV_EMPTY, NODE_AMCL),
    ("/set_map", SRV_SET_MAP, NODE_AMCL),
    ("/amcl/set_parameters", SRV_RECONFIGURE, NODE_AMCL),
    ("/move_base/make_plan", SRV_GET_PLAN, NODE_MOVE_BASE),
    ("/move_base/clear_costmaps", SRV_EMPTY, NODE_MOVE_BASE),
    ("/move_base/GlobalPlanner/make_plan", SRV_GET_PLAN, NODE_MOVE_BASE),
    ("/move_base/set_parameters", SRV_RECONFIGURE, NODE_MOVE_BASE),
    ("/move_base/global_costmap/set_parameters", SRV_RECONFIGURE, NODE_MOVE_BASE),
    ("/move_base/local_costmap/set_parameters", SRV_RECONFIGURE, NODE_MOVE_BASE),
    ("/move_base/global_costmap/static_layer/set_parameters", SRV_RECONFIGURE, NODE_MOVE_BASE),
    ("/move_base/local_costmap/static_layer/set_parameters", SRV_RECONFIGURE, NODE_MOVE_BASE),
    ("/move_base/global_costmap/obstacle_layer/set_parameters", SRV_RECONFIGURE,
     NODE_MOVE_BASE),
    ("/move_base/local_costmap/obstacle_layer/set_parameters", SRV_RECONFIGURE, NODE_MOVE_BASE),
    ("/move_base/global_costmap/inflation_layer/set_parameters", SRV_RECONFIGURE,
     NODE_MOVE_BASE),
    ("/move_base/local_costmap/inflation_layer/set_parameters", SRV_RECONFIGURE,
     NODE_MOVE_BASE),
    ("/move_base/GlobalPlanner/set_parameters", SRV_RECONFIGURE, NODE_MOVE_BASE),
    ("/move_base/TrajectoryPlannerROS/set_parameters", SRV_RECONFIGURE, NODE_MOVE_BASE),
)

# -------------------------------------------------------------------------- parameters

#: Where the launch's map and parameter files are, under `robots_specs/<id>/`.
NAVIGATION_DIR = "myagv_navigation"
MAP_FILE = "map/composite_robot_map.yaml"
PARAM_DIR = "param/base_local_planner_param"

#: The launch, as the parameter server gets it: `(namespace, file)` for each `rosparam
#: load`, in launch order, then each `<param>` (which a later load does not override).
#: amcl's is the path ros.yml records as the user-fixed one (the launch's own is missing).
PARAM_LOADS: tuple[tuple[str, str], ...] = (
    ("/amcl", "amcl.yaml"),
    ("/move_base/global_costmap", "costmap_common_params.yaml"),
    ("/move_base/local_costmap", "costmap_common_params.yaml"),
    ("/move_base", "local_costmap_params.yaml"),
    ("/move_base", "global_costmap_params.yaml"),
    ("/move_base/move_base", "base_global_planner_params.yaml"),
    ("/move_base", "base_local_planner_params.yaml"),
)
LAUNCH_PARAMS: dict[str, object] = {
    "/map_server_for_test/frame_id": "map",
    "/amcl/initial_pose_x": 0.795,
    "/amcl/initial_pose_y": -0.452,
    "/amcl/initial_pose_a": -2.229,
    "/move_base/base_global_planner": "global_planner/GlobalPlanner",
    "/move_base/planner_frequency": 1.0,
    "/move_base/planner_patience": 2.0,
    "/move_base/base_local_planner": "base_local_planner/TrajectoryPlannerROS",
    "/move_base/controller_frequency": 5.0,
    "/move_base/controller_patience": 3.0,
    "/move_base/clearing_rotation_allowed": True,
}

# ------------------------------------------------------------------------- behaviour

#: The arm and gripper at the pose the robot is spawned in: every servo at its zero, the
#: myCobot 280's calibrated upright pose. The composite's bringup does not move the arm.
ARM_JOINTS = ("joint2_to_joint1", "joint3_to_joint2", "joint4_to_joint3", "joint5_to_joint4",
              "joint6_to_joint5", "joint6output_to_joint6")
GRIPPER_JOINT = "gripper_controller"
REST_POSITIONS: dict[str, float] = {**{j: 0.0 for j in ARM_JOINTS}, GRIPPER_JOINT: 0.0}

#: actionlib's GoalStatus values.
PENDING, ACTIVE, PREEMPTED, SUCCEEDED, ABORTED = 0, 1, 2, 3, 4
#: How long actionlib keeps a finished goal in its status list (status_list_timeout).
STATUS_LIST_TIMEOUT_S = 5.0
#: A goal the base has not closed on by this much over `STUCK_S` is aborted.
STUCK_PROGRESS_M, STUCK_PROGRESS_RAD, STUCK_S = 0.05, 0.1, 30.0
#: How far off the goal's bearing the base turns in place before driving at it.
HEADING_WINDOW_RAD = 0.3

#: The stop command, as the ROS file states it: cancel every navigation goal (an empty
#: GoalID), then a zero Twist on /cmd_vel.
STOP_CANCEL = {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}
STOP_COMMAND = base_contract.STOP_COMMAND

#: The myAGV's scan and camera rate: amcl re-sends map -> odom on every scan.
SCAN_HZ = base_contract.rate_of(base_contract.TOPIC_SCAN, base_contract.NODE_LIDAR)

LOOP_HZ = base_contract.LOOP_HZ

# --------------------------------------------------------------- physical figures

_ARM_PAGE = "https://www.elephantrobotics.com/en/mycobot-280-pi-2023-en/"

#: The published figures: the myAGV's (the composite stands on it unchanged), and the
#: myCobot 280 Pi's degrees of freedom.
PHYSICAL_FIGURES: tuple[Figure, ...] = tuple(
    f for f in base_contract.PHYSICAL_FIGURES if f.key != "mass_kg"
) + (
    Figure("arm_dof", "arm degrees of freedom", 6.0, "", 0.0, _ARM_PAGE,
           "Degree of Freedom: 6",
           "hinge joints of the arm chain with a position servo each, gripper excluded"),
)


# ---------------------------------------------------------------------- the launch's files


def load_yaml_params(namespace: str, text: str) -> dict[str, object]:
    """`rosparam load` of one YAML under `namespace`: nested maps flattened to leaves."""
    import yaml

    out: dict[str, object] = {}

    def walk(prefix: str, value) -> None:
        if isinstance(value, dict):
            for key, sub in value.items():
                walk(f"{prefix}/{key}", sub)
        else:
            out[prefix] = value

    walk(namespace.rstrip("/"), yaml.safe_load(text) or {})
    return out


def parameters(folder) -> dict[str, object]:
    """Every parameter the navigation launch puts on the server, with its value."""
    params: dict[str, object] = {}
    for namespace, name in PARAM_LOADS:
        params.update(load_yaml_params(namespace, (folder / PARAM_DIR / name).read_text()))
    params.update(LAUNCH_PARAMS)
    return params


def load_map(yaml_path) -> dict:
    """map_server's reading of a map YAML and its image (trinary mode, PGM only)."""
    from pathlib import Path

    import yaml

    yaml_path = Path(yaml_path)
    meta = yaml.safe_load(yaml_path.read_text())
    image = yaml_path.parent / meta["image"]
    width, height, pixels = _read_pgm(image)
    negate = int(meta.get("negate", 0))
    occ_t, free_t = float(meta["occupied_thresh"]), float(meta["free_thresh"])
    data = [0] * (width * height)
    for row in range(height):
        for col in range(width):
            p = pixels[row * width + col]
            occ = p / 255.0 if negate else (255 - p) / 255.0
            value = 100 if occ > occ_t else 0 if occ < free_t else -1
            data[(height - 1 - row) * width + col] = value
    origin = [float(v) for v in meta["origin"]]
    return {"resolution": float(meta["resolution"]), "width": width, "height": height,
            "origin": origin, "data": data}


def _read_pgm(path) -> tuple[int, int, bytes]:
    raw = path.read_bytes()
    fields, i = [], 0
    while len(fields) < 4:
        while raw[i:i + 1].isspace():
            i += 1
        if raw[i:i + 1] == b"#":
            while raw[i:i + 1] not in (b"\n", b""):
                i += 1
            continue
        j = i
        while not raw[j:j + 1].isspace():
            j += 1
        fields.append(raw[i:j])
        i = j
    if fields[0] != b"P5":
        raise ValueError(f"{path}: only binary PGM (P5) is read")
    width, height = int(fields[1]), int(fields[2])
    return width, height, raw[i + 1:i + 1 + width * height]


# ---------------------------------------------------------------------- message shapes


def _stamp(seq: int, frame_id: str, stamp_s: float) -> dict:
    return {"seq": int(seq),
            "stamp": {"secs": int(stamp_s), "nsecs": int(round((stamp_s % 1) * 1e9)) % 10 ** 9},
            "frame_id": frame_id}


def _time(stamp_s: float) -> dict:
    return {"secs": int(stamp_s), "nsecs": int((stamp_s % 1) * 1e9)}


def _pose(x: float, y: float, yaw: float) -> dict:
    return {"position": {"x": float(x), "y": float(y), "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": math.sin(yaw / 2.0),
                            "w": math.cos(yaw / 2.0)}}


def _yaw_of(q: dict) -> float:
    w, x, y, z = (float(q.get(k, 0.0) or 0.0) for k in ("w", "x", "y", "z"))
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _compose(a, b):
    ax, ay, ayaw = a
    bx, by, byaw = b
    c, s = math.cos(ayaw), math.sin(ayaw)
    return (ax + c * bx - s * by, ay + s * bx + c * by, _wrap(ayaw + byaw))


def _inverse(a):
    ax, ay, ayaw = a
    c, s = math.cos(ayaw), math.sin(ayaw)
    return (-c * ax - s * ay, s * ax - c * ay, _wrap(-ayaw))


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _grid(seq: int, frame_id: str, stamp_s: float, grid: dict, load_time: float) -> dict:
    return {"header": _stamp(seq, frame_id, stamp_s),
            "info": _info(grid, load_time), "data": grid["data"]}


def _info(grid: dict, load_time: float) -> dict:
    ox, oy, oyaw = grid["origin"]
    return {"map_load_time": _time(load_time), "resolution": grid["resolution"],
            "width": grid["width"], "height": grid["height"], "origin": _pose(ox, oy, oyaw)}


def _path(seq: int, frame_id: str, stamp_s: float, points) -> dict:
    return {"header": _stamp(seq, frame_id, stamp_s),
            "poses": [{"header": _stamp(0, frame_id, stamp_s), "pose": _pose(x, y, yaw)}
                      for x, y, yaw in points]}


def straight_line(start, goal, spacing: float) -> list:
    """GlobalPlanner's stand-in: poses from `start` to `goal` every `spacing` metres."""
    (sx, sy, _), (gx, gy, gyaw) = start, goal
    d = math.hypot(gx - sx, gy - sy)
    n = max(1, int(d / spacing))
    heading = math.atan2(gy - sy, gx - sx) if d > 1e-9 else gyaw
    return [(sx + (gx - sx) * i / n, sy + (gy - sy) * i / n, heading if i < n else gyaw)
            for i in range(n + 1)]


# ------------------------------------------------------------------------- the surface


def attach_ros(bus, base, model, camera: str | None, *, jpeg_quality: int = 80,
               lidar: dict | None = None, scene_option=None, world_reset=None,
               prefix: str = ""):
    """Wire one myAGV + myCobot 280 onto `bus`: the myAGV's surface for the base, then the
    navigation launch's nodes. Returns the per-step callback (`rate_hz`: the myAGV's)."""
    import threading

    import robots_spec
    from contracts.tf import tf_message

    base_step = base_contract.attach_ros(
        bus, base, model, camera, jpeg_quality=jpeg_quality, lidar=lidar,
        scene_option=scene_option, world_reset=world_reset, prefix=prefix)
    drive = base_step.on_cmd_vel
    folder = robots_spec.robot("myagv_mycobot280").folder / NAVIGATION_DIR
    lock = threading.Lock()
    frame = bus.frame
    f_map, f_odom, f_base = frame("map"), frame(base_contract.FRAME_ODOM), \
        frame(base_contract.FRAME_BASE)

    # -- declarations -----------------------------------------------------------------
    for name, mtype, direction, node, _rate in TOPICS:
        if direction == "out" and name != TOPIC_CMD_VEL:
            bus.advertise(name, mtype, node=node)
    bus.advertise(TOPIC_CMD_VEL, TYPE_TWIST, node=NODE_MOVE_BASE)
    params = parameters(folder)
    for name, value in params.items():
        bus.set_param(name, value)

    # -- map_server_for_test -----------------------------------------------------------------
    seqs: dict[str, int] = {}

    def seq(key: str) -> int:
        seqs[key] = seqs.get(key, 0) + 1
        return seqs[key]

    state = {"map": load_map(folder / MAP_FILE), "map_time": 0.0, "now": 0.0}

    def publish_map() -> None:
        grid, t = state["map"], state["map_time"]
        bus.publish(TOPIC_MAP_METADATA, _info(grid, t), TYPE_MAP_METADATA, latched=True,
                    node=NODE_MAP_SERVER)
        bus.publish(TOPIC_MAP, _grid(seq("map"), f_map, t, grid, t), TYPE_GRID, latched=True,
                    node=NODE_MAP_SERVER)

    publish_map()

    def static_map(_args: dict) -> dict:
        grid = state["map"]
        return {"map": _grid(0, f_map, state["now"], grid, state["map_time"])}

    def change_map(args: dict) -> dict:
        from pathlib import Path

        url = str((args or {}).get("map_url", ""))
        empty = {"header": _stamp(0, "", 0.0), "info": _info(
            {"resolution": 0.0, "width": 0, "height": 0, "origin": [0, 0, 0]}, 0.0), "data": []}
        if not Path(url).is_file():
            return {"map": empty, "result": 1}
        try:
            grid = load_map(url)
        except Exception:
            return {"map": empty, "result": 2}
        with lock:
            state["map"], state["map_time"] = grid, state["now"]
        publish_map()
        return {"map": _grid(0, f_map, state["now"], grid, state["map_time"]), "result": 0}

    # -- amcl ------------------------------------------------------------------------------------
    initial = (LAUNCH_PARAMS["/amcl/initial_pose_x"], LAUNCH_PARAMS["/amcl/initial_pose_y"],
               LAUNCH_PARAMS["/amcl/initial_pose_a"])
    amcl = {"map_odom": None, "reset_to": initial, "last_update": None, "force": True,
            "odom": None}
    update_min_d = float(params.get("/amcl/update_min_d", 0.2))
    update_min_a = float(params.get("/amcl/update_min_a", math.pi / 6))
    tolerance = float(params.get("/amcl/transform_tolerance", 0.1))
    particles = int(params.get("/amcl/min_particles", 100))

    def on_initialpose(msg: dict) -> None:
        pose = ((msg.get("pose") or {}).get("pose")) or {}
        position = pose.get("position") or {}
        with lock:
            amcl["reset_to"] = (float(position.get("x", 0.0)), float(position.get("y", 0.0)),
                                _yaw_of(pose.get("orientation") or {}))
            amcl["force"] = True

    def nomotion(_args: dict) -> dict:
        amcl["force"] = True
        return {}

    def set_map(args: dict) -> dict:
        args = args or {}
        grid = args.get("map") or {}
        info = grid.get("info") or {}
        if info.get("width") and grid.get("data") is not None:
            origin = (info.get("origin") or {})
            op = origin.get("position") or {}
            with lock:
                state["map"] = {"resolution": float(info.get("resolution", 0.05)),
                                "width": int(info["width"]), "height": int(info["height"]),
                                "origin": [float(op.get("x", 0.0)), float(op.get("y", 0.0)),
                                           _yaw_of(origin.get("orientation") or {})],
                                "data": list(grid["data"])}
        on_initialpose(args.get("initial_pose") or {})
        return {"success": True}

    # -- move_base ------------------------------------------------------------------------------
    nav = {"goal": None, "finished": [], "footprint": None}
    footprint = [tuple(p) for p in params["/move_base/global_costmap/footprint"]]
    padding = float(params.get("/move_base/global_costmap/footprint_padding", 0.0))
    tp = "/move_base/TrajectoryPlannerROS/"
    max_vel_x = float(params[tp + "max_vel_x"])
    max_vel_theta = float(params[tp + "max_vel_theta"])
    in_place = float(params[tp + "min_in_place_vel_theta"])
    xy_tol = float(params[tp + "xy_goal_tolerance"])
    yaw_tol = float(params[tp + "yaw_goal_tolerance"])

    def goal_status(goal: dict, status: int, text: str = "") -> dict:
        return {"goal_id": {"stamp": _time(goal["stamp"]), "id": goal["id"]},
                "status": status, "text": text}

    def finish(status: int, text: str) -> None:
        """End the active goal: its result, a zero velocity, and its status kept a while."""
        goal = nav["goal"]
        if goal is None:
            return
        nav["goal"] = None
        now = state["now"]
        nav["finished"].append((now, goal_status(goal, status, text)))
        bus.publish(TOPIC_RESULT, {"header": _stamp(seq("result"), "", now),
                                   "status": goal_status(goal, status, text), "result": {}},
                    TYPE_ACTION_RESULT, node=NODE_MOVE_BASE)
        send_velocity(0.0, 0.0)

    def send_velocity(vx: float, wz: float) -> None:
        twist = {"linear": {"x": vx, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": wz}}
        bus.publish(TOPIC_CMD_VEL, twist, TYPE_TWIST, node=NODE_MOVE_BASE)
        drive(twist)

    def accept(goal_id: str, stamp: float, target: dict) -> None:
        pose = target.get("pose") or {}
        position = pose.get("position") or {}
        goal = {"id": goal_id or f"{bus.node(NODE_MOVE_BASE)}-{seq('goal_id')}-{stamp:.9f}",
                "stamp": stamp, "target": (float(position.get("x", 0.0)),
                                           float(position.get("y", 0.0)),
                                           _yaw_of(pose.get("orientation") or {})),
                "progress": None, "since": state["now"]}
        with lock:
            if nav["goal"] is not None:
                finish(PREEMPTED, "This goal was canceled because another goal was received")
            nav["goal"] = goal
        bus.publish(TOPIC_CURRENT_GOAL, {"header": _stamp(seq("current_goal"), f_map,
                                                          state["now"]),
                                         "pose": _pose(*goal["target"])},
                    TYPE_POSE_STAMPED, node=NODE_MOVE_BASE)

    def on_goal(msg: dict) -> None:
        goal_id = (msg.get("goal_id") or {})
        target = ((msg.get("goal") or {}).get("target_pose")) or {}
        accept(str(goal_id.get("id", "") or ""), state["now"], target)

    def on_simple_goal(msg: dict) -> None:
        accept("", state["now"], msg)

    def on_cancel(msg: dict) -> None:
        wanted = str((msg or {}).get("id", "") or "")
        stamp = (msg or {}).get("stamp") or {}
        cancel_all = not wanted and not (stamp.get("secs") or stamp.get("nsecs"))
        with lock:
            goal = nav["goal"]
            if goal is not None and (cancel_all or goal["id"] == wanted or
                                     (not wanted and goal["stamp"] <=
                                      float(stamp.get("secs", 0)) +
                                      float(stamp.get("nsecs", 0)) * 1e-9)):
                finish(PREEMPTED, "")

    def on_footprint(msg: dict) -> None:
        points = (msg or {}).get("points") or []
        nav["footprint"] = [(float(p.get("x", 0.0)), float(p.get("y", 0.0))) for p in points]

    def make_plan(args: dict) -> dict:
        args = args or {}
        start = (args.get("start") or {}).get("pose") or {}
        goal = (args.get("goal") or {}).get("pose") or {}
        sp, gp = start.get("position") or {}, goal.get("position") or {}
        points = straight_line((float(sp.get("x", 0.0)), float(sp.get("y", 0.0)), 0.0),
                               (float(gp.get("x", 0.0)), float(gp.get("y", 0.0)),
                                _yaw_of(goal.get("orientation") or {})),
                               state["map"]["resolution"])
        return {"plan": _path(0, f_map, state["now"], points)}

    def reconfigure(args: dict) -> dict:
        config = (args or {}).get("config") or {}
        return {"config": {k: list(config.get(k) or []) for k in
                           ("bools", "ints", "strs", "doubles", "groups")}}

    bus.on(TOPIC_INITIALPOSE, on_initialpose, TYPE_POSE_COV, node=NODE_AMCL)
    bus.on(TOPIC_GOAL, on_goal, TYPE_ACTION_GOAL, node=NODE_MOVE_BASE)
    bus.on(TOPIC_CANCEL, on_cancel, TYPE_GOAL_ID, node=NODE_MOVE_BASE)
    bus.on(TOPIC_SIMPLE_GOAL, on_simple_goal, TYPE_POSE_STAMPED, node=NODE_MOVE_BASE)
    bus.on(TOPIC_GLOBAL_FOOTPRINT, on_footprint, TYPE_POLYGON, node=NODE_MOVE_BASE)
    bus.on(TOPIC_LOCAL_FOOTPRINT, on_footprint, TYPE_POLYGON, node=NODE_MOVE_BASE)

    handlers = {
        "/static_map": static_map, "/change_map": change_map,
        "/global_localization": nomotion, "/request_nomotion_update": nomotion,
        "/set_map": set_map, "/move_base/make_plan": make_plan,
        "/move_base/GlobalPlanner/make_plan": make_plan,
        "/move_base/clear_costmaps": lambda _a: {},
    }
    for name, stype, node in SERVICES:
        bus.service(name, handlers.get(name, reconfigure), stype, node=node)

    # -- the loop --------------------------------------------------------------------------
    from ros_surfaces.myagv import _Every

    slack = 0.5 / LOOP_HZ
    clocks = {"status": _Every(5.0, slack), "control": _Every(
        float(params["/move_base/controller_frequency"]), slack),
        "planner": _Every(float(params["/move_base/planner_frequency"]), slack),
        "global": _Every(0.3, slack), "local": _Every(2.0, slack), "amcl_tf": _Every(SCAN_HZ, slack)}

    def map_pose(odom_pose):
        return _compose(amcl["map_odom"], odom_pose)

    def footprint_msg(key: str, pose, stamp: float) -> dict:
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        pts = nav["footprint"] or footprint
        out = []
        for px, py in pts:
            # footprint_padding pushes each corner out along both axes.
            px += math.copysign(padding, px)
            py += math.copysign(padding, py)
            out.append({"x": x + c * px - s * py, "y": y + s * px + c * py, "z": 0.0})
        return {"header": _stamp(seq(key), f_map, stamp), "polygon": {"points": out}}

    def costmap(grid, pose, window: float | None, resolution: float | None) -> dict:
        """The static map, or a rolling window of it around `pose`, as a costmap."""
        if window is None:
            return {"resolution": grid["resolution"], "width": grid["width"],
                    "height": grid["height"], "origin": list(grid["origin"]),
                    "data": list(grid["data"])}
        res = resolution or grid["resolution"]
        n = int(round(window / res))
        ox, oy = pose[0] - window / 2.0, pose[1] - window / 2.0
        gx0, gy0, _ = grid["origin"]
        data = []
        for j in range(n):
            wy = oy + (j + 0.5) * res
            gj = int(math.floor((wy - gy0) / grid["resolution"]))
            for i in range(n):
                wx = ox + (i + 0.5) * res
                gi = int(math.floor((wx - gx0) / grid["resolution"]))
                inside = 0 <= gi < grid["width"] and 0 <= gj < grid["height"]
                data.append(grid["data"][gj * grid["width"] + gi] if inside else -1)
        return {"resolution": res, "width": n, "height": n, "origin": [ox, oy, 0.0],
                "data": data}

    def step(data):
        base_step(data)
        if data is None:
            return
        now = float(getattr(data, "time", 0.0))
        state["now"] = now
        pose4 = base.pose
        odom_pose = (float(pose4[0, 3]), float(pose4[1, 3]),
                     math.atan2(float(pose4[1, 0]), float(pose4[0, 0])))
        with lock:
            reset_to, force = amcl["reset_to"], amcl["force"]
            amcl["reset_to"] = None
        if reset_to is not None:
            amcl["map_odom"] = _compose(reset_to, _inverse(odom_pose))
        current = map_pose(odom_pose)

        # -- amcl: an update when the base has moved enough, or when asked for one ----------
        last = amcl["last_update"]
        moved = last is None or math.hypot(odom_pose[0] - last[0], odom_pose[1] - last[1]) \
            >= update_min_d or abs(_wrap(odom_pose[2] - last[2])) >= update_min_a
        if force or moved:
            amcl["force"] = False
            amcl["last_update"] = odom_pose
            cov = [0.0] * 36
            cov[0] = cov[7] = 0.25
            cov[35] = (math.pi / 12) ** 2
            bus.publish(TOPIC_AMCL_POSE, {"header": _stamp(seq("amcl_pose"), f_map, now),
                                          "pose": {"pose": _pose(*current), "covariance": cov}},
                        TYPE_POSE_COV, latched=True, node=NODE_AMCL)
            rng = __import__("random").Random(seqs.get("amcl_pose", 0))
            cloud = [_pose(current[0] + rng.gauss(0.0, 0.05), current[1] + rng.gauss(0.0, 0.05),
                           current[2] + rng.gauss(0.0, 0.03)) for _ in range(particles)]
            bus.publish(TOPIC_PARTICLECLOUD, {"header": _stamp(seq("particles"), f_map, now),
                                              "poses": cloud},
                        TYPE_POSE_ARRAY, latched=True, node=NODE_AMCL)
        if clocks["amcl_tf"].due(now):
            mx, my, myaw = amcl["map_odom"]
            bus.publish(TOPIC_TF, tf_message(
                [(f_map, f_odom, (mx, my, 0.0), (math.cos(myaw / 2), 0.0, 0.0, math.sin(myaw / 2)))],
                stamp_s=now + tolerance, seq=seq("amcl_tf")), TYPE_TF_MESSAGE, node=NODE_AMCL)

        # -- move_base: status, and the controller while a goal is active ----------------------
        with lock:
            goal = nav["goal"]
        if clocks["control"].due(now) and goal is not None:
            tx, ty, tyaw = goal["target"]
            dx, dy = tx - current[0], ty - current[1]
            dist = math.hypot(dx, dy)
            heading_err = _wrap(math.atan2(dy, dx) - current[2])
            yaw_err = _wrap(tyaw - current[2])
            if dist <= xy_tol and abs(yaw_err) <= yaw_tol:
                with lock:
                    finish(SUCCEEDED, "Goal reached.")
            else:
                if dist > xy_tol and abs(heading_err) > HEADING_WINDOW_RAD:
                    vx, wz = 0.0, math.copysign(in_place, heading_err)
                elif dist > xy_tol:
                    vx = min(max_vel_x, dist)
                    wz = max(-max_vel_theta, min(max_vel_theta, heading_err))
                else:
                    vx, wz = 0.0, math.copysign(in_place, yaw_err)
                send_velocity(vx, wz)
                progress = goal["progress"]
                if progress is None or dist < progress[0] - STUCK_PROGRESS_M or \
                        abs(yaw_err) < progress[1] - STUCK_PROGRESS_RAD:
                    goal["progress"], goal["since"] = (dist, abs(yaw_err)), now
                elif now - goal["since"] > STUCK_S:
                    with lock:
                        finish(ABORTED, "Failed to find a valid control. Even after executing "
                                        "recovery behaviors.")
                else:
                    bus.publish(TOPIC_FEEDBACK, {
                        "header": _stamp(seq("feedback"), "", now),
                        "status": goal_status(goal, ACTIVE),
                        "feedback": {"base_position": {
                            "header": _stamp(0, f_map, now), "pose": _pose(*current)}}},
                        TYPE_ACTION_FEEDBACK, node=NODE_MOVE_BASE)
                    plan = straight_line(current, goal["target"], state["map"]["resolution"])
                    bus.publish(TOPIC_TP_GLOBAL_PLAN, _path(seq("tp_global"), f_map, now, plan),
                                TYPE_PATH, node=NODE_MOVE_BASE)
                    ahead = [(current[0] + math.cos(current[2]) * vx * t,
                              current[1] + math.sin(current[2]) * vx * t, current[2] + wz * t)
                             for t in (0.0, 0.2, 0.4, 0.6)]
                    bus.publish(TOPIC_TP_LOCAL_PLAN, _path(seq("tp_local"), f_map, now, ahead),
                                TYPE_PATH, node=NODE_MOVE_BASE)
                    bus.publish(TOPIC_COST_CLOUD, {
                        "header": _stamp(seq("cost_cloud"), f_map, now), "height": 1, "width": 0,
                        "fields": [], "is_bigendian": False, "point_step": 0, "row_step": 0,
                        "data": "", "is_dense": True}, TYPE_POINT_CLOUD2, node=NODE_MOVE_BASE)
        if clocks["planner"].due(now) and goal is not None and nav["goal"] is not None:
            plan = straight_line(current, goal["target"], state["map"]["resolution"])
            bus.publish(TOPIC_PLAN, _path(seq("plan"), f_map, now, plan), TYPE_PATH,
                        node=NODE_MOVE_BASE)
            grid = state["map"]
            potential = dict(grid, data=[0] * (grid["width"] * grid["height"]))
            bus.publish(TOPIC_POTENTIAL, _grid(seq("potential"), f_map, now, potential, now),
                        TYPE_GRID, node=NODE_MOVE_BASE)
        if clocks["status"].due(now):
            with lock:
                nav["finished"] = [(t, s) for t, s in nav["finished"]
                                   if now - t <= STATUS_LIST_TIMEOUT_S]
                status_list = [s for _t, s in nav["finished"]]
                if nav["goal"] is not None:
                    status_list.append(goal_status(nav["goal"], ACTIVE))
            bus.publish(TOPIC_STATUS, {"header": _stamp(seq("status"), "", now),
                                       "status_list": status_list},
                        TYPE_STATUS_ARRAY, node=NODE_MOVE_BASE)
        if clocks["global"].due(now):
            grid = costmap(state["map"], current, None, None)
            bus.publish(TOPIC_GLOBAL_COSTMAP, _grid(seq("global"), f_map, now, grid, now),
                        TYPE_GRID, node=NODE_MOVE_BASE)
            bus.publish(TOPIC_GLOBAL_FOOTPRINT, footprint_msg("global_fp", current, now),
                        TYPE_POLYGON_STAMPED, node=NODE_MOVE_BASE)
            if bus.has_subscribers(TOPIC_CLEARING_ENDPOINTS):
                bus.publish(TOPIC_CLEARING_ENDPOINTS, {
                    "header": _stamp(seq("clearing"), f_map, now), "points": [],
                    "channels": []}, TYPE_POINT_CLOUD, node=NODE_MOVE_BASE)
        if clocks["local"].due(now):
            window = float(params.get("/move_base/local_costmap/width", 3.0))
            res = float(params.get("/move_base/local_costmap/resolution", 0.02))
            grid = costmap(state["map"], current, window, res)
            bus.publish(TOPIC_LOCAL_COSTMAP, _grid(seq("local"), f_map, now, grid, now),
                        TYPE_GRID, node=NODE_MOVE_BASE)
            bus.publish(TOPIC_LOCAL_FOOTPRINT, footprint_msg("local_fp", current, now),
                        TYPE_POLYGON_STAMPED, node=NODE_MOVE_BASE)

    step.rate_hz = base_step.rate_hz
    print(f"myAGV + myCobot 280 navigation under namespace {bus.ns or '<bare>'}",
          file=sys.stderr)
    return step
