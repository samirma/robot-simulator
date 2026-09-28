"""Workspace spec §1: the console's copies of the contract facts equal the simulator's.

Console spec §4 (Contract parity): every console-owned topic, service, action, type,
namespace composition rule, joint definition, camera calibration, task constant, robot id
and periodic rate is compared here with the simulator's normative contract modules
(`simulator/shared/ros_surfaces/*`, `contracts/namespace.py`, `contracts/tf.py`,
`tasks/apple_on_plate.py`) and with the ROS files in `robots_specs/` that those modules
transcribe (`test_simulator_transcription.py` holds the second link).

Both projects are read as data (`_workspace.py`), never imported, so this runs on any
Python >= 3.11 with no venv. These are the only tests that need both source trees; the
console's own suite needs neither.
"""

from __future__ import annotations

import json
import math
import re
import unittest
import xml.etree.ElementTree as ET

from _workspace import CONSOLE, SPECS, console, robots_yml, ros_file, simulator

# ---------------------------------------------------------------------- the modules

C_TOPICS = console("robot_console.topics")
C_X3 = console("robot_console.x3plus_topics")
C_COMPOSITE = console("robot_console.composite_topics")
C_AINEX = console("robot_console.ainex_topics")
C_ROBOTS = console("robot_console.robots")
C_DISCOVERY = console("robot_console.discovery")
C_FLEET = console("robot_console.fleet")
C_SCAN = console("robot_console.slam.scan")
C_ARM = console("robot_console.arm.ros_settings")
C_KIN = console("robot_console.arm.kinematics")
C_TASK = console("robot_console.arm.task")
C_VISION = console("robot_console.arm.vision_success")

S_MYAGV = simulator("ros_surfaces/myagv.py")
S_X3 = simulator("ros_surfaces/rosmaster_x3_plus.py")
S_COMPOSITE = simulator("ros_surfaces/myagv_mycobot280.py")
S_AINEX = simulator("ros_surfaces/ainex/topics.py")
S_SERVOS = simulator("ros_surfaces/ainex/servos.py")
S_SO101 = simulator("ros_surfaces/so101.py")
S_NAMESPACE = simulator("contracts/namespace.py")
S_TF = simulator("contracts/tf.py")
S_TASK = simulator("tasks/apple_on_plate.py")

MYAGV_ROS = ros_file(SPECS / "myagv" / "ros.yml")
SO101_ROS = ros_file(SPECS / "so101" / "ros2.yml")
AINEX_ROS = ros_file(SPECS / "ainex" / "ros.yml")
X3_ROS = ros_file(SPECS / "rosmaster_x3_plus" / "ros.yml")
COMPOSITE_ROS = ros_file(SPECS / "myagv_mycobot280" / "ros.yml")
#: The composite's whole interface: its ROS file `extends` the myAGV's.
COMPOSITE_ALL = {"topics": MYAGV_ROS["topics"] + COMPOSITE_ROS["topics"],
                 "services": MYAGV_ROS["services"] + COMPOSITE_ROS["services"]}


def _rows(ros: dict, section: str) -> dict[str, str]:
    """`{name: type}` of one section of a ROS file (the first row of a repeated name)."""
    out: dict[str, str] = {}
    for row in ros.get(section) or ():
        out.setdefault(row["name"], row["type"])
    return out


def _directed(ros: dict, direction: str) -> dict[str, str]:
    return {r["name"]: r["type"] for r in ros["topics"] if r.get("direction") == direction}


# ---------------------------------------------------------------------- namespacing


class NamespaceComposition(unittest.TestCase):
    """The two copies of the composition rule, compared by behaviour."""

    NAMESPACES = ("so101", "myagv", "ainex", "", "robot_2", "/lead/", "a/b")
    TOPICS = ("/cmd_vel", "cmd_vel", "/odom", "/scan", "/joint_states", "/reset",
              "/joint_trajectory_controller/joint_trajectory", "/gripper_controller/gripper_cmd",
              "/wrist/image_raw/compressed", "/walking/set_param", "/so101/joint_states",
              "/so101", "/so1010/x", "/overhead/color/compressed")

    def test_both_sides_compose_every_name_the_same_way(self) -> None:
        mine, theirs = C_TOPICS["namespaced"], S_NAMESPACE["ns_topic"]
        for namespace in self.NAMESPACES:
            for topic in self.TOPICS:
                self.assertEqual(mine(topic, namespace), theirs(namespace, topic),
                                 (namespace, topic))

    def test_both_sides_normalise_a_relative_name_the_same_way(self) -> None:
        for topic in self.TOPICS:
            self.assertEqual(C_TOPICS["normalise"](topic), S_NAMESPACE["normalise"](topic))

    def test_the_empty_namespace_is_the_bare_contract(self) -> None:
        for topic in ("/joint_states", "/cmd_vel", "/odom"):
            self.assertEqual(C_TOPICS["namespaced"](topic, ""), topic)
            self.assertEqual(S_NAMESPACE["ns_topic"]("", topic), topic)

    def test_frames_take_no_leading_slash(self) -> None:
        """Only the simulator composes frames; the console reads them against this."""
        frame = S_NAMESPACE["ns_frame"]
        self.assertEqual(frame("myagv", C_TOPICS["FRAME_BASE"]), "myagv/base_footprint")
        self.assertEqual(frame("", C_TOPICS["FRAME_ODOM"]), "odom")
        self.assertEqual(frame("so101", ""), "")

    def test_the_rig_has_its_own_namespace_on_both_sides(self) -> None:
        self.assertEqual(C_ARM["SCENE_NAMESPACE"], S_SO101["SCENE_NAMESPACE"])
        self.assertEqual(C_DISCOVERY["RIG_NAMESPACE"], S_SO101["SCENE_NAMESPACE"])
        self.assertEqual(C_ARM["SCENE_ROOT_FRAME"], S_SO101["SCENE_ROOT_FRAME"])


# ---------------------------------------------------------------------- the myAGV


class MyAGV(unittest.TestCase):
    def test_every_topic_and_type(self) -> None:
        sim = {name: kind for name, kind, *_ in S_MYAGV["TOPICS"]}
        self.assertEqual(C_TOPICS["CONTRACT_TOPICS"], sim)
        self.assertEqual(C_TOPICS["CONTRACT_TOPICS"], _rows(MYAGV_ROS, "topics"))

    def test_every_service_and_type(self) -> None:
        sim = {name: kind for name, kind, _node in S_MYAGV["SERVICES"]}
        self.assertEqual(C_TOPICS["CONTRACT_SERVICES"], sim)
        self.assertEqual(C_TOPICS["CONTRACT_SERVICES"], _rows(MYAGV_ROS, "services"))

    def test_the_named_constants(self) -> None:
        for name in ("TOPIC_CMD_VEL", "TOPIC_ODOM", "TOPIC_IMU", "TOPIC_SCAN", "TOPIC_CAMERA",
                     "TOPIC_TF", "TOPIC_TF_STATIC", "TOPIC_JOINT_STATES", "TYPE_TWIST",
                     "TYPE_ODOM", "TYPE_TF_MESSAGE", "PARAM_ROBOT_DESCRIPTION", "FRAME_ODOM",
                     "FRAME_BASE", "FRAME_LASER", "FRAME_CAMERA", "FRAME_IMU"):
            self.assertEqual(C_TOPICS[name], S_MYAGV[name], name)

    def test_the_command_limit_and_the_stop_command(self) -> None:
        self.assertEqual(C_TOPICS["CMD_VEL_LIMIT"], S_MYAGV["CMD_VEL_LIMIT"])
        stop = S_MYAGV["STOP_COMMAND"]
        self.assertTrue(all(v == 0.0 for part in stop.values() for v in part.values()))
        self.assertIn("zero geometry_msgs/Twist on /cmd_vel", MYAGV_ROS["stop_command"])
        self.assertIn("zero geometry_msgs/Twist on /cmd_vel",
                      C_ROBOTS["STOP_COMMANDS"][C_ROBOTS["MYAGV"]])

    def test_the_lidar_mount_and_ranges_the_mapper_projects_with(self) -> None:
        parent, child, (x, y, _z), (yaw, _p, _r) = \
            S_MYAGV["STATIC_TRANSFORMS"][S_MYAGV["NODE_BASE2LASER"]]
        self.assertEqual((parent, child), (C_TOPICS["FRAME_BASE"], C_TOPICS["FRAME_LASER"]))
        self.assertEqual((x, y), (C_SCAN["LASER_OFFSET_X"], C_SCAN["LASER_OFFSET_Y"]))
        self.assertTrue(math.isclose(yaw, C_SCAN["LASER_YAW"]))
        self.assertEqual((S_MYAGV["SCAN_RANGE_MIN"], S_MYAGV["SCAN_RANGE_MAX"]),
                         (C_SCAN["DEFAULT_RANGE_MIN"], C_SCAN["DEFAULT_RANGE_MAX"]))


# ---------------------------------------------------------------------- the ROSMASTER X3 PLUS


class RosmasterX3Plus(unittest.TestCase):
    def test_every_name_and_type_the_console_uses(self) -> None:
        sim = {name: kind for name, kind, *_ in S_X3["TOPICS"]}
        files = _rows(X3_ROS, "topics")
        for name, kind in C_X3["CONTRACT_TOPICS"].items():
            self.assertEqual(sim[name], kind, name)
            self.assertEqual(files[name], kind, name)
        for name in ("TOPIC_CMD_VEL", "TOPIC_ODOM", "TOPIC_SCAN", "TOPIC_TARGET_ANGLE",
                     "TYPE_TWIST", "TYPE_ODOM", "TYPE_LASER_SCAN", "TYPE_ARM_JOINT"):
            self.assertEqual(C_X3[name], S_X3[name], name)
        self.assertEqual(C_X3["TOPIC_CAMERA"], S_X3["TOPIC_RGB_IMAGE"])
        self.assertEqual(C_X3["TYPE_IMAGE"], S_X3["TYPE_IMAGE"])

    def test_what_the_console_sends_the_driver_subscribes_with_that_type(self) -> None:
        driver = {n: ty for n, ty, d, node, _r in S_X3["TOPICS"] if d == "in"
                  and node == S_X3["NODE_DRIVER"]}
        self.assertEqual(driver[C_X3["TOPIC_CMD_VEL"]], C_X3["TYPE_TWIST"])
        self.assertEqual(_directed(X3_ROS, "in")[C_X3["TOPIC_CMD_VEL"]], C_X3["TYPE_TWIST"])

    def test_the_command_range_the_teleop_cap_and_the_stop_command(self) -> None:
        self.assertEqual(tuple(C_X3["CMD_VEL_LIMITS"]), tuple(S_X3["CMD_VEL_LIMITS"]))
        self.assertEqual(C_X3["SPEED_MAX"], S_X3["CMD_VEL_LIMITS"][0])
        self.assertEqual(C_X3["TURN_MAX"], S_X3["CMD_VEL_LIMITS"][2])
        self.assertIn("v_x, v_y [-0.7, 0.7] m/s and v_z [-3.2, 3.2] rad/s",
                      next(r["description"] for r in X3_ROS["topics"]
                           if r["name"] == "/cmd_vel" and r["direction"] == "in"))
        stop = S_X3["STOP_COMMAND"]
        self.assertTrue(all(v == 0.0 for part in stop.values() for v in part.values()))
        self.assertIn("/cmd_vel geometry_msgs/Twist", X3_ROS["stop_command"])
        self.assertIn("zero geometry_msgs/Twist on /cmd_vel",
                      C_ROBOTS["STOP_COMMANDS"][C_ROBOTS["ROSMASTER_X3_PLUS"]])


# ---------------------------------------------------------------------- the myAGV + myCobot 280


class MyAGVMyCobot280(unittest.TestCase):
    def test_it_extends_the_myagv_and_the_console_drives_it_with_the_myagvs_names(self) -> None:
        self.assertEqual(COMPOSITE_ROS["extends"], "myagv/ros.yml")
        files = _rows(COMPOSITE_ALL, "topics")
        for name, kind in C_TOPICS["CONTRACT_TOPICS"].items():
            self.assertEqual(files[name], kind, name)

    def test_the_navigation_names_the_console_uses(self) -> None:
        sim = {(n, d): ty for n, ty, d, _node, _r in S_COMPOSITE["TOPICS"]}
        self.assertEqual(C_COMPOSITE["TOPIC_GOAL"], S_COMPOSITE["TOPIC_GOAL"])
        self.assertEqual(C_COMPOSITE["TOPIC_CANCEL"], S_COMPOSITE["TOPIC_CANCEL"])
        self.assertEqual(sim[(C_COMPOSITE["TOPIC_GOAL"], "in")], C_COMPOSITE["TYPE_GOAL"])
        self.assertEqual(sim[(C_COMPOSITE["TOPIC_CANCEL"], "in")], C_COMPOSITE["TYPE_GOAL_ID"])
        files = _directed(COMPOSITE_ROS, "in")
        self.assertEqual(files[C_COMPOSITE["TOPIC_GOAL"]], C_COMPOSITE["TYPE_GOAL"])
        self.assertEqual(files[C_COMPOSITE["TOPIC_CANCEL"]], C_COMPOSITE["TYPE_GOAL_ID"])

    def test_the_stop_command_cancels_navigation_then_sends_a_zero_twist(self) -> None:
        self.assertEqual(C_COMPOSITE["CANCEL_ALL"], S_COMPOSITE["STOP_CANCEL"])
        self.assertEqual(S_COMPOSITE["STOP_CANCEL"]["id"], "")
        text = COMPOSITE_ROS["stop_command"]
        self.assertIn("zero geometry_msgs/Twist on /cmd_vel", text)
        self.assertIn("first cancel it", text)
        self.assertIn("actionlib_msgs/GoalID on /move_base/cancel (empty id cancels all)", text)
        mine = C_ROBOTS["STOP_COMMANDS"][C_ROBOTS["MYAGV_MYCOBOT280"]]
        self.assertLess(mine.index("/move_base/cancel"), mine.index("zero geometry_msgs/Twist"))


# ---------------------------------------------------------------------- the AiNex


class AiNex(unittest.TestCase):
    NAMES = ("TOPIC_SET_WALKING_PARAM", "TOPIC_APP_ACTION", "TOPIC_HEAD_PAN", "TOPIC_HEAD_TILT",
             "TOPIC_IS_WALKING", "TOPIC_IMU", "TOPIC_CAMERA", "SRV_WALKING_COMMAND",
             "SRV_IS_WALKING", "SRV_BUS_SERVO_GET", "TYPE_WALKING_PARAM", "TYPE_HEAD_STATE",
             "TYPE_STRING", "TYPE_BOOL", "TYPE_IMU", "TYPE_COMPRESSED_IMAGE",
             "SRV_TYPE_SET_WALKING_COMMAND", "SRV_TYPE_GET_WALKING_STATE",
             "SRV_TYPE_GET_BUS_SERVOS_POSITION")

    def test_the_named_constants(self) -> None:
        for name in self.NAMES:
            self.assertEqual(C_AINEX[name], S_AINEX[name], name)

    def test_what_the_console_sends_the_robot_subscribes_with_that_type(self) -> None:
        for sub in ({t.name: t.type for t in S_AINEX["TOPICS"] if t.direction == "in"},
                    _directed(AINEX_ROS, "in")):
            self.assertEqual(sub[C_AINEX["TOPIC_SET_WALKING_PARAM"]], C_AINEX["TYPE_WALKING_PARAM"])
            self.assertEqual(sub[C_AINEX["TOPIC_APP_ACTION"]], C_AINEX["TYPE_STRING"])
            self.assertEqual(sub[C_AINEX["TOPIC_HEAD_PAN"]], C_AINEX["TYPE_HEAD_STATE"])
            self.assertEqual(sub[C_AINEX["TOPIC_HEAD_TILT"]], C_AINEX["TYPE_HEAD_STATE"])

    def test_what_the_console_reads_the_robot_publishes_with_that_type(self) -> None:
        for pub in ({t.name: t.type for t in S_AINEX["TOPICS"] if t.direction == "out"},
                    _directed(AINEX_ROS, "out")):
            self.assertEqual(pub[C_AINEX["TOPIC_IS_WALKING"]], C_AINEX["TYPE_BOOL"])
            self.assertEqual(pub[C_AINEX["TOPIC_IMU"]], C_AINEX["TYPE_IMU"])
            self.assertEqual(pub[C_AINEX["TOPIC_CAMERA"]], C_AINEX["TYPE_COMPRESSED_IMAGE"])

    def test_every_service_the_console_calls(self) -> None:
        for srv in ({s.name: s.type for s in S_AINEX["SERVICES"]}, _rows(AINEX_ROS, "services")):
            self.assertEqual(srv[C_AINEX["SRV_WALKING_COMMAND"]],
                             C_AINEX["SRV_TYPE_SET_WALKING_COMMAND"])
            self.assertEqual(srv[C_AINEX["SRV_IS_WALKING"]], C_AINEX["SRV_TYPE_GET_WALKING_STATE"])
            self.assertEqual(srv[C_AINEX["SRV_BUS_SERVO_GET"]],
                             C_AINEX["SRV_TYPE_GET_BUS_SERVOS_POSITION"])

    def test_the_contract_topics_are_names_the_robot_presents(self) -> None:
        presented = {t.name for t in S_AINEX["TOPICS"]}
        self.assertLessEqual(set(C_AINEX["CONTRACT_TOPICS"]), presented)
        self.assertLessEqual(set(C_AINEX["CONTRACT_TOPICS"]), set(_rows(AINEX_ROS, "topics")))
        for absent in ("/scan", "/joint_states", "/tf", "/tf_static", "/cmd_vel", "/odom"):
            self.assertNotIn(absent, presented)

    def test_the_walking_commands(self) -> None:
        self.assertEqual(C_AINEX["WALKING_COMMANDS"], S_AINEX["WALKING_COMMANDS"])
        for command in ("enable_control", "enable", "start", "stop"):
            self.assertIn(command, C_AINEX["WALKING_COMMANDS"])
        self.assertIn("'enable_control', then with 'stop'",
                      C_ROBOTS["STOP_COMMANDS"][C_ROBOTS["AINEX"]])

    def test_the_joint_table_servo_scale_and_head_limits(self) -> None:
        self.assertEqual(C_AINEX["JOINT_NAMES"], S_AINEX["JOINT_NAMES"])
        self.assertEqual(list(C_AINEX["JOINT_NAMES"]), AINEX_ROS["joints"])
        servos = S_SERVOS["SERVOS"]
        for joint in C_AINEX["JOINT_NAMES"]:
            self.assertEqual(C_AINEX["servo_id"](joint), servos[joint][0], joint)
        self.assertTrue(math.isclose(C_AINEX["SERVO_TICKS_PER_RADIAN"],
                                     S_SERVOS["TICKS_PER_RADIAN"], rel_tol=1e-12))
        for head in ("head_pan", "head_tilt"):
            self.assertEqual(servos[head][1:], (C_AINEX["HEAD_SERVO_CENTRE"], False))
        limits = S_SERVOS["joint_limits"]
        self.assertEqual(limits("head_pan"), (-C_AINEX["HEAD_PAN_LIMIT"], C_AINEX["HEAD_PAN_LIMIT"]))
        self.assertEqual(limits("head_tilt"),
                         (-C_AINEX["HEAD_TILT_LIMIT"], C_AINEX["HEAD_TILT_LIMIT"]))


# ---------------------------------------------------------------------- the SO-101 and the rig


class SO101(unittest.TestCase):
    def test_the_named_topics_action_and_reset(self) -> None:
        pairs = {
            "ARM_COMMAND_TOPIC": "TOPIC_ARM_COMMAND", "JOINT_STATES_TOPIC": "TOPIC_JOINT_STATES",
            "GRIPPER_ACTION": "ACTION_GRIPPER_COMMAND", "WRIST_CAMERA_TOPIC": "TOPIC_WRIST_COMPRESSED",
            "RESET_SERVICE": "SERVICE_RESET", "RESET_SERVICE_TYPE": "SRV_TYPE_TRIGGER",
            "TF_TOPIC": "TOPIC_TF", "TF_STATIC_TOPIC": "TOPIC_TF_STATIC",
        }
        for mine, theirs in pairs.items():
            self.assertEqual(C_ARM[mine], S_SO101[theirs], mine)

    def test_every_typed_name_the_arm_task_uses(self) -> None:
        arm = C_ARM["arm_interface"]("")
        topics, actions = S_SO101["TOPICS"], S_SO101["ACTIONS"]
        ros_topics, ros_actions = _rows(SO101_ROS, "topics"), _rows(SO101_ROS, "actions")
        for topic, kind in arm["topics"].items():
            self.assertEqual(topics[topic][0], kind, topic)
            self.assertEqual(ros_topics[topic], kind, topic)
        for action, kind in arm["actions"].items():
            self.assertEqual(actions[action][0], kind, action)
            self.assertEqual(ros_actions[action], kind, action)
        self.assertEqual(arm["services"], {S_SO101["SERVICE_RESET"]: S_SO101["SRV_TYPE_TRIGGER"]})

    def test_the_composed_interface_is_the_simulators_composition(self) -> None:
        compose = S_NAMESPACE["ns_topic"]
        for namespace in ("so101", "", "arm_2"):
            arm = C_ARM["arm_interface"](namespace)
            expected = {compose(namespace, t): S_SO101["TOPICS"][t][0] for t in
                        (S_SO101["TOPIC_JOINT_STATES"], S_SO101["TOPIC_ARM_COMMAND"],
                         S_SO101["TOPIC_WRIST_COMPRESSED"], S_SO101["TOPIC_TF"],
                         S_SO101["TOPIC_TF_STATIC"])}
            self.assertEqual(arm["topics"], expected)
            self.assertEqual(C_ARM["namespaced_reset"](namespace),
                             compose(namespace, S_SO101["SERVICE_RESET"]))

    def test_the_rig_interface(self) -> None:
        compose, ns = S_NAMESPACE["ns_topic"], S_SO101["SCENE_NAMESPACE"]
        published = {compose(ns, t): S_SO101["TYPE_SCENE_IMAGE"]
                     for t in S_SO101["SCENE_CAMERA_TOPICS"]}
        published.update({compose(ns, t): S_SO101["TYPE_SCENE_CAMERA_INFO"]
                          for t in S_SO101["SCENE_CAMERA_INFO_TOPICS"].values()})
        published[compose(ns, S_SO101["SCENE_TF_STATIC"])] = C_ARM["TF_TYPE"]
        self.assertEqual(C_ARM["rig_interface"](), published)
        # View by view: the image topic and its camera_info topic belong together.
        for image, (name, *_size) in S_SO101["SCENE_CAMERA_TOPICS"].items():
            self.assertEqual(C_ARM["CAMERA_SPECS"][name][0], image, name)
            self.assertEqual(C_ARM["SCENE_CAMERA_INFO_TOPICS"][name],
                             S_SO101["SCENE_CAMERA_INFO_TOPICS"][image], name)

    def test_the_camera_topics_and_sizes(self) -> None:
        sim = {topic: tuple(size) for topic, (_name, *size) in
               S_SO101["SCENE_CAMERA_TOPICS"].items()}
        sim[S_SO101["TOPIC_WRIST_COMPRESSED"]] = tuple(S_SO101["WRIST_SIZE"])
        mine = {topic: (w, h) for topic, w, h in C_ARM["CAMERA_SPECS"].values()}
        self.assertEqual(mine, sim)

    def test_no_task_verdict_on_the_wire(self) -> None:
        self.assertNotIn("TOPIC_TASK_SUCCESS", S_SO101)
        for name in S_SO101["TOPICS"]:
            self.assertNotRegex(name, r"success|verdict|task")

    def test_the_joints_and_their_order(self) -> None:
        self.assertEqual(C_KIN["ARM_JOINTS"], S_SO101["ARM_JOINTS"])
        self.assertEqual(C_KIN["GRIPPER_JOINT"], S_SO101["GRIPPER_JOINT"])
        self.assertEqual(C_KIN["JOINT_ORDER"], S_SO101["JOINT_ORDER"])
        self.assertEqual(list(C_KIN["JOINT_ORDER"]), SO101_ROS["joints"])

    def test_the_gripper_offset(self) -> None:
        self.assertEqual(C_KIN["GRIPPER_OFFSET_RAD"], S_SO101["GRIPPER_OFFSET_RAD"])
        lo, hi = C_KIN["JOINT_LIMITS"][C_KIN["GRIPPER_JOINT"]]
        self.assertGreaterEqual(lo, S_SO101["GRIPPER_RANGE"][0] - 1e-9)
        self.assertLessEqual(hi, S_SO101["GRIPPER_RANGE"][1] + 1e-9)


class SO101Model(unittest.TestCase):
    """The console's kinematics transcribe the official MJCF in `robots_specs/so101/`."""

    MJCF = SPECS / "so101" / "so101_new_calib.xml"
    CHAIN = ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper")

    @classmethod
    def setUpClass(cls) -> None:
        cls.bodies = {b.get("name"): b for b in ET.parse(cls.MJCF).iter("body")}

    @staticmethod
    def _floats(text: str) -> tuple[float, ...]:
        return tuple(float(v) for v in text.split())

    def assertClose(self, a, b, tol: float = 2e-6) -> None:  # noqa: N802
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertLessEqual(abs(x - y), tol, (a, b))

    def test_every_link_is_the_mjcf_body(self) -> None:
        for (pos, quat), name in zip(C_KIN["_LINKS"], self.CHAIN, strict=True):
            body = self.bodies[name]
            self.assertClose(pos, self._floats(body.get("pos")))
            self.assertClose(quat, self._floats(body.get("quat")))

    def test_the_tool_centre_point_is_the_gripperframe_site(self) -> None:
        site = next(s for s in self.bodies["gripper"].iter("site")
                    if s.get("name") == "gripperframe")
        self.assertClose(C_KIN["_TCP_POS"], self._floats(site.get("pos")))
        self.assertClose(C_KIN["_TCP_QUAT"], self._floats(site.get("quat")))

    def test_every_commanded_range_is_inside_the_joints(self) -> None:
        offset = C_KIN["GRIPPER_OFFSET_RAD"]
        joints = {j.get("name"): self._floats(j.get("range")) for j in ET.parse(self.MJCF).iter("joint")
                  if j.get("name")}
        for name, (lo, hi) in C_KIN["JOINT_LIMITS"].items():
            mlo, mhi = joints[name.removesuffix("_joint")]
            if name == C_KIN["GRIPPER_JOINT"]:
                mlo, mhi = mlo + offset, mhi + offset
            self.assertGreaterEqual(lo, mlo - 1e-5, name)
            self.assertLessEqual(hi, mhi + 1e-5, name)


# ---------------------------------------------------------------------- the task and the rig


class TaskAndRig(unittest.TestCase):
    def test_the_rig_mount_poses_and_intrinsics(self) -> None:
        staged = {name: (tuple(pos), tuple(xy), fovy, tuple(res))
                  for name, pos, xy, fovy, res in S_TASK["SCENE_CAMERAS"]}
        self.assertEqual(C_ARM["SCENE_CAMERAS"], staged)
        self.assertEqual(C_ARM["SCENE_CAMERA_HZ"], S_TASK["SCENE_CAMERA_HZ"])

    def test_the_pass_thresholds(self) -> None:
        pairs = {"MAX_HORIZONTAL_DIST_M": "MAX_HORIZONTAL_DIST", "RESTING_Z_M": "RESTING_Z",
                 "Z_TOLERANCE_M": "Z_TOLERANCE", "MAX_SPEED_MPS": "MAX_SPEED",
                 "HOLD_SECONDS": "SUSTAIN_SECONDS", "APPLE_RADIUS_M": "APPLE_RADIUS",
                 "PLATE_TOP_Z_M": "PLATE_TOP_Z"}
        for mine, theirs in pairs.items():
            self.assertEqual(C_VISION[mine], S_TASK[theirs], mine)

    def test_the_start_pose(self) -> None:
        self.assertEqual(tuple(C_TASK["START_ARM_QPOS"]), tuple(S_TASK["START_ARM_QPOS"]))

    def test_the_task_objects_fit_the_criterion(self) -> None:
        """The plate is wider than the gate by the apple: an apple on it can pass."""
        self.assertTrue(math.isclose(S_TASK["PLATE_RADIUS"] - S_TASK["APPLE_RADIUS"],
                                     C_VISION["MAX_HORIZONTAL_DIST_M"]))


# ---------------------------------------------------------------------- transform trees


class TransformTree(unittest.TestCase):
    def test_the_tf_names_and_both_dialects(self) -> None:
        self.assertEqual((S_TF["TOPIC_TF"], S_TF["TOPIC_TF_STATIC"]),
                         (C_TOPICS["TOPIC_TF"], C_TOPICS["TOPIC_TF_STATIC"]))
        self.assertEqual((S_TF["TOPIC_TF"], S_TF["TOPIC_TF_STATIC"]),
                         (C_ARM["TF_TOPIC"], C_ARM["TF_STATIC_TOPIC"]))
        self.assertEqual(S_TF["PARAM_ROBOT_DESCRIPTION"], C_TOPICS["PARAM_ROBOT_DESCRIPTION"])
        self.assertEqual(S_TF["TYPE_TF_MESSAGE"], C_TOPICS["TYPE_TF_MESSAGE"])
        self.assertEqual(S_TF["TYPE_TF_MESSAGE_ROS2"], C_ARM["TF_TYPE"])


# ---------------------------------------------------------------------- discovery and ids


class DiscoveryAndIds(unittest.TestCase):
    def test_every_signature_is_its_robots_typed_command_topic(self) -> None:
        sim = {"so101": {n: v[0] for n, v in S_SO101["TOPICS"].items()},
               "myagv": {n: k for n, k, *_ in S_MYAGV["TOPICS"]},
               "ainex": {t.name: t.type for t in S_AINEX["TOPICS"]},
               "myagv_mycobot280": {**{n: k for n, k, *_ in S_MYAGV["TOPICS"]},
                                    **{n: k for n, k, d, *_ in S_COMPOSITE["TOPICS"]
                                       if d == "in"}},
               "rosmaster_x3_plus": {n: k for n, k, *_ in S_X3["TOPICS"]}}
        files = {"so101": _directed(SO101_ROS, "in"), "myagv": _directed(MYAGV_ROS, "in"),
                 "ainex": _directed(AINEX_ROS, "in"),
                 "myagv_mycobot280": _directed(COMPOSITE_ALL, "in"),
                 "rosmaster_x3_plus": _directed(X3_ROS, "in")}
        for kind, topic, kind_type in C_DISCOVERY["MEMBER_SIGNATURES"]:
            self.assertEqual(sim[kind][topic], kind_type, kind)
            self.assertEqual(files[kind][topic], kind_type, kind)
        for kind, companions in C_DISCOVERY["COMPANIONS"].items():
            for topic, kind_type in companions.items():
                self.assertEqual(sim[kind][topic], kind_type, (kind, topic))

    def test_the_rig_signature_is_the_rigs_overhead_view(self) -> None:
        compose = S_NAMESPACE["ns_topic"]
        overhead = next(t for t, (name, *_s) in S_SO101["SCENE_CAMERA_TOPICS"].items()
                        if name == "overhead")
        self.assertEqual(C_DISCOVERY["RIG_SIGNATURE"],
                         (compose(S_SO101["SCENE_NAMESPACE"], overhead),
                          S_SO101["TYPE_SCENE_IMAGE"]))

    def test_every_console_robot_id_is_a_simulated_robot_in_robots_yml(self) -> None:
        entries = robots_yml()
        kinds = {"myagv": "mobile_base", "ainex": "humanoid", "so101": "arm",
                 "myagv_mycobot280": "mobile_manipulator",
                 "rosmaster_x3_plus": "mobile_manipulator"}
        ids = {C_ROBOTS["MYAGV"], C_ROBOTS["AINEX"], C_DISCOVERY["SO101"],
               C_ROBOTS["MYAGV_MYCOBOT280"], C_ROBOTS["ROSMASTER_X3_PLUS"]}
        ids |= {k for k, _t, _y in C_DISCOVERY["MEMBER_SIGNATURES"]}
        ids |= {k for k in C_FLEET["PERIODIC"] if k != C_DISCOVERY["RIG_KIND"]}
        self.assertEqual(ids, set(kinds))
        for rid in ids:
            self.assertIn(rid, entries, rid)
            self.assertIs(entries[rid]["simulated"], True, rid)
            self.assertEqual(entries[rid]["kind"], kinds[rid], rid)
        self.assertEqual(entries[C_ROBOTS["MYAGV"]]["name"], "myAGV")
        self.assertEqual(entries[C_ROBOTS["AINEX"]]["name"], "AiNex")
        # Teleop drives every mobile robot robots.yml has: every kind but a fixed arm.
        mobile = {rid for rid, e in entries.items()
                  if e.get("simulated") is True and e.get("kind") != "arm"}
        self.assertEqual(set(C_ROBOTS["TELEOP_ROBOTS"]), mobile)
        self.assertEqual(C_ROBOTS["TELEOP_ROBOTS"][:2], (C_ROBOTS["MYAGV"], C_ROBOTS["AINEX"]))

    def test_the_rig_is_not_a_robot_id(self) -> None:
        self.assertNotIn(C_DISCOVERY["RIG_KIND"], robots_yml())

    def test_each_teleop_robot_has_a_stop_command_in_its_ros_file(self) -> None:
        entries = robots_yml()
        for rid in C_ROBOTS["TELEOP_ROBOTS"]:
            self.assertTrue(ros_file(SPECS / entries[rid]["ros"]).get("stop_command"), rid)
            self.assertIn(rid, C_ROBOTS["STOP_COMMANDS"])


# ---------------------------------------------------------------------- periodic rates


def _declared_rates(ros: dict) -> tuple[dict, dict, set]:
    """`(periodic, aperiodic, optional)` as `fleet.py` models them, from one ROS file (a
    composite's with its base's rows merged in)."""
    periodic: dict[str, tuple[str, float, float]] = {}
    for row in ros["topics"]:
        rate = row.get("rate_hz")
        if row.get("direction") == "out" and isinstance(rate, (int, float)) \
                and not isinstance(rate, bool):
            kind, total, fastest = periodic.get(row["name"], (row["type"], 0.0, 0.0))
            periodic[row["name"]] = (kind, total + rate, max(fastest, rate))
    aperiodic: dict[str, str] = {}
    for row in ros["topics"]:
        if row["name"] not in periodic:
            aperiodic.setdefault(row["name"], row["type"])
    optional = {r["name"] for r in ros["topics"] if r.get("unverified") is True
                and re.search(r"/image_raw/(compressedDepth|theora|zstd)$", r["name"])}
    return periodic, aperiodic, optional


class PeriodicRates(unittest.TestCase):
    FILES = {"so101": SO101_ROS, "myagv": MYAGV_ROS, "ainex": AINEX_ROS,
             "myagv_mycobot280": COMPOSITE_ALL, "rosmaster_x3_plus": X3_ROS}

    def _console(self, kind: str) -> dict[str, tuple[str, float, float]]:
        return {n: (p.type, float(p.hz), float(p.fastest_hz))
                for n, p in C_FLEET["PERIODIC"][kind].items()}

    def test_the_rate_table_is_the_ros_files_both_ways(self) -> None:
        for kind, ros in self.FILES.items():
            periodic, aperiodic, optional = _declared_rates(ros)
            self.assertEqual(self._console(kind), periodic, kind)
            self.assertEqual(C_FLEET["APERIODIC"][kind], aperiodic, kind)
            self.assertEqual(set(C_FLEET["OPTIONAL"].get(kind, ())), optional, kind)
            conditional = {r["name"] for r in ros["topics"] if r.get("active_while")}
            self.assertEqual(set(C_FLEET["CONDITIONAL"].get(kind, ())), conditional, kind)

    def test_the_rate_table_is_the_simulators(self) -> None:
        sim: dict[str, dict[str, float]] = {"so101": {}, "myagv": {}, "ainex": {},
                                            "myagv_mycobot280": {}, "rosmaster_x3_plus": {}}
        for name, (kind, direction, rate, _node) in S_SO101["TOPICS"].items():
            if direction == "out" and isinstance(rate, (int, float)):
                sim["so101"][name] = float(rate)
        for name, kind, direction, _node, rate in S_MYAGV["TOPICS"]:
            if direction == "out" and isinstance(rate, (int, float)):
                sim["myagv"][name] = sim["myagv"].get(name, 0.0) + float(rate)
        sim["ainex"] = {n: float(hz) for n, hz in S_AINEX["RATES_HZ"].items()}
        for name, rate in sim["myagv"].items():
            sim["myagv_mycobot280"][name] = rate
        for kind, module in (("myagv_mycobot280", S_COMPOSITE), ("rosmaster_x3_plus", S_X3)):
            for name, _kind, direction, _node, rate in module["TOPICS"]:
                if direction == "out" and isinstance(rate, (int, float)):
                    sim[kind][name] = sim[kind].get(name, 0.0) + float(rate)
        self.assertEqual(set(C_FLEET["CONDITIONAL"]["myagv_mycobot280"]),
                         set(S_COMPOSITE["CONDITIONAL"]))
        for kind, rates in sim.items():
            served = {n: hz for n, (_t, hz, _f) in self._console(kind).items()
                      if n not in C_FLEET["OPTIONAL"].get(kind, ())}
            self.assertEqual(served.keys(), rates.keys(), kind)
            for name, hz in rates.items():
                self.assertAlmostEqual(served[name], hz, places=6, msg=f"{kind} {name}")

    def test_the_rigs_rate(self) -> None:
        rig = self._console(C_DISCOVERY["RIG_KIND"])
        self.assertEqual({hz for _t, hz, _f in rig.values()}, {S_TASK["SCENE_CAMERA_HZ"]})
        compose, ns = S_NAMESPACE["ns_topic"], S_SO101["SCENE_NAMESPACE"]
        self.assertEqual({compose(ns, n): t for n, (t, _h, _f) in rig.items()},
                         {k: v for k, v in C_ARM["rig_interface"]().items()
                          if not k.endswith("tf_static")})


# ---------------------------------------------------------------------- the camera page


class ViewPage(unittest.TestCase):
    """`live_cameras.html` carries its own copy (one static file, no Python behind it)."""

    @classmethod
    def setUpClass(cls) -> None:
        page = (CONSOLE / "live_cameras.html").read_text(encoding="utf-8")
        cls.contract = json.loads(re.search(
            r'<script type="application/json" id="contract">(.*?)</script>', page, re.S).group(1))

    def test_the_pages_so101_names_are_the_official_interfaces(self) -> None:
        so101 = self.contract["so101"]
        actions, topics = _rows(SO101_ROS, "actions"), _rows(SO101_ROS, "topics")
        for key in ("trajectory_action", "gripper_action"):
            name, kind = so101[key]
            self.assertEqual(actions[name], kind)
            self.assertEqual(S_SO101["ACTIONS"][name][0], kind)
        for key in ("joint_states", "robot_description"):
            name, kind = so101[key]
            self.assertEqual(topics[name], kind)
            self.assertEqual(S_SO101["TOPICS"][name][0], kind)
        self.assertEqual(so101["arm_joints"] + [so101["gripper_joint"]], SO101_ROS["joints"])

    def test_the_pages_ainex_names_are_the_official_interfaces(self) -> None:
        topics = _rows(AINEX_ROS, "topics")
        for key in ("head_pan", "head_tilt", "set_action", "is_walking"):
            name, kind = self.contract["ainex"][key]
            self.assertEqual(topics[name], kind, key)

    def test_the_pages_signatures_are_the_robots(self) -> None:
        files = {"so101": _rows(SO101_ROS, "topics"), "myagv": _rows(MYAGV_ROS, "topics"),
                 "ainex": _rows(AINEX_ROS, "topics"),
                 "myagv_mycobot280": _rows(COMPOSITE_ALL, "topics"),
                 "rosmaster_x3_plus": _rows(X3_ROS, "topics")}
        for kind, topic, kind_type in self.contract["member_signatures"]:
            self.assertEqual(files[kind][topic], kind_type)

    def test_the_fallback_limits_are_the_urdfs(self) -> None:
        """The URDF's arm limits, and the jaw's shifted by the bringup's offset."""
        urdf = (SPECS / "so101" / "so101_new_calib.urdf").read_text(encoding="utf-8")
        offset = S_SO101["GRIPPER_OFFSET_RAD"]
        for joint, (lo, hi) in self.contract["so101"]["fallback_limits"].items():
            m = re.search(rf'<joint name="{joint.removesuffix("_joint")}" type="revolute">.*?'
                          r'lower="([-\d.]+)" upper="([-\d.]+)"', urdf, re.S)
            ulo, uhi = float(m.group(1)), float(m.group(2))
            if joint == "gripper_joint":
                ulo, uhi = ulo + offset, uhi + offset
            self.assertLessEqual(abs(lo - ulo), 1e-5, joint)
            self.assertLessEqual(abs(hi - uhi), 1e-5, joint)


if __name__ == "__main__":
    unittest.main()
