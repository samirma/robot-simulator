"""The simulator's contract modules transcribe the ROS files in `robots_specs/`.

Simulator spec §3: `ros_surfaces/myagv.py`, `ros_surfaces/so101.py` and
`ros_surfaces/ainex/topics.py` transcribe the names, types, nodes and periodic rates of
each robot's ROS file. `simulator/shared/contracts/test_fleet.py` checks the same thing
from inside the simulator's venv; this is the workspace half, read as data with the
standard library, so the chain console -> simulator module -> ROS file that
`test_contract_parity.py` relies on is closed without either project installed.
"""

from __future__ import annotations

import unittest

from _workspace import SPECS, ros_file, simulator

S_MYAGV = simulator("ros_surfaces/myagv.py")
S_SO101 = simulator("ros_surfaces/so101.py")
S_AINEX = simulator("ros_surfaces/ainex/topics.py")

MYAGV_ROS = ros_file(SPECS / "myagv" / "ros.yml")
SO101_ROS = ros_file(SPECS / "so101" / "ros2.yml")
AINEX_ROS = ros_file(SPECS / "ainex" / "ros.yml")
S_X3 = simulator("ros_surfaces/rosmaster_x3_plus.py")
S_COMPOSITE = simulator("ros_surfaces/myagv_mycobot280.py")
X3_ROS = ros_file(SPECS / "rosmaster_x3_plus" / "ros.yml")
COMPOSITE_ROS = ros_file(SPECS / "myagv_mycobot280" / "ros.yml")


def _node(name) -> str:
    """A node as the contract modules write it: the file's, without a leading slash."""
    return str(name).lstrip("/")


def _rate(value) -> object:
    """A ROS file's `rate_hz`: a number, or `event` / `latched` as written."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) \
        else value


def _topic_rows(ros: dict) -> list[tuple]:
    return sorted((r["name"], r["type"], r["direction"], r["node"], _rate(r["rate_hz"]))
                  for r in ros["topics"])


class MyAGV(unittest.TestCase):
    def test_every_topic_row(self) -> None:
        sim = sorted((n, t, d, node, _rate(hz)) for n, t, d, node, hz in S_MYAGV["TOPICS"])
        self.assertEqual(sim, _topic_rows(MYAGV_ROS))

    def test_every_service_row(self) -> None:
        sim = sorted(S_MYAGV["SERVICES"])
        self.assertEqual(sim, sorted((r["name"], r["type"], r["node"])
                                     for r in MYAGV_ROS["services"]))

    def test_every_parameter_name(self) -> None:
        sim = set(S_MYAGV["PARAMETERS"]) | {S_MYAGV["PARAM_ROBOT_DESCRIPTION"]}
        self.assertEqual(sim, {r["name"] for r in MYAGV_ROS["parameters"]})


class SO101(unittest.TestCase):
    def test_every_topic_row_but_the_optional_plugin_streams(self) -> None:
        sim = sorted((n, t, d, node, _rate(hz)) for n, (t, d, hz, node) in S_SO101["TOPICS"].items())
        omitted = set(S_SO101["OMITTED_TOPICS"])
        rows = [r for r in _topic_rows(SO101_ROS) if r[0] not in omitted]
        self.assertEqual(sim, rows)
        unverified = {r["name"] for r in SO101_ROS["topics"] if r.get("unverified") is True}
        self.assertLessEqual(omitted, unverified)

    def test_every_service_and_action_row(self) -> None:
        self.assertEqual(sorted((n, t, node) for n, (t, node) in S_SO101["SERVICES"].items()),
                         sorted((r["name"], r["type"], r["node"]) for r in SO101_ROS["services"]))
        self.assertEqual(sorted((n, t, node) for n, (t, node) in S_SO101["ACTIONS"].items()),
                         sorted((r["name"], r["type"], r["node"]) for r in SO101_ROS["actions"]))

    def test_every_parameter(self) -> None:
        """A row may list several names (`brightness, contrast, ...`). A `<component>/<name>`
        row is a ros2_control hardware parameter from the xacro, not a node parameter."""
        rows = {(r["node"], name.strip()) for r in SO101_ROS["parameters"]
                for name in str(r["name"]).split(",")}
        hardware = {(node, name) for node, name in rows if "/" in name}
        self.assertEqual(set(S_SO101["PARAMETERS"]), rows - hardware)

    def test_the_joints(self) -> None:
        self.assertEqual(list(S_SO101["JOINT_ORDER"]), SO101_ROS["joints"])


class AiNex(unittest.TestCase):
    def test_every_topic_row(self) -> None:
        sim = sorted((t.name, t.type, t.direction, t.node, _rate(t.rate_hz))
                     for t in S_AINEX["TOPICS"])
        self.assertEqual(sim, _topic_rows(AINEX_ROS))

    def test_every_service_row(self) -> None:
        self.assertEqual(sorted(tuple(s) for s in S_AINEX["SERVICES"]),
                         sorted((r["name"], r["type"], r["node"]) for r in AINEX_ROS["services"]))

    def test_the_joints(self) -> None:
        self.assertEqual(list(S_AINEX["JOINT_NAMES"]), AINEX_ROS["joints"])


class RosmasterX3Plus(unittest.TestCase):
    def test_every_topic_row(self) -> None:
        sim = sorted((n, t, d, node, _rate(hz)) for n, t, d, node, hz in S_X3["TOPICS"])
        rows = sorted((r["name"], r["type"], r["direction"], _node(r["node"]), _rate(r["rate_hz"]))
                      for r in X3_ROS["topics"])
        self.assertEqual(sim, rows)

    def test_every_service_row(self) -> None:
        self.assertEqual(sorted(S_X3["SERVICES"]),
                         sorted((r["name"], r["type"], _node(r["node"]))
                                for r in X3_ROS["services"]))

    def test_every_parameter_name_and_the_joints(self) -> None:
        self.assertEqual(set(S_X3["PARAMETERS"]) | {S_X3["PARAM_ROBOT_DESCRIPTION"]},
                         {r["name"] for r in X3_ROS["parameters"]})
        self.assertEqual(list(S_X3["JOINTS"]), X3_ROS["joints"])


class MyAGVMyCobot280(unittest.TestCase):
    """The composite's module transcribes only what its file adds to the myAGV's."""

    def test_every_added_topic_row(self) -> None:
        sim = sorted((n, t, d, node, _rate(hz)) for n, t, d, node, hz in S_COMPOSITE["TOPICS"])
        self.assertEqual(sim, _topic_rows(COMPOSITE_ROS))

    def test_the_conditional_rows(self) -> None:
        self.assertEqual(set(S_COMPOSITE["CONDITIONAL"]),
                         {r["name"] for r in COMPOSITE_ROS["topics"] if r.get("active_while")})

    def test_every_added_service_row(self) -> None:
        self.assertEqual(sorted(S_COMPOSITE["SERVICES"]),
                         sorted((r["name"], r["type"], r["node"])
                                for r in COMPOSITE_ROS["services"]))


if __name__ == "__main__":
    unittest.main()
