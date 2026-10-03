"""The wire images are built from pinned sources (spec §2.4): every ROS package an image
installs is fixed -- by its exact version, or by a dated snapshot of the ROS repository in
place of the live one -- and the stock nodes the boots run are the releases the robot
specification records."""

import re

import pytest

import registry
import wirecheck
from conftest import SHARED

DOCKER = SHARED / "wire" / "docker"
DISTROS = ("noetic", "humble", "jazzy")
APT_ROS = re.compile(r"\bros-(noetic|humble|jazzy)-([a-z0-9][a-z0-9-]*)(=\S+)?")


def install_lines(text: str) -> str:
    """The `apt-get install` commands of a Dockerfile, continuation lines joined."""
    joined = re.sub(r"\\\n", " ", text)
    return "\n".join(line for line in joined.splitlines() if "apt-get install" in line)


def snapshot(distro: str):
    """The dated snapshot of the ROS repository a distribution's base image installs from."""
    text = (DOCKER / distro / "Dockerfile").read_text()
    m = re.search(rf"http://snapshots\.ros\.org/{distro}/(\d{{4}}-\d{{2}}-\d{{2}})/ubuntu", text)
    return m.group(1) if m else None


@pytest.mark.parametrize("dockerfile", sorted(DOCKER.glob("*/Dockerfile")),
                         ids=lambda p: p.parent.name)
def test_every_ros_package_is_pinned(dockerfile):
    unpinned = []
    for distro, name, pin in APT_ROS.findall(install_lines(dockerfile.read_text())):
        if not pin and snapshot(distro) is None:
            unpinned.append(f"ros-{distro}-{name}")
    assert unpinned == [], f"{dockerfile.parent.name}: {unpinned}"


@pytest.mark.parametrize("distro", DISTROS)
def test_base_images_never_install_from_the_live_repository(distro):
    text = (DOCKER / distro / "Dockerfile").read_text()
    assert re.search(r"^FROM \S+@sha256:[0-9a-f]{64}$", text, re.M), "base image not by digest"
    if snapshot(distro) is not None:
        # the base image's own source for the live repository is taken out, checked at build
        assert re.search(r"! grep -rqs '?packages\.ros\.org", text), distro


#: The stock packages the ROS 2 wires run, by the record source that cites their release.
STOCK = {
    "so101": {"robot_state_publisher": "robot_state_publisher",
              "joint_trajectory_controller": "ros2_controllers",
              "joint_state_broadcaster": "ros2_controllers",
              "parallel_gripper_controller": "ros2_controllers",
              "controller_manager": "ros2_control", "hardware_interface": "ros2_control",
              "diagnostic_updater": "diagnostics", "pal_statistics": "pal_statistics",
              "tf2_ros": "geometry2"},
    "mycobot280": {"robot_state_publisher": "robot_state_publisher",
                   "joint_state_publisher": "joint_state_publisher"},
}


@pytest.mark.parametrize("rid", sorted(STOCK))
def test_stock_nodes_are_the_recorded_releases(rid):
    iface = wirecheck.interface(rid)
    distro = iface["ros_distribution"]
    text = (DOCKER / distro / "Dockerfile").read_text()
    layer = DOCKER / rid / "Dockerfile"
    if layer.is_file():
        text += layer.read_text()
    pins = {name.replace("-", "_"): pin[1:] for d, name, pin in APT_ROS.findall(install_lines(text))
            if d == distro and pin}
    sources = {s["id"]: s for s in iface["sources"]}
    wrong = []
    for package, sid in STOCK[rid].items():
        src = sources[sid]
        release = re.search(r"release (\d+(?:\.\d+)+)", src["role"]).group(1)
        if package in pins:
            if not pins[package].startswith(f"{release}-"):
                wrong.append(f"{package} {pins[package]}, recorded {release} ({sid})")
        elif not (src["revision"] in text and re.search(rf"\b{package}\b", text)
                  and release in text):
            wrong.append(f"{package}: neither pinned to {release} nor built from {sid} "
                         f"{src['revision'][:8]}")
    assert wrong == [], wrong
    assert registry.get(rid).dialect == "ros2"
