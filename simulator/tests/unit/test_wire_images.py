"""The wire images are built from pinned sources (spec §2.4): every ROS package an image
installs is fixed -- by its exact version, or by a dated snapshot of the ROS repository in
place of the live one -- every Ubuntu package by a dated snapshot of the Ubuntu archive, and
the stock nodes the boots run are the releases the robot specification records. A changed
image definition is a new image tag, so spawn.sh rebuilds it."""

import re
import shutil
import types

import pytest

import registry
import spawn
import wirecheck
from conftest import SHARED

DOCKER = SHARED / "wire" / "docker"
DISTROS = ("noetic", "jazzy")
UBUNTU = {"noetic": "focal", "jazzy": "noble"}
APT_ROS = re.compile(r"\bros-(noetic|jazzy)-([a-z0-9][a-z0-9-]*)(=\S+)?")
LAYERS = sorted(p.parent.name for p in DOCKER.glob("*/Dockerfile")
                if p.parent.name not in DISTROS)


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


@pytest.mark.parametrize("distro", DISTROS)
def test_base_images_install_ubuntu_packages_from_a_dated_archive_snapshot(distro):
    """Every apt source of the Ubuntu archive is one dated snapshot.ubuntu.com snapshot,
    written over the base image's own sources; the build fails if a live one is left."""
    text = (DOCKER / distro / "Dockerfile").read_text()
    stamps = set(re.findall(r"https://snapshot\.ubuntu\.com/ubuntu/(\d{8}T\d{6}Z)", text))
    assert len(stamps) == 1, stamps
    stamp = stamps.pop()
    assert f"org.rsim.ubuntu_snapshot={UBUNTU[distro]}/{stamp}" in text
    readme = " ".join((SHARED.parent / "README.md").read_text().split())
    assert f"{UBUNTU[distro]} {stamp}" in readme, "simulator/README.md cites another snapshot"
    sources = "/etc/apt/sources.list" if distro == "noetic" else \
        "/etc/apt/sources.list.d/ubuntu.sources"
    assert re.search(rf"> {re.escape(sources)} ", text), f"{distro} keeps its live sources"
    assert re.search(r"! grep -rqsE '\(archive\|ports\|security\)\\\.ubuntu\\\.com' "
                     r"/etc/apt/sources\.list /etc/apt/sources\.list\.d", text), distro


@pytest.mark.parametrize("layer", LAYERS)
def test_robot_layers_install_only_from_their_base_images_snapshots(layer):
    """A robot layer adds no package source (it installs from its base image's snapshots)
    and nothing from PyPI."""
    text = (DOCKER / layer / "Dockerfile").read_text()
    assert re.search(r"^FROM \$\{BASE\}$", text, re.M), layer
    assert not re.search(r"sources\.list|add-apt-repository|\bpip3? install", text), layer


@pytest.fixture
def docker_dir(tmp_path, monkeypatch):
    """A copy of the image definitions, with Docker stubbed: every tag is missing, and the
    tags spawn.sh would build are recorded instead of built."""
    shutil.copytree(DOCKER, tmp_path / "docker")
    built = []
    monkeypatch.setattr(spawn, "DOCKER_DIR", tmp_path / "docker")
    monkeypatch.setattr(spawn, "docker",
                        lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="", stderr=""))
    monkeypatch.setattr(spawn, "_build", lambda tag, context, args: built.append(tag))
    monkeypatch.setattr(spawn, "_vendor_files", lambda spec, ctx: None)
    monkeypatch.setattr(spawn, "distro_of", lambda robot: wirecheck.interface(robot.id)
                        ["ros_distribution"])
    return tmp_path / "docker", built


@pytest.mark.parametrize("layer", LAYERS)
def test_a_changed_image_definition_is_rebuilt(docker_dir, layer):
    """The image tags are content hashes of the definitions: the same definition gives the
    same tag (no rebuild), a changed base definition gives new base and layer tags, and a
    changed layer a new layer tag on the same base, so spawn.sh never reuses a stale image."""
    root, built = docker_dir
    robot = types.SimpleNamespace(id=layer)
    distro = spawn.distro_of(robot)

    def tags():
        built.clear()
        spawn.ensure_image(robot)
        assert len(built) == 2, built
        return tuple(built)

    base, top = tags()
    assert base.startswith(f"rsim-wire/{distro}:") and top.startswith(f"rsim-wire/{layer}:")
    assert tags() == (base, top)
    dockerfile = root / distro / "Dockerfile"
    dockerfile.write_text(dockerfile.read_text().replace("snapshot.ubuntu.com/ubuntu/2",
                                                         "snapshot.ubuntu.com/ubuntu/1"))
    base2, top2 = tags()
    assert base2 != base and top2 != top
    layerfile = root / layer / "Dockerfile"
    layerfile.write_text(layerfile.read_text() + "LABEL org.rsim.test=1\n")
    base3, top3 = tags()
    assert base3 == base2 and top3 not in (top, top2)


#: The stock packages the ROS 2 wires run, by the record source that cites their release.
STOCK = {
    "rebot_b601_rs": {"robot_state_publisher": "robot_state_publisher",
                      "tf2_ros": "geometry2", "rclcpp": "rclcpp", "rclpy": "rclpy",
                      "rmw": "rmw", "launch": "launch"},
}


def test_the_vendor_workspace_is_the_pinned_revisions():
    """The reBot's layer builds Seeed's packages and the SDK the boot imports at the revisions
    the robot specification records, and never installs the real CAN driver."""
    iface = wirecheck.interface("rebot_b601_rs")
    sources = {s["id"]: s["revision"] for s in iface["sources"]}
    text = (DOCKER / "rebot_b601_rs" / "Dockerfile").read_text()
    assert sources["reBotArmController_ROS2"] in text and sources["reBotArm_control_py"] in text
    assert "third_party/reBotArm_control_py" in text
    for pkg in ("rebotarm_msgs", "rebotarmcontroller", "rebotarm_bringup"):
        assert re.search(rf"--packages-select[^\n]*\b{pkg}\b", text), pkg
    assert not re.search(r"\bpip3? install|install[^\n]*\bmotorbridge\b", text)


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
