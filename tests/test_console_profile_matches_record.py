"""The console profiles agree with the recorded interfaces on the cameras and sensors the view page shows.

The oracles are the robot records: robots_specs/rebot_b601_rs/ros2.yml, robots_specs/ainex/ros.yml and
robots_specs/myagv/ros.yml. The profiles (robot_console/src/robot_console/profiles/<id>.yaml) are read as data and
never imported: the console is independent of the workspace records, while this workspace check may read both trees.

    uv run --no-project --with pytest --with pyyaml pytest tests/test_console_profile_matches_record.py
"""

import pathlib

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROFILES = ROOT / "robot_console" / "src" / "robot_console" / "profiles"
ROBOTS = {
    "rebot_b601_rs": ROOT / "robots_specs" / "rebot_b601_rs" / "ros2.yml",
    "ainex": ROOT / "robots_specs" / "ainex" / "ros.yml",
    "myagv": ROOT / "robots_specs" / "myagv" / "ros.yml",
}

COLOR = "/camera/camera/color/image_raw"
ALIGNED = "/camera/camera/aligned_depth_to_color/image_raw"
RECTIFIED = "/camera/camera/depth/image_rect_raw"
#: The camera streams each profile shows, in the profile's order (the record's served rows plus any the boot publishes).
EXPECTED_CAMERAS = {
    "rebot_b601_rs": [COLOR, ALIGNED],
    "ainex": ["/camera/image_raw", "/camera/image_rect_color"],
    "myagv": ["/usb_cam/image_raw", "/usb_cam/image_raw/compressed"],
}
#: The read-only sensor topics each profile shows (the ones the page subscribes to).
EXPECTED_SENSORS = {
    "ainex": ["/ros_robot_controller/imu_raw", "/imu_corrected", "/imu", "/ros_robot_controller/mag"],
    "myagv": ["/imu", "/scan", "/point_cloud"],
}
def load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def profile(robot_id):
    return load(PROFILES / f"{robot_id}.yaml")


def record(robot_id):
    return load(ROBOTS[robot_id])


def topics_by_name(rec):
    return {t["name"]: t for t in rec["topics"]}


def wire(robot_id, name):
    """The fully qualified name the profile's row names on the wire: a profile with a namespace composes its
    arm-namespace rows as the profile's `compose` says, and its global names stay as they are."""
    ns = profile(robot_id).get("namespace") or {}
    if not ns.get("default") or name in ns.get("global_names", []):
        return name
    return ns["compose"].format(ns=ns["default"], name=name)


def served_record_cameras(rec):
    """The served camera streams of the record: sensors.cameras rows that are not optional, keyed by image topic."""
    return {c["image_topic"]: c for c in rec["sensors"]["cameras"] if not c.get("optional")}


def test_the_record_serves_the_colour_and_aligned_depth_streams_only():
    rec = record("rebot_b601_rs")
    raw = {t["name"] for t in rec["topics"] if t.get("camera") and not t.get("optional")}
    assert set(served_record_cameras(rec)) == raw == {COLOR, ALIGNED}
    rect = next(t for t in rec["topics"] if t["name"] == RECTIFIED)
    assert rect.get("optional") is True and "not served" in rect["notes"]           # the guide uses the aligned depth


def test_profile_cameras_are_the_served_recorded_cameras():
    served = served_record_cameras(record("rebot_b601_rs"))
    cams = profile("rebot_b601_rs")["cameras"]
    assert sorted(c["topic"] for c in cams) == sorted(served) == sorted([COLOR, ALIGNED])
    assert not any(c.get("optional") for c in cams)                                 # the record serves them: required
    assert RECTIFIED not in {c["topic"] for c in cams}
    for c in cams:
        r = served[c["topic"]]
        assert c["camera_info"] == r["info_topic"], c["topic"]
        assert c["rate_hz"] == r["rate"], c["topic"]
        assert c["encodings"] == [r["encoding"]], c["topic"]
        assert (c.get("kind", "color") == "depth") == (r["encoding"] == "16UC1"), c["topic"]
        assert c["type"] == "sensor_msgs/msg/Image", c["topic"]
    depth = next(c for c in cams if c["topic"] == ALIGNED)
    assert depth["unit_m"] == 0.001                                                 # 16UC1 in millimetres


@pytest.mark.parametrize("robot_id", sorted(ROBOTS))
def test_profile_cameras_are_the_expected_streams(robot_id):
    assert [c["topic"] for c in profile(robot_id)["cameras"]] == EXPECTED_CAMERAS[robot_id]


@pytest.mark.parametrize("robot_id", sorted(ROBOTS))
def test_every_served_record_camera_is_a_profile_camera_with_its_recorded_figures(robot_id):
    cams = {c["topic"]: c for c in profile(robot_id)["cameras"]}
    for row in record(robot_id)["sensors"]["cameras"]:
        if row.get("optional"):
            continue
        c = cams[row["image_topic"]]                       # a KeyError names a served stream the profile lacks
        assert c["camera_info"] == row["info_topic"], (robot_id, row["image_topic"])
        assert c["rate_hz"] == row["rate"], (robot_id, row["image_topic"])
        assert c["encodings"] == [row["encoding"]], (robot_id, row["image_topic"])


@pytest.mark.parametrize("robot_id", sorted(ROBOTS))
def test_every_profile_camera_is_a_recorded_out_topic_with_its_type_rate_and_optionality(robot_id):
    rec = topics_by_name(record(robot_id))
    for c in profile(robot_id)["cameras"]:
        t = rec[wire(robot_id, c["topic"])]
        assert t["direction"] == "out" and t["type"] == c["type"], c["topic"]
        assert bool(t.get("optional")) == bool(c.get("optional")), c["topic"]
        assert t["rate"] == c["rate_hz"], c["topic"]
        if c.get("camera_info"):
            info = rec[wire(robot_id, c["camera_info"])]
            assert info["direction"] == "out" and info["type"].endswith("CameraInfo"), c["camera_info"]
            assert bool(info.get("optional")) <= bool(c.get("optional")), c["camera_info"]   # a served camera's info is served


@pytest.mark.parametrize("robot_id", sorted(ROBOTS))
def test_profile_sensors_are_recorded_served_out_topics_of_the_same_type(robot_id):
    rec = topics_by_name(record(robot_id))
    for s in profile(robot_id).get("sensors", []):
        t = rec[wire(robot_id, s["topic"])]
        assert t["direction"] == "out" and t["type"] == s["type"], s["topic"]
        assert not t.get("optional"), s["topic"]


@pytest.mark.parametrize("robot_id", sorted(EXPECTED_SENSORS))
def test_profile_sensor_topics_are_the_expected_ones(robot_id):
    assert sorted(wire(robot_id, s["topic"]) for s in profile(robot_id)["sensors"]) == sorted(EXPECTED_SENSORS[robot_id])
