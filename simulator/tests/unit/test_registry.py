"""The robot registry is the robot specification's robot files, robots_specs/<id>.md, parsed
as they are now (robot specification §2, amended 2026-10-02)."""

import re

import registry


def test_every_robot_file_is_a_registry_entry():
    table = re.findall(r"^\| `([^`]+)` \|", registry.SPEC.read_text(), re.M)
    files = {p.stem for p in registry.SPECS.glob("*.md")} - {"high_level_spec", "SCHEMA"}
    assert set(registry.ids()) == set(table) == files
    assert set(registry.ids()) == {"myagv", "so101", "ainex", "mycobot280", "rosmaster_x3_plus"}
    for r in registry.robots():
        assert r.name in registry.SPEC.read_text(), r.id


def test_kinds_files_and_dialects():
    for r in registry.robots():
        assert r.kind in ("mobile_base", "arm", "humanoid", "mobile_manipulator"), r.id
        assert r.folder == f"robots_specs/{r.id}/"
        assert r.urdf and r.urdf.startswith(r.folder)
        assert r.ros_file in (f"{r.folder}ros.yml", f"{r.folder}ros2.yml")
    assert registry.get("so101").dialect == "ros2"
    assert registry.get("myagv").dialect == "ros1"
    assert registry.get("so101").mjcf == "robots_specs/so101/so101_new_calib.xml"


def test_robots_are_single_bodies():
    """No assembly of robots (simulator spec §2.3, amended 2026-10-02)."""
    for name in ("components", "wires", "mounting_transform"):
        assert not hasattr(registry, name), name
    for r in registry.robots():
        for name in ("composite", "base", "arm", "mounting", "mounting_transform"):
            assert not hasattr(r, name), (r.id, name)


def test_unknown_id_lists_every_accepted_id_with_its_name():
    try:
        registry.get("nope")
    except registry.RegistryError as exc:
        msg = str(exc)
    else:
        raise AssertionError("an unknown id must be refused")
    for r in registry.robots():
        assert r.id in msg and r.name in msg
    assert "myagv_mycobot280" not in msg


def test_required_files_present_after_setup():
    for r in registry.robots():
        assert registry.missing_files(r) == [], r.id


def test_missing_files_are_reported(tmp_path, monkeypatch):
    r = registry.get("myagv")
    monkeypatch.setattr(r, "urdf", "robots_specs/myagv/does_not_exist.urdf")
    assert "robots_specs/myagv/does_not_exist.urdf" in registry.missing_files(r)


def test_a_reference_in_an_xml_comment_is_not_required(tmp_path):
    """The myAGV URDF's commented-out base_footprint names urdf/myagv_all.dae: a comment is
    no reference, so that mesh is not a required file."""
    urdf = tmp_path / "r.urdf"
    urdf.write_text('<robot><!-- <mesh filename="gone.dae"/> -->'
                    '<link><visual><geometry><mesh filename="here.stl"/></geometry></visual>'
                    '</link></robot>')
    assert registry._urdf_files(urdf) == [tmp_path / "here.stl"]
    mjcf = tmp_path / "m.xml"
    mjcf.write_text('<mujoco><asset><!-- <mesh file="gone.stl"/> --><mesh file="a.stl"/>'
                    '</asset></mujoco>')
    assert registry._model_files(mjcf) == [tmp_path / "a.stl"]
    myagv = registry.get("myagv")
    assert not any("myagv_all" in str(p) for p in registry._urdf_files(myagv.path(myagv.urdf)))
