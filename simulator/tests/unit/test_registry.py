"""The robot registry is robots_specs/high_level_spec.md, parsed as it is now."""

import re

import registry


def test_every_robot_section_is_a_registry_entry():
    text = registry.SPEC.read_text()
    ids_in_spec = re.findall(r"\*\*Robot id:\*\* `([^`]+)`", text)
    assert registry.ids() == ids_in_spec
    assert set(registry.ids()) == {"myagv", "so101", "ainex", "mycobot280", "myagv_mycobot280",
                                   "rosmaster_x3_plus"}


def test_kinds_files_and_dialects():
    for r in registry.robots():
        assert r.kind in registry.KINDS, r.id
        if r.composite:
            assert r.base and r.arm and not r.ros_file
            continue
        assert r.folder == f"robots_specs/{r.id}/"
        assert r.urdf and r.urdf.startswith(r.folder)
        assert r.ros_file in (f"{r.folder}ros.yml", f"{r.folder}ros2.yml")
    assert registry.get("so101").dialect == "ros2"
    assert registry.get("myagv").dialect == "ros1"
    assert registry.get("so101").mjcf == "robots_specs/so101/so101_new_calib.xml"


def test_composite_components_and_wires():
    c = registry.get("myagv_mycobot280")
    assert c.composite and c.mobile
    assert [x.id for x in registry.components(c)] == ["myagv", "mycobot280"]
    assert [(role, r.id) for role, r in registry.wires(c)] == [("base", "myagv"),
                                                              ("arm", "mycobot280")]
    assert registry.wires(registry.get("so101")) == [("main", registry.get("so101"))]


def test_mounting_transform_is_read_from_the_spec():
    link, xyz, rpy = registry.get("myagv_mycobot280").mounting_transform()
    text = registry.get("myagv_mycobot280").mounting
    assert link in text and "g_base" in text
    assert link == "base_footprint"
    assert len(xyz) == 3 and len(rpy) == 3
    assert xyz[2] > 0.0


def test_unknown_id_lists_every_accepted_id_with_its_name():
    try:
        registry.get("nope")
    except registry.RegistryError as exc:
        msg = str(exc)
    else:
        raise AssertionError("an unknown id must be refused")
    for r in registry.robots():
        assert r.id in msg and r.name in msg


def test_required_files_present_after_setup():
    for r in registry.robots():
        assert registry.missing_files(r) == [], r.id


def test_missing_files_are_reported(tmp_path, monkeypatch):
    r = registry.get("myagv")
    monkeypatch.setattr(r, "urdf", "robots_specs/myagv/does_not_exist.urdf")
    assert "robots_specs/myagv/does_not_exist.urdf" in registry.missing_files(r)
