"""`robots_specs/robots.yml` names files that exist, and every mesh is in the manifest.

Workspace spec §2: each entry carries the paths of its URDF, MJCF and ROS interface
files, and `meshes` names the folder(s) holding the official meshes they reference. The
meshes themselves are not tracked (`fetch_robot_assets.sh` restores them and verifies them
against `robots_specs/meshes.sha256`), so what is checked here is the manifest: every mesh
a URDF or MJCF references is listed in it, every listed mesh lies in its robot's `meshes`
folder(s), and a mesh that *is* on disk is one the manifest lists.
"""

from __future__ import annotations

import re
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

from _workspace import SPECS, robots_yml, ros_file

MANIFEST = SPECS / "meshes.sha256"
MESH_SUFFIXES = {".stl", ".dae", ".obj"}


def _manifest() -> dict[str, str]:
    out = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, path = line.split(maxsplit=1)
            out[path.strip()] = digest
    return out


def _as_list(value) -> list[str]:
    return list(value) if isinstance(value, list) else [value]


def _norm(path: PurePosixPath) -> str:
    parts: list[str] = []
    for part in path.parts:
        if part == "..":
            parts.pop()
        elif part != ".":
            parts.append(part)
    return "/".join(parts)


def mesh_references(robot: str, model: str) -> set[str]:
    """Every mesh a URDF or MJCF references, as a path under `robots_specs/`.

    `package://<pkg>/<rest>` resolves to `<robot>/<rest>` (robots.yml's rule); a relative
    URDF path to the file's folder; an MJCF `file` to its compiler's `meshdir`.
    """
    path = PurePosixPath(model)
    text = (SPECS / model).read_text(encoding="utf-8")
    refs: set[str] = set()
    if model.endswith(".urdf"):
        for ref in re.findall(r'filename="([^"]+)"', text):
            package = re.match(r"package://[^/]+/(.+)", ref)
            refs.add(_norm(PurePosixPath(robot) / package.group(1)) if package
                     else _norm(path.parent / ref))
    else:
        root = ET.fromstring(text)
        compiler = root.find("compiler")
        meshdir = compiler.get("meshdir", "") if compiler is not None else ""
        for mesh in root.iter("mesh"):
            if mesh.get("file"):
                refs.add(_norm(path.parent / meshdir / mesh.get("file")))
    return {r for r in refs if PurePosixPath(r).suffix.lower() in MESH_SUFFIXES}


class RobotsYml(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.entries = robots_yml()
        cls.manifest = _manifest()

    def test_there_are_robots_and_the_three_simulated_ones(self) -> None:
        simulated = {rid for rid, e in self.entries.items() if e.get("simulated") is True}
        self.assertEqual(simulated, {"myagv", "so101", "ainex"})
        for rid, entry in self.entries.items():
            self.assertIn(entry.get("simulated"), (True, False), rid)
            self.assertIn(entry.get("placement"), ("floor", "worktop"), rid)

    def test_every_urdf_mjcf_and_ros_path_exists_in_the_robots_folder(self) -> None:
        for rid, entry in self.entries.items():
            for key in ("urdf", "mjcf", "ros"):
                if key not in entry:
                    self.assertNotEqual(key, "urdf", f"{rid} has no urdf")
                    self.assertNotEqual(key, "ros", f"{rid} has no ros file")
                    continue
                path = SPECS / entry[key]
                self.assertTrue(path.is_file(), f"{rid}.{key}: {path}")
                self.assertEqual(Path(entry[key]).parts[0], rid, f"{rid}.{key}")

    def test_each_ros_file_is_named_for_its_dialect_and_has_a_stop_command(self) -> None:
        for rid, entry in self.entries.items():
            ros = ros_file(SPECS / entry["ros"])
            expected = {"ros.yml": "ros1", "ros2.yml": "ros2"}[Path(entry["ros"]).name]
            if "extends" not in ros:
                self.assertEqual(ros.get("dialect"), expected, rid)
            self.assertTrue(ros.get("stop_command"), f"{rid} has no stop_command")

    def test_a_composite_names_its_base_and_extends_its_ros_file(self) -> None:
        for rid, entry in self.entries.items():
            if "base" not in entry:
                continue
            self.assertIn(entry["base"], self.entries, rid)
            ros = ros_file(SPECS / entry["ros"])
            base_ros = self.entries[entry["base"]]["ros"]
            self.assertIn(Path(base_ros).parent.name, str(ros.get("extends")), rid)

    def test_every_mesh_folder_is_in_the_robots_folder(self) -> None:
        for rid, entry in self.entries.items():
            for folder in _as_list(entry["meshes"]):
                self.assertEqual(Path(folder).parts[0], rid, f"{rid}.meshes: {folder}")

    def test_every_referenced_mesh_is_in_the_manifest(self) -> None:
        for rid, entry in self.entries.items():
            for key in ("urdf", "mjcf"):
                if key in entry:
                    missing = mesh_references(rid, entry[key]) - set(self.manifest)
                    self.assertEqual(missing, set(), f"{rid}.{key}")

    def test_every_manifest_entry_is_in_a_declared_mesh_folder(self) -> None:
        folders = [f.rstrip("/") + "/" for e in self.entries.values()
                   for f in _as_list(e["meshes"])]
        for path, digest in self.manifest.items():
            self.assertRegex(digest, r"^[0-9a-f]{64}$", path)
            self.assertTrue(any(path.startswith(f) for f in folders), path)

    def test_every_mesh_on_disk_is_in_the_manifest(self) -> None:
        for entry in self.entries.values():
            for folder in _as_list(entry["meshes"]):
                base = SPECS / folder
                if not base.is_dir():
                    continue  # not fetched here; fetch_robot_assets.sh restores it
                for path in base.rglob("*"):
                    if path.suffix.lower() in MESH_SUFFIXES:
                        rel = path.relative_to(SPECS).as_posix()
                        self.assertIn(rel, self.manifest, rel)


if __name__ == "__main__":
    unittest.main()
