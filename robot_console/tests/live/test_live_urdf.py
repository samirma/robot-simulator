"""The packaged vendor excerpts (tests/urdf/<id>.xml) against the pinned files themselves (opt in:
``pytest -m live``; downloads from GitHub). Each file must have its recorded digest, each excerpt
must be the cited lines of it (indentation aside), and the excerpts must cover every joint of the
file (every static transform of a launch file) and be the inertial origin of the link they name.
The ROSMASTER file is in a Google Drive archive, not on GitHub: set YAHBOOMCAR_WS_ZIP to a local
yahboomcar_ws.zip (checked against the profile's digest) to include it."""

import hashlib
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
import xml.parsers.expat
import zipfile

import pytest

from robot_console.profiles import SUPPORTED_IDS, load

pytestmark = pytest.mark.live
TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE = re.compile(r'<file source="([^"]+)" path="([^"]+)".*?>(.*?)</file>', re.S)
EXCERPT = re.compile(r'<excerpt lines="(\d+)(?:-(\d+))?"([^>]*)>\n(.*?)\n\s*</excerpt>', re.S)


def pinned(pid, key, path):
    src = load(pid).sources[key]
    if "revision" in src:
        owner_repo = src["url"].removeprefix("https://github.com/")
        url = f"https://raw.githubusercontent.com/{owner_repo}/{src['revision']}/{path}"
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read()
    z = os.environ.get("YAHBOOMCAR_WS_ZIP")
    if not z:
        pytest.skip(f"{key} is a Google Drive archive: set YAHBOOMCAR_WS_ZIP to a local copy")
    with open(z, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == src["sha256"], z
    with zipfile.ZipFile(z) as zf:
        return zf.read(f"yahboomcar_ws/{path}")


def structure(text):
    """Top-level joints (start, end) and each link's inertial origin start line, by expat."""
    p = xml.parsers.expat.ParserCreate()
    stack, joints, inertial, open_ = [], set(), {}, {}

    def start(tag, a):
        parent = stack[-1] if stack else None
        stack.append(tag)
        if tag == "joint" and parent in ("robot", "xacro:macro"):
            open_["joint"] = p.CurrentLineNumber
        elif tag == "link" and parent in ("robot", "xacro:macro"):
            open_["link"] = a["name"].replace("${prefix}", "")
        elif tag == "origin" and parent == "inertial" and stack[-3:-2] == ["link"]:
            inertial[open_["link"]] = p.CurrentLineNumber

    def end(tag):
        stack.pop()
        if tag == "joint" and (stack[-1] if stack else None) in ("robot", "xacro:macro"):
            joints.add((open_.pop("joint"), p.CurrentLineNumber))
    p.StartElementHandler, p.EndElementHandler = start, end
    p.Parse(text.encode(), True)
    return joints, inertial


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_excerpts_are_the_pinned_files(pid):
    fixture = open(os.path.join(TESTS, "urdf", f"{pid}.xml"), encoding="utf-8").read()
    files = ET.fromstring(fixture).findall("file")
    blocks = FILE.findall(fixture)
    assert [(f.get("source"), f.get("path")) for f in files] == [b[:2] for b in blocks]
    for f, (key, path, body) in zip(files, blocks):
        data = pinned(pid, key, path)
        assert hashlib.sha256(data).hexdigest() == f.get("sha256"), (pid, path)
        text = data.decode("utf-8")
        lines = text.split("\n")
        covered, origins = set(), {}
        for a, b, attrs, excerpt in EXCERPT.findall(body):
            a, b = int(a), int(b or a)
            want = [x.strip() for x in lines[a - 1:b]]
            assert [x.strip() for x in excerpt.split("\n")] == want, (pid, path, a, b)
            m = re.search(r'inertial="([^"]+)"', attrs)
            if m:
                origins[m.group(1)] = a
            else:
                covered.add((a, b))
        if path.endswith(".launch"):
            assert covered == {(i, i) for i, x in enumerate(lines, 1) if "static_transform_publisher" in x}
        else:
            joints, inertial = structure(text)
            assert covered == joints, (pid, path, sorted(joints - covered), sorted(covered - joints))
            for link, a in origins.items():
                assert inertial[link] == a, (pid, path, link)
