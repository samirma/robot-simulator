import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from fake_rosbridge import FakeRosbridge  # noqa: E402


@pytest.fixture
def fake():
    """fake(spec) -> a running FakeRosbridge, closed after the test."""
    servers = []

    def start(spec):
        s = FakeRosbridge(spec)
        servers.append(s)
        return s
    yield start
    for s in servers:
        s.close()
