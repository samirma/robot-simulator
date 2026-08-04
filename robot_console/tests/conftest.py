import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fake_bridge import FakeBridge  # noqa: E402


@pytest.fixture
def bridge():
    server = FakeBridge()
    server.start()
    try:
        yield server
    finally:
        server.stop()
