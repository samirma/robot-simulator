"""Standalone protocol test for the simulator-hosted generic control server."""

from __future__ import annotations

import threading
import time

import msgpack_numpy
import numpy as np
import websockets.sync.client as ws_client
from control_server import ControlServer


def main() -> int:
    server = ControlServer(
        "127.0.0.1",
        0,
        {"model_name": "test-arm", "protocol": "molmospaces-control-v1"},
    )
    server.start()
    received_metadata = []
    first_publish = time.monotonic()
    assert server.publish({"qpos": {"arm": [0.1, 0.2]}}) is None
    assert time.monotonic() - first_publish < 0.1

    def client() -> None:
        with ws_client.connect(
            f"ws://127.0.0.1:{server.port}", compression=None, max_size=None
        ) as connection:
            received_metadata.append(msgpack_numpy.unpackb(connection.recv(timeout=5)))
            observation = msgpack_numpy.unpackb(connection.recv(timeout=5))
            connection.send(
                msgpack_numpy.packb(
                    {"arm": np.asarray(observation["qpos"]["arm"], dtype=np.float64)}
                )
            )

    thread = threading.Thread(target=client)
    thread.start()
    started = time.monotonic()
    action = None
    while action is None and time.monotonic() - started < 5:
        action = server.publish({"qpos": {"arm": [0.1, 0.2]}})
        time.sleep(0.01)
    server.stop()
    thread.join(timeout=5)

    assert time.monotonic() - started < 5
    assert received_metadata == [
        {"model_name": "test-arm", "protocol": "molmospaces-control-v1"}
    ]
    assert action is not None
    np.testing.assert_allclose(action["arm"], [0.1, 0.2])
    print("generic control server protocol: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
