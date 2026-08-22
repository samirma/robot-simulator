import threading

import msgpack_numpy
import numpy as np
import websockets.sync.server as ws_server

from robot_console.arm_client import ArmClient, hold


def test_arm_client_receives_metadata_and_returns_action():
    received = []

    def handler(connection):
        connection.send(msgpack_numpy.packb({"model_name": "test-arm"}))
        connection.send(
            msgpack_numpy.packb(
                {"actions/joint_pos": {"arm": [0.1, 0.2], "gripper": [0.3]}}
            )
        )
        received.append(msgpack_numpy.unpackb(connection.recv(timeout=5)))
        connection.close()

    server = ws_server.serve(handler, "127.0.0.1", 0, compression=None, max_size=None)
    port = server.socket.getsockname()[1]
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    client = ArmClient(hold, port=port, connect_timeout=5)

    try:
        client.run()
    finally:
        server.shutdown()
        server_thread.join(timeout=5)

    assert client.metadata == {"model_name": "test-arm"}
    np.testing.assert_allclose(received[0]["arm"], [0.1, 0.2])
    np.testing.assert_allclose(received[0]["gripper"], [0.3])


def test_control_flag_sets_host_and_port(monkeypatch):
    """`--control host:port` mirrors the simulator's flag and overrides --host/--port."""
    import sys

    import robot_console.arm_client as ac

    captured = {}

    class _StubClient:
        def __init__(self, controller, host, port, connect_timeout):
            captured["host"] = host
            captured["port"] = port

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr(ac, "ArmClient", _StubClient)
    monkeypatch.setattr(sys, "argv", ["robot-console-arm", "--control", "192.168.1.7:8123"])
    ac.main()

    assert captured == {"host": "192.168.1.7", "port": 8123, "ran": True}
