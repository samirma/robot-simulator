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
