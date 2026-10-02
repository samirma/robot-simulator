"""The control-port framing and the rosbridge CBOR decoder."""

import socket
import struct
import threading

import protocol
from rosbridge_client import cbor_decode


def test_frame_round_trip_with_payload():
    a, b = socket.socketpair()
    payload = bytes(range(256)) * 400

    def writer():
        protocol.send(a, {"op": "x", "id": 3, "v": [1, 2]}, payload)
        protocol.send(a, {"event": "e"})

    threading.Thread(target=writer, daemon=True).start()
    h, p = protocol.recv(b)
    assert h["op"] == "x" and h["v"] == [1, 2] and h["nbytes"] == len(payload) and p == payload
    h, p = protocol.recv(b)
    assert h == {"event": "e"} and p is None


def test_client_matches_replies_and_events():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    events = []

    def serve():
        c, _ = srv.accept()
        while True:
            try:
                h, _ = protocol.recv(c)
            except ConnectionError:
                return
            protocol.send(c, {"event": "tick"})
            protocol.send(c, {"id": h["id"], "ok": h["op"] != "bad", "error": "no"})

    threading.Thread(target=serve, daemon=True).start()
    cl = protocol.Client("127.0.0.1", port, on_event=lambda h, p: events.append(h))
    assert cl.call("good")["ok"]
    try:
        cl.call("bad")
    except protocol.RemoteError as exc:
        assert "no" in str(exc)
    else:
        raise AssertionError
    assert len(events) >= 1
    cl.close()


def test_cbor_decoder():
    # {"op": "publish", "n": [1, -2, 3.5], "b": h'0102'}
    buf = bytes([0xA3, 0x62]) + b"op" + bytes([0x67]) + b"publish" + bytes([0x61]) + b"n" + \
        bytes([0x83, 0x01, 0x21, 0xFB]) + struct.pack(">d", 3.5) + bytes([0x61]) + b"b" + \
        bytes([0x42, 0x01, 0x02])
    v, i = cbor_decode(buf, 0)
    assert i == len(buf)
    assert v == {"op": "publish", "n": [1, -2, 3.5], "b": b"\x01\x02"}
