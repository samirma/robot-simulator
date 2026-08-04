import io
import socket

from robot_console.preflight import preflight, probe_tcp, startup_instructions


def test_probe_succeeds_against_a_listener():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        result = probe_tcp("127.0.0.1", port, timeout=2.0)
    finally:
        listener.close()
    assert result.ok
    assert result.detail == "connected"
    assert result.url == f"ws://127.0.0.1:{port}"


def test_probe_reports_refused():
    # Bind and immediately close, so the port is almost certainly free.
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()

    result = probe_tcp("127.0.0.1", port, timeout=1.0)
    assert not result.ok
    assert "refused" in result.detail


def test_probe_reports_a_bad_hostname():
    result = probe_tcp("nonexistent-host.invalid", 9090, timeout=1.0)
    assert not result.ok
    # Three failure modes point at three different fixes, so they are distinguished.
    assert "resolve" in result.detail


def test_instructions_cover_every_way_to_start_a_robot():
    text = startup_instructions("192.168.1.42", 9091)
    assert "9091" in text
    assert "run.sh view --robot myagv" in text
    assert "rosbridge_server.py" in text  # standalone, no MuJoCo
    assert "rosbridge_websocket.launch" in text  # real hardware
    assert "myagv_active.launch" in text
    assert "--no-preflight" in text


def test_preflight_prints_instructions_on_failure():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()

    stream = io.StringIO()
    assert preflight("127.0.0.1", port, timeout=1.0, stream=stream) is False
    output = stream.getvalue()
    assert "no rosbridge" in output
    assert "run.sh view --robot myagv" in output


def test_preflight_is_quiet_on_success():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        stream = io.StringIO()
        assert preflight("127.0.0.1", port, timeout=2.0, stream=stream) is True
        assert stream.getvalue() == ""
    finally:
        listener.close()
