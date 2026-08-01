import socket
import pytest
from port_utils import find_available_port


def test_uses_available_preferred_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    assert find_available_port("127.0.0.1", port) == port


def test_advances_past_occupied_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        occupied = listener.getsockname()[1]
        listener.listen()
        assert find_available_port("127.0.0.1", occupied) > occupied


def test_localhost_does_not_ignore_an_occupied_ipv4_address():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        occupied = listener.getsockname()[1]
        listener.listen()
        assert find_available_port("localhost", occupied) > occupied


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_rejects_invalid_ports(port):
    with pytest.raises(ValueError, match="Invalid server port"):
        find_available_port("127.0.0.1", port)


def test_rejects_empty_search_range():
    with pytest.raises(ValueError, match="max_attempts"):
        find_available_port("127.0.0.1", 5050, max_attempts=0)
