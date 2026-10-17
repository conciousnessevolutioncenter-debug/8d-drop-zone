import socket
import pytest
from studio.launch import reserve_port, wait_ready

def test_port_conflict():
    with socket.socket() as occupied:
        occupied.bind(('127.0.0.1',0));occupied.listen()
        port=occupied.getsockname()[1]
        with pytest.raises(OSError): reserve_port(port)

def test_failed_service_never_ready():
    assert wait_ready('http://127.0.0.1:1/api/health', timeout=.05) is False
