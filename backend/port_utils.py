"""Dependency-free helpers used while starting the LDS web server."""
import socket


def can_bind(host: str, port: int) -> bool:
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM,
                                       flags=socket.AI_PASSIVE)
    except socket.gaierror:
        return False
    bound_any = False
    for family, socktype, proto, _, sockaddr in addresses:
        try:
            with socket.socket(family, socktype, proto) as candidate:
                if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    candidate.setsockopt(socket.SOL_SOCKET,
                                         socket.SO_EXCLUSIVEADDRUSE, 1)
                candidate.bind(sockaddr)
            bound_any = True
        except OSError:
            # A hostname such as localhost can resolve to IPv4 and IPv6. If
            # either address is occupied, Werkzeug may choose it and fail.
            return False
    return bound_any


def find_available_port(host: str, preferred_port: int, max_attempts: int = 100) -> int:
    if not 1 <= preferred_port <= 65535:
        raise ValueError(f"Invalid server port: {preferred_port}")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    stop = min(65536, preferred_port + max_attempts)
    for port in range(preferred_port, stop):
        if can_bind(host, port):
            return port
    raise OSError(f"No available TCP port found from {preferred_port} to {stop - 1}")
