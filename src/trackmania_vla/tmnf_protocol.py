"""TMInterface 2.x socket commands used by Proximity's TMNF environment.

The message numbers and wire layouts are adapted from
``proximity/trackmania_rl/tmi_interaction/tminterface2.py``. No game process is
started by this module.
"""

import socket
import struct
from enum import IntEnum


class MessageType(IntEnum):
    SC_RUN_STEP_SYNC = 1
    SC_CHECKPOINT_COUNT_CHANGED_SYNC = 2
    SC_LAP_COUNT_CHANGED_SYNC = 3
    SC_REQUESTED_FRAME_SYNC = 4
    SC_ON_CONNECT_SYNC = 5
    C_SET_SPEED = 6
    C_REWIND_TO_STATE = 7
    C_REWIND_TO_CURRENT_STATE = 8
    C_GET_SIMULATION_STATE = 9
    C_SET_INPUT_STATE = 10
    C_GIVE_UP = 11
    C_PREVENT_SIMULATION_FINISH = 12
    C_SHUTDOWN = 13
    C_EXECUTE_COMMAND = 14
    C_SET_TIMEOUT = 15
    C_RACE_FINISHED = 16
    C_REQUEST_FRAME = 17
    C_RESET_CAMERA = 18
    C_SET_ON_STEP_PERIOD = 19
    C_UNREQUEST_FRAME = 20
    C_TOGGLE_INTERFACE = 21
    C_IS_IN_MENUS = 22
    C_GET_INPUTS = 23


class TMInterface:
    """Connect only to an already-running local TMInterface instance."""

    def __init__(self, port: int = 8478):
        if not 1 <= port <= 65535:
            raise ValueError("invalid TMInterface port")
        self.port = port
        self.sock: socket.socket | None = None

    def register(self, timeout_s: float = 30.0) -> None:
        if self.sock is not None:
            raise RuntimeError("TMInterface is already connected")
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.settimeout(timeout_s)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            connection.connect(("127.0.0.1", self.port))
        except Exception:
            connection.close()
            raise
        self.sock = connection

    def close(self) -> None:
        connection, self.sock = self.sock, None
        if connection is not None:
            try:
                connection.sendall(struct.pack("<i", MessageType.C_SHUTDOWN))
            except OSError:
                pass
            connection.close()

    def _socket(self) -> socket.socket:
        if self.sock is None:
            raise RuntimeError("TMInterface is not connected")
        return self.sock

    def _send(self, layout: str, *values: object) -> None:
        self._socket().sendall(struct.pack("<" + layout, *values))

    def read_exact(self, length: int) -> bytes:
        if length < 0:
            raise ValueError("length must be nonnegative")
        chunks = []
        remaining = length
        while remaining:
            chunk = self._socket().recv(remaining)
            if not chunk:
                raise ConnectionError("TMInterface disconnected mid-message")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def read_int32(self) -> int:
        return struct.unpack("<i", self.read_exact(4))[0]

    def respond(self, event: MessageType) -> None:
        self._send("i", int(event))

    def set_input_state(self, left: bool, right: bool, accelerate: bool, brake: bool) -> None:
        self._send("iBBBB", MessageType.C_SET_INPUT_STATE, left, right, accelerate, brake)

    def set_speed(self, speed: float) -> None:
        self._send("if", MessageType.C_SET_SPEED, speed)

    def set_timeout(self, timeout_ms: int) -> None:
        self._send("iI", MessageType.C_SET_TIMEOUT, timeout_ms)

    def set_on_step_period(self, period_ms: int) -> None:
        self._send("ii", MessageType.C_SET_ON_STEP_PERIOD, period_ms)

    def execute_command(self, command: str) -> None:
        encoded = command.encode("utf-8")
        self._send("ii", MessageType.C_EXECUTE_COMMAND, len(encoded))
        self._socket().sendall(encoded)

    def rewind_to_current_state(self) -> None:
        self._send("i", MessageType.C_REWIND_TO_CURRENT_STATE)

    def give_up(self) -> None:
        self._send("i", MessageType.C_GIVE_UP)

    def request_frame(self, width: int, height: int) -> None:
        self._send("iii", MessageType.C_REQUEST_FRAME, width, height)

    def unrequest_frame(self) -> None:
        self._send("i", MessageType.C_UNREQUEST_FRAME)

    def get_frame_bgra(self, width: int, height: int) -> bytes:
        return self.read_exact(width * height * 4)

    def get_simulation_state_bytes(self) -> bytes:
        """Return the raw SimStateData buffer for the current simulation step."""
        self._send("i", MessageType.C_GET_SIMULATION_STATE)
        length = self.read_int32()
        return self.read_exact(length)

    def get_inputs_text(self) -> str:
        """Return TMInterface's authoritative input-event timeline."""
        self._send("i", MessageType.C_GET_INPUTS)
        length = self.read_int32()
        return self.read_exact(length).decode("utf-8")
