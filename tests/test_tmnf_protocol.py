import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trackmania_vla.tmnf_protocol import MessageType, TMInterface


class FakeSocket:
    def __init__(self, incoming=b""):
        self.incoming = incoming
        self.sent = bytearray()
        self.closed = False

    def recv(self, length):
        piece, self.incoming = self.incoming[: min(length, 2)], self.incoming[min(length, 2) :]
        return piece

    def sendall(self, payload):
        self.sent.extend(payload)

    def close(self):
        self.closed = True


class ProtocolTests(unittest.TestCase):
    def test_wire_numbers_match_proximity(self):
        self.assertEqual(MessageType.SC_REQUESTED_FRAME_SYNC, 4)
        self.assertEqual(MessageType.C_SET_INPUT_STATE, 10)
        self.assertEqual(MessageType.C_REQUEST_FRAME, 17)

    def test_binary_keys_and_fragmented_frame(self):
        interface = TMInterface()
        fake = FakeSocket(bytes(range(8)))
        interface.sock = fake
        interface.set_input_state(True, False, True, False)
        self.assertEqual(fake.sent, struct.pack("<iBBBB", 10, 1, 0, 1, 0))
        self.assertEqual(interface.get_frame_bgra(2, 1), bytes(range(8)))
        interface.close()
        self.assertTrue(fake.closed)
        self.assertTrue(fake.sent.endswith(struct.pack("<i", MessageType.C_SHUTDOWN)))

    def test_reads_fragmented_authoritative_input_timeline(self):
        encoded = "0 press up\n50 release up".encode("utf-8")
        interface = TMInterface()
        fake = FakeSocket(struct.pack("<i", len(encoded)) + encoded)
        interface.sock = fake
        self.assertEqual(interface.get_inputs_text(), encoded.decode("utf-8"))
        self.assertEqual(fake.sent, struct.pack("<i", MessageType.C_GET_INPUTS))


if __name__ == "__main__":
    unittest.main()
