import base64
import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trackmania_vla.lzo1x import decompress

# Produced by the python-lzo C extension: lzo.compress(DATA, 1, False).
COMPRESSED = base64.b64decode(
    "AAlUcmFja01hbmlhIE5hdGlvbnMgRm9yZXZlciAgAAAA/2gAAPMAAQIDBAUGBwgJCgsMDQ4PEBESExQVFhcYGRobHB0eHyAhIiMkJSYnKCkqKywtLi8wMTIzNDU2Nzg5Ojs8PT4/QEFCQ0RFRkdISUpLTE1OT1BRUlNUVVZXWFlaW1xdXl9gYWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXp7fH1+f4CBgoOEhYaHiImKi4yNjo+QkZKTlJWWl5iZmpucnZ6foKGio6SlpqeoqaqrrK2ur7CxsrO0tba3uLm6u7y9vr/AwcLDxMXGx8jJysvMzc7P0NHS09TV1tfY2drb3N3e3+Dh4uPk5ebn6Onq6+zt7u/w8fLz9PX29/j5+vv8/f7/AAECAwQgANz9AwAgANIAAAAIQTAxLVJhY2UgcmVwbGF5IGdob3N0IGJvZHkRAAA="
)
DATA = (b"TrackMania Nations Forever " * 40) + bytes(range(256)) * 3 + b"\x00" * 500 + b"A01-Race replay ghost body"


class Lzo1xTests(unittest.TestCase):
    def test_matches_c_extension_output(self):
        output = decompress(COMPRESSED, False, len(DATA))
        self.assertEqual(output, DATA)
        self.assertEqual(
            hashlib.sha256(output).hexdigest(),
            "f118ea5969e0cad2ca2b66dff10414f5c774c1abc1ca10987d399dc9956d9528",
        )

    def test_rejects_wrong_declared_length(self):
        with self.assertRaises(ValueError):
            decompress(COMPRESSED, False, len(DATA) + 1)


if __name__ == "__main__":
    unittest.main()
