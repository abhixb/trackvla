import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trackmania_vla.start_frame_check import HUD_ROW2, REFERENCE, opponent_ghost_at_start


class StartFrameCheckTests(unittest.TestCase):
    def episode(self, hud_row2):
        frame = np.full((120, 160, 3), 90, np.uint8)
        frame[HUD_ROW2] = hud_row2
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        (Path(temporary.name) / "frames.rgb").write_bytes(frame.tobytes() * 2)
        return Path(temporary.name)

    def test_clean_start_passes(self):
        reference = np.asarray(Image.open(REFERENCE).convert("RGB"), np.uint8)
        self.assertFalse(opponent_ghost_at_start(self.episode(reference)))

    def test_extra_hud_entry_is_detected(self):
        reference = np.asarray(Image.open(REFERENCE).convert("RGB"), np.int16)
        ghost_row = np.clip(reference - 60, 0, 255).astype(np.uint8)  # dark opponent panel
        self.assertTrue(opponent_ghost_at_start(self.episode(ghost_row)))


if __name__ == "__main__":
    unittest.main()
