import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dagger import SectionTakeover  # noqa: E402
from trackmania_vla.line_follower import RacingLine  # noqa: E402


def telemetry(t, x, z, y=0.0, sliding=False):
    return {"race_time_ms": t, "position_m": [x, y, z], "car_sliding": sliding}


class SectionTakeoverTests(unittest.TestCase):
    def setUp(self):
        n = 300
        points = np.stack([np.zeros(n), np.zeros(n), np.arange(n, dtype=float)], axis=1)
        self.line = RacingLine(points, np.full(n, 50.0), np.arange(n, dtype=float))

    def test_only_inside_sections(self):
        takeover = SectionTakeover(self.line, [(100, 200)], drift_m=3, release_m=1.5, hold_ms=500)
        self.assertFalse(takeover(telemetry(0, 5.0, 50)))      # off-line but outside the section
        self.assertTrue(takeover(telemetry(50, 5.0, 120)))     # off-line inside it
        self.assertFalse(takeover(telemetry(100, 5.0, 210)))   # section over: hand back

    def test_sliding_triggers_and_release_needs_hold(self):
        takeover = SectionTakeover(self.line, [(100, 200)], drift_m=3, release_m=1.5, hold_ms=500)
        self.assertTrue(takeover(telemetry(0, 0.0, 110, sliding=True)))
        self.assertTrue(takeover(telemetry(100, 0.5, 120)))    # back on line, hold not over
        self.assertFalse(takeover(telemetry(600, 0.5, 140)))   # back for 500 ms
        self.assertEqual(takeover.triggers, 1)


if __name__ == "__main__":
    unittest.main()
