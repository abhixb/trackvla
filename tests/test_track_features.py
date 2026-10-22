import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trackmania_vla.line_follower import RacingLine
from trackmania_vla.track_features import HORIZONS, ExpertEnvelope, TrackFeatures, outcome_targets


def straight_line(length: int = 400) -> RacingLine:
    """Along +x at y=10; with y up, +z is to the car's right."""
    distance = np.arange(float(length))
    points = np.stack([distance, np.full(length, 10.0), np.zeros(length)], axis=1)
    return RacingLine(points, np.full(length, 200.0), distance)


def telemetry(t_ms: int, x: float, z: float = 0.0, y: float = 10.0, forward=(1.0, 0.0, 0.0), speed: float = 200.0):
    return {
        "race_time_ms": t_ms, "position_m": [x, y, z], "forward_world": list(forward),
        "display_speed_kmh": speed, "local_velocity_mps": [0.0, 0.0, speed / 3.6],
        "wheels_ground_contact": [True] * 4,
    }


class TrackFeatureTests(unittest.TestCase):
    def test_offset_heading_and_jump_signs(self):
        track = TrackFeatures(straight_line(), jumps_m=(100.0, 300.0))
        heading_right = (np.cos(0.1), 0.0, np.sin(0.1))
        f = track.features(telemetry(0, 50.0, z=3.0, forward=heading_right))
        self.assertAlmostEqual(f["offset_m"], 3.0, places=3)  # +z is right
        self.assertGreater(f["heading_deg"], 5.0)
        self.assertEqual((f["jump_number"], round(f["next_jump_m"])), (1, 50))
        text = track.text(telemetry(50, 200.0, z=-2.0))
        self.assertIn("2.0 m left of the experts' line", text)
        self.assertIn("100 m to jump 2", text)

    def test_envelope_recentres_the_offset(self):
        line = straight_line()
        bins = 80
        envelope = ExpertEnvelope(np.full(bins, 4.0), np.full(bins, 2.0), np.full(bins, -1.0))
        f = TrackFeatures(line, envelope).features(telemetry(0, 50.0, z=5.0))
        self.assertAlmostEqual(f["offset_m"], 1.0, places=3)  # experts run 4 m right here

    def test_crash_flag_marks_the_two_seconds_before_a_failed_lap_stalls(self):
        line = straight_line()
        # Drives 2 m per frame for 100 frames, then stays put (stuck) for 60.
        xs = [2.0 * min(i, 100) for i in range(160)]
        steps = [{"race_time_ms": 50 * i, "telemetry": telemetry(50 * i, x)} for i, x in enumerate(xs)]
        failed = outcome_targets(TrackFeatures(line), steps, finished=False)
        # Furthest point 200 m; within 20 m of it from frame 90 on.
        first = int(np.argmax(failed[:, -1] > 0))
        self.assertEqual(first, 90 - HORIZONS[-1])
        self.assertTrue((failed[first:, -1] == 1).all())
        finished = outcome_targets(TrackFeatures(line), steps, finished=True)
        self.assertEqual(finished[:, -1].sum(), 0)
        # Offsets are NaN once the horizon runs past the lap end.
        self.assertTrue(np.isnan(finished[-1, 0]) and not np.isnan(finished[0, 0]))


if __name__ == "__main__":
    unittest.main()
