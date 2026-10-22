import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from tminterface.structs import SimStateData
except ImportError:  # optional game dependency
    SimStateData = None

from trackmania_vla.telemetry import TELEMETRY_FIELDS, parse_simulation_state


def synthetic_state(race_time_ms: int = 50, *, flags: int = 1 | 2 | 128) -> bytes:
    state = SimStateData()
    state.flags = flags
    # Car forward (+z in car frame) points along world +x.
    state.dyna.current_state.rotation = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    state.dyna.current_state.linear_speed = [10.0, 0.0, 0.0]
    state.dyna.current_state.angular_speed = [0.0, 0.5, 0.0]
    state.player_info.display_speed = 36
    state.player_info.race_time = race_time_ms
    state.scene_mobil.engine.actual_rpm = 5000.0
    state.scene_mobil.engine.gear = 2
    return bytes(state.data)


@unittest.skipIf(SimStateData is None, "tminterface is not installed")
class TelemetryTests(unittest.TestCase):
    def test_parses_speed_engine_and_car_frame_velocity(self):
        telemetry = parse_simulation_state(synthetic_state())
        self.assertEqual(tuple(telemetry), TELEMETRY_FIELDS)
        self.assertEqual(telemetry["race_time_ms"], 50)
        self.assertEqual(telemetry["display_speed_kmh"], 36)
        self.assertAlmostEqual(telemetry["speed_mps"], 10.0)
        self.assertEqual(telemetry["rpm"], 5000.0)
        self.assertEqual(telemetry["gear"], 2)
        self.assertEqual(telemetry["local_velocity_mps"], [0.0, 0.0, 10.0])
        self.assertEqual(telemetry["local_angular_velocity_rps"], [0.0, 0.5, 0.0])
        self.assertEqual(len(telemetry["wheels_ground_contact"]), 4)

    def test_rejects_state_without_dyna_or_player_info(self):
        with self.assertRaises(ValueError):
            parse_simulation_state(synthetic_state(flags=1))


if __name__ == "__main__":
    unittest.main()
