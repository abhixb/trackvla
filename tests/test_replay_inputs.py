import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trackmania_vla.lzo1x import install as install_lzo

install_lzo()
try:
    import pygbx  # noqa: F401
except ImportError:  # optional replay dependency
    pygbx = None

from trackmania_vla.human_recording import ControlState
from trackmania_vla.replay_inputs import InputTimeline, ReplayRejected, load_keyboard_replay


def entry(time, name, enabled=1):
    return SimpleNamespace(time=time, event_name=name, enabled=enabled, flags=0)


def fake_gbx(entries, race_time=24_040, cp_times=(15_650, 19_630, 24_040)):
    ghost = SimpleNamespace(
        control_entries=entries, race_time=race_time, cp_times=list(cp_times), login="tester"
    )
    return SimpleNamespace(get_classes_by_ids=lambda ids: [ghost])


KEYBOARD_LAP = [
    entry(-2_500, "Respawn", 0),
    entry(0, "_FakeIsRaceRunning"),
    entry(10, "Accelerate"),
    entry(500, "SteerRight"),
    entry(610, "SteerRight", 0),
    entry(700, "SteerLeft"),
    entry(750, "Brake"),
    entry(800, "SteerLeft", 0),
    entry(800, "Brake", 0),
    entry(24_040, "_FakeFinishLine"),
]


class InputTimelineTests(unittest.TestCase):
    def test_offset_shifts_replay_time_to_race_time(self):
        timeline = InputTimeline(((10, "accelerate", True), (500, "right", True)), offset_ms=-10)
        self.assertEqual(timeline.state_at(-20), ControlState(False, False, False, False))
        self.assertEqual(timeline.state_at(0), ControlState(False, False, True, False))
        self.assertEqual(timeline.state_at(480), ControlState(False, False, True, False))
        self.assertEqual(timeline.state_at(490), ControlState(False, True, True, False))

    def test_simultaneous_events_resolve_to_final_state(self):
        timeline = InputTimeline(
            ((700, "left", True), (750, "brake", True), (800, "left", False), (800, "brake", False)),
            offset_ms=0,
        )
        self.assertEqual(timeline.state_at(760), ControlState(True, False, False, True))
        self.assertEqual(timeline.state_at(800), ControlState(False, False, False, False))


@unittest.skipIf(pygbx is None, "pygbx is not installed")
class LoadKeyboardReplayTests(unittest.TestCase):
    def load(self, entries, **kwargs):
        with patch("pygbx.Gbx", return_value=fake_gbx(entries, **kwargs)):
            return load_keyboard_replay(Path("123.Replay.Gbx"))

    def test_loads_keyboard_lap(self):
        replay = self.load(KEYBOARD_LAP)
        self.assertEqual(replay.replay_id, "123")
        self.assertEqual(replay.race_time_ms, 24_040)
        self.assertEqual(replay.checkpoint_times_ms, (15_650, 19_630, 24_040))
        self.assertEqual(replay.events[0], (10, "accelerate", True))
        self.assertIn((610, "right", False), replay.events)

    def test_rejects_analog_steering(self):
        with self.assertRaisesRegex(ReplayRejected, "analog"):
            self.load([entry(10, "Accelerate"), entry(60, "Steer", 63861), entry(24_040, "_FakeFinishLine")])

    def test_rejects_mid_race_respawn(self):
        with self.assertRaisesRegex(ReplayRejected, "respawn"):
            self.load(KEYBOARD_LAP[:4] + [entry(5_000, "Respawn")] + KEYBOARD_LAP[4:])

    def test_rejects_unfinished_replay(self):
        with self.assertRaisesRegex(ReplayRejected, "finish"):
            self.load(KEYBOARD_LAP[:-1])


if __name__ == "__main__":
    unittest.main()
