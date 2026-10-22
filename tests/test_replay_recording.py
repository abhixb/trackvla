import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trackmania_vla.human_recording import RecordingConfig
from trackmania_vla.replay_inputs import ReplayInputs
from trackmania_vla.replay_recording import ReplayRunRecorder
from trackmania_vla.tmnf_protocol import MessageType

STEP = MessageType.SC_RUN_STEP_SYNC
FRAME = MessageType.SC_REQUESTED_FRAME_SYNC
CHECKPOINT = MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC

REPLAY = ReplayInputs(
    replay_id="42",
    player="tester",
    race_time_ms=70,
    checkpoint_times_ms=(70,),
    events=((10, "accelerate", True), (40, "left", True)),
)


class FakeInterface:
    """Scripted TMInterface: countdown, two captured frames, then the finish."""

    def __init__(self, finish_step_ms=60):
        self.values = iter(
            [
                MessageType.SC_ON_CONNECT_SYNC,
                STEP, -10,
                STEP, 0, FRAME,
                STEP, 10, STEP, 20, STEP, 30, STEP, 40,
                STEP, 50, FRAME,
                STEP, finish_step_ms,
                CHECKPOINT, 1, 1,
            ]
        )
        self.frame_times = iter([0, 50])
        self.calls = []

    def read_int32(self):
        return next(self.values)

    def get_frame_bgra(self, width, height):
        return bytes([1, 2, 3, 255] * width * height)

    def get_inputs_text(self):
        return "0 press up"

    def __getattr__(self, name):
        def record(*args):
            self.calls.append((name, *args))

        return record


def telemetry_reader(interface):
    return {"race_time_ms": next(interface.frame_times)}


class ReplayRecordingTests(unittest.TestCase):
    def record(self, interface):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        config = RecordingConfig(output_root=Path(temporary.name), width=2, height=1, min_frames=2)
        recorder = ReplayRunRecorder(
            interface,
            config,
            REPLAY,
            input_offset_ms=-10,
            game_speed=4.0,
            telemetry_reader=telemetry_reader,
        )
        return recorder.record_connected()

    def test_reproduced_lap_is_accepted_with_replay_labels(self):
        interface = FakeInterface()
        result = self.record(interface)

        self.assertTrue(result.complete, result.reason)
        self.assertEqual(result.frame_count, 2)
        self.assertEqual(result.checkpoint_errors_ms, (-10,))
        steps = [json.loads(line) for line in (result.episode_dir / "steps.jsonl").read_text().splitlines()]
        # Offset -10: accelerate (replay 10 ms) from race 0, left (40 ms) from 30.
        self.assertEqual(steps[0]["controls"], {"left": False, "right": False, "accelerate": True, "brake": False})
        self.assertEqual(steps[1]["controls"], {"left": True, "right": False, "accelerate": True, "brake": False})
        self.assertEqual(steps[1]["race_time_ms"], 50)
        manifest = json.loads((result.episode_dir / "manifest.json").read_text())
        self.assertEqual(manifest["source"]["replay_id"], "42")
        self.assertEqual(manifest["behavior_mode"], "fast")

        calls = interface.calls
        self.assertIn(("set_on_step_period", 10), calls)
        self.assertIn(("set_input_state", True, False, True, False), calls)
        # Each capture pauses the game, and the frame resumes it.
        request = calls.index(("request_frame", 2, 1))
        self.assertEqual(calls[request - 1], ("set_speed", 0.0))
        self.assertIn(("set_speed", 4.0), calls[request:])

    def test_diverging_lap_is_rejected(self):
        result = self.record(FakeInterface(finish_step_ms=90))
        self.assertFalse(result.complete)
        self.assertEqual(result.reason, "replay_not_reproduced")
        self.assertEqual(result.episode_dir.parent.name, "failed")


if __name__ == "__main__":
    unittest.main()
