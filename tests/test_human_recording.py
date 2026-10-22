import json
import struct
import sys
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trackmania_vla.human_recording import (
    ControlState,
    HumanRunRecorder,
    RecordingConfig,
    bgra_to_rgb,
    launch_and_connect,
)
from trackmania_vla.tmnf_protocol import MessageType
from test_telemetry import SimStateData, synthetic_state


class FakeKeyboard:
    def __init__(self):
        self.states = iter(
            [
                ControlState(False, False, True, False),
                ControlState(True, False, True, False),
            ]
        )

    def sample(self):
        return next(self.states)


class FakeInterface:
    def __init__(self):
        self.values = iter(
            [
                MessageType.SC_ON_CONNECT_SYNC,
                MessageType.SC_RUN_STEP_SYNC,
                -3000,
                MessageType.SC_RUN_STEP_SYNC,
                0,
                MessageType.SC_REQUESTED_FRAME_SYNC,
                MessageType.SC_RUN_STEP_SYNC,
                50,
                MessageType.SC_REQUESTED_FRAME_SYNC,
                MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC,
                1,
                1,
            ]
        )
        self.calls = []
        self.frame = bytes([1, 2, 3, 255, 10, 20, 30, 255])
        self.step_times = [0, 50]

    def read_int32(self):
        return next(self.values)

    def respond(self, event):
        self.calls.append(("respond", event))

    def request_frame(self, width, height):
        self.calls.append(("request_frame", width, height))

    def get_frame_bgra(self, width, height):
        return self.frame

    def get_simulation_state_bytes(self):
        self.calls.append(("get_simulation_state_bytes",))
        return synthetic_state(self.step_times.pop(0))

    def get_inputs_text(self):
        self.calls.append(("get_inputs_text",))
        return "0 press up\n50 press left"

    def close(self):
        self.calls.append(("close",))

    def __getattr__(self, name):
        def record(*args):
            self.calls.append((name, *args))

        return record


@unittest.skipIf(SimStateData is None, "tminterface is not installed")
class HumanRecordingTests(unittest.TestCase):
    def test_bgra_to_rgb(self):
        self.assertEqual(
            bgra_to_rgb(bytes([1, 2, 3, 255, 10, 20, 30, 128]), 2, 1),
            bytes([3, 2, 1, 30, 20, 10]),
        )
        with self.assertRaises(ValueError):
            bgra_to_rgb(b"short", 2, 1)

    def test_finished_run_is_atomically_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = RecordingConfig(
                output_root=Path(temporary) / "dataset",
                width=2,
                height=1,
                period_ms=50,
                min_frames=2,
            )
            interface = FakeInterface()
            completed: Future[bool] = Future()
            completed.set_result(True)
            schedule_no_ghost_setup = Mock(return_value=completed)
            result = HumanRunRecorder(
                interface,
                config,
                FakeKeyboard(),
                schedule_no_ghost_setup=schedule_no_ghost_setup,
            ).record_connected(game_pid=123)

            self.assertTrue(result.complete)
            self.assertEqual(result.reason, "finished")
            self.assertEqual(result.frame_count, 2)
            self.assertEqual(result.episode_dir.parent.name, "episodes")
            manifest = json.loads((result.episode_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["finish_detection"]["checkpoint_current"], 1)
            self.assertEqual(manifest["finish_detection"]["checkpoint_target"], 1)
            self.assertEqual((result.episode_dir / "tm_inputs.txt").read_text(), "0 press up\n50 press left")
            self.assertEqual(
                (result.episode_dir / "frames.rgb").read_bytes(),
                bytes([3, 2, 1, 30, 20, 10]) * 2,
            )
            steps = [json.loads(line) for line in (result.episode_dir / "steps.jsonl").read_text().splitlines()]
            self.assertEqual(steps[0]["action_gas_brake_steer"], [1.0, -1.0, 0.0])
            self.assertEqual(steps[1]["action_gas_brake_steer"], [1.0, -1.0, -1.0])
            self.assertEqual(steps[1]["telemetry"]["race_time_ms"], 50)
            self.assertEqual(steps[1]["telemetry"]["gear"], 2)
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["telemetry"]["max_race_time_offset_ms"], 0)
            # Telemetry is read before the frame request in the same step.
            self.assertLess(
                interface.calls.index(("get_simulation_state_bytes",)),
                interface.calls.index(("request_frame", 2, 1)),
            )
            schedule_no_ghost_setup.assert_called_once_with(123)
            self.assertEqual(interface.calls[-1], ("close",))
            self.assertEqual(interface.calls.count(("close",)), 1)
            self.assertLess(
                interface.calls.index(("get_inputs_text",)), interface.calls.index(("close",))
            )
            self.assertNotIn(
                ("respond", MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC), interface.calls
            )
            self.assertTrue(manifest["visual_environment"]["opponent_dialog_configured"])
            self.assertTrue(
                manifest["visual_environment"]["ghost_setup_completed_before_race"]
            )
            self.assertEqual(manifest["behavior_mode"], "fast")
            self.assertEqual(
                manifest["behavior_labeling"],
                {"method": "finish_time_threshold", "fast_if_below_ms": 26_000},
            )

    def test_waits_for_dialog_worker_at_race_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = RecordingConfig(
                output_root=Path(temporary) / "dataset",
                width=2,
                height=1,
                period_ms=50,
                min_frames=2,
            )
            interface = FakeInterface()
            delayed = Mock()
            delayed.done.side_effect = [False, False]
            delayed.result.return_value = True

            result = HumanRunRecorder(
                interface,
                config,
                FakeKeyboard(),
                schedule_no_ghost_setup=Mock(return_value=delayed),
            ).record_connected(game_pid=123)

            self.assertTrue(result.complete)
            delayed.result.assert_called_once_with(timeout=1.0)

    def test_dialog_setup_timeout_shuts_down_inside_run_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = RecordingConfig(
                output_root=Path(temporary) / "dataset",
                width=2,
                height=1,
                period_ms=50,
                min_frames=2,
            )
            interface = FakeInterface()
            stalled = Mock()
            stalled.done.side_effect = [False, False]
            stalled.result.side_effect = TimeoutError

            result = HumanRunRecorder(
                interface,
                config,
                FakeKeyboard(),
                schedule_no_ghost_setup=Mock(return_value=stalled),
            ).record_connected(game_pid=123)

            self.assertFalse(result.complete)
            self.assertEqual(result.reason, "opponent_dialog_setup_not_completed")
            self.assertEqual(interface.calls.count(("close",)), 1)
            self.assertNotIn(("respond", MessageType.SC_RUN_STEP_SYNC), interface.calls[-2:])

    @patch("trackmania_vla.human_recording.connect_tminterface")
    @patch("trackmania_vla.human_recording.wait_for_windowed_trackmania")
    @patch("trackmania_vla.human_recording.launch_trackmania")
    @patch("trackmania_vla.human_recording._tmforever_pids")
    def test_reuses_existing_game_without_launching(
        self, find_pids, launch, wait_for_window, connect
    ):
        find_pids.return_value = {4321}
        interface = Mock()
        connect.return_value = interface

        result = launch_and_connect(Path("unused-loader.exe"), "default", 8478)

        self.assertEqual(result, (interface, 4321))
        launch.assert_not_called()
        wait_for_window.assert_called_once()
        connect.assert_called_once_with(8478, 60.0)

    @patch("trackmania_vla.human_recording._tmforever_pids", return_value=set())
    def test_no_launch_requires_existing_game(self, _find_pids):
        with self.assertRaisesRegex(RuntimeError, "no TmForever process"):
            launch_and_connect(Path("unused-loader.exe"), "default", 8478, launch=False)


if __name__ == "__main__":
    unittest.main()
