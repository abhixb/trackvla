import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trackmania_vla.closed_loop import ChunkEnsemble, PolicyDriver, SigmaDelta, SteerFilter
from trackmania_vla.human_recording import RecordingConfig
from trackmania_vla.tmnf_protocol import MessageType

STEP, FRAME = MessageType.SC_RUN_STEP_SYNC, MessageType.SC_REQUESTED_FRAME_SYNC


class FakeGame:
    """Countdown, then race steps every 10 ms to 400 ms with a frame every 50 ms,
    then the finish checkpoint."""

    def __init__(self):
        script = [MessageType.SC_ON_CONNECT_SYNC, STEP, -10]
        for t in range(0, 410, 10):
            script += [STEP, t]
            if t % 50 == 0:
                script.append(FRAME)
        script += [MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC, 1, 1]
        self.values = iter(script)
        self.frame_times = iter(range(0, 410, 50))
        self.calls = []

    def read_int32(self):
        return next(self.values)

    def get_frame_bgra(self, width, height):
        return bytes(width * height * 4)

    def __getattr__(self, name):
        def record(*args):
            self.calls.append((name, *args))

        return record


def telemetry(game):
    return {
        "race_time_ms": next(game.frame_times), "display_speed_kmh": 200, "rpm": 8000.0, "gear": 4,
        "local_velocity_mps": [0.0, 0.0, 55.0], "local_angular_velocity_rps": [0.0, 0.1, 0.0],
        "wheels_ground_contact": [True] * 4, "car_sliding": False, "lateral_contact": False,
    }


class SlowGame(FakeGame):
    """Race to 6 s with frames every 50 ms, then the finish."""

    def __init__(self):
        script = [MessageType.SC_ON_CONNECT_SYNC, STEP, -10]
        for t in range(0, 6010, 10):
            script += [STEP, t]
            if t % 50 == 0:
                script.append(FRAME)
        script += [MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC, 1, 1]
        self.values = iter(script)
        self.frame_times = iter(range(0, 6010, 50))
        self.calls = []


def stopped_car(game):
    return {**telemetry(game), "display_speed_kmh": 0}


class ClosedLoopTests(unittest.TestCase):
    def test_a_stuck_car_ends_the_lap(self):
        game = SlowGame()
        planner = lambda *args: (np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]))
        with tempfile.TemporaryDirectory() as temporary:
            config = RecordingConfig(output_root=Path(temporary), width=2, height=1, min_frames=1)
            result = PolicyDriver(game, config, planner, telemetry_reader=stopped_car, stuck_ms=1000).drive()
            del config
        self.assertEqual(result.reason, "stuck")

    def test_plans_every_200ms_and_executes_the_chunk(self):
        planned_at = []

        def planner(previous, current, telemetry):
            planned_at.append(telemetry["display_speed_kmh"])
            # left, straight, right, straight with gas, gas, brake, coast
            return np.array([0, 1, 2, 1]), np.array([1, 1, 2, 0])

        game = FakeGame()
        with tempfile.TemporaryDirectory() as temporary:
            config = RecordingConfig(output_root=Path(temporary), width=2, height=1, min_frames=1)
            result = PolicyDriver(game, config, planner, telemetry_reader=telemetry).drive()
            del config

        self.assertTrue(result.finished)
        self.assertEqual(result.plans, 2)  # at 200 and 400 ms (100 ms history needed)
        inputs = [c[1:] for c in game.calls if c[0] == "set_input_state"]
        # Before the first plan: straight with gas.
        self.assertEqual(inputs[0], (False, False, True, False))
        # The first plan's actions appear in order: left, straight, right+brake, coast.
        self.assertIn((True, False, True, False), inputs)
        self.assertIn((False, True, False, True), inputs)
        self.assertIn((False, False, False, False), inputs)
        self.assertEqual(planned_at[0], 200)

    def test_probability_planner_averages_chunks_and_modulates_steering(self):
        calls = []

        def planner(*args):
            *frames, telemetry = args
            calls.append(len(frames))
            # Every slot: 50% "half right" (+0.5), 50% "full right" (+1) -> amount 0.75; gas.
            steer = np.tile([0.0, 0.0, 0.0, 0.5, 0.5], (4, 1))
            throttle = np.tile([0.0, 1.0, 0.0], (4, 1))
            return steer, throttle

        planner.steer_values = np.linspace(-1, 1, 5)
        game = FakeGame()
        with tempfile.TemporaryDirectory() as temporary:
            config = RecordingConfig(output_root=Path(temporary), width=2, height=1, min_frames=1)
            result = PolicyDriver(game, config, planner, history=3, replan_ms=50, telemetry_reader=telemetry).drive()
            del config

        self.assertTrue(result.finished)
        self.assertEqual(set(calls), {3})
        self.assertEqual(result.plans, 5)  # 200..400 ms: history 3 needs 200 ms of frames
        inputs = [c[1:] for c in game.calls if c[0] == "set_input_state"]
        after_first_plan = inputs[inputs.index((False, True, True, False)):]
        right = sum(1 for left, r, gas, brake in after_first_plan if r)
        self.assertFalse(any(left for left, *_ in after_first_plan))
        self.assertGreater(right / len(after_first_plan), 0.6)
        self.assertLess(right / len(after_first_plan), 0.9)

    def test_chunk_ensemble_and_sigma_delta(self):
        ensemble = ChunkEnsemble(np.array([-1.0, 0.0, 1.0]), 50)
        ensemble.add(0, np.tile([1.0, 0.0, 0.0], (4, 1)), np.tile([0.0, 1.0, 0.0], (4, 1)))
        ensemble.add(50, np.tile([0.0, 1.0, 0.0], (4, 1)), np.tile([0.0, 0.0, 1.0], (4, 1)))
        self.assertEqual(ensemble.action(0), (-1.0, 1))
        duty, _ = ensemble.action(50)
        self.assertAlmostEqual(duty, -0.5)
        self.assertIsNone(ensemble.action(1000))
        # Mostly straight with a right tail: expected taps, argmax/rounded do not.
        probabilities = np.tile([0.0, 0.0, 0.7, 0.2, 0.1], (4, 1))
        values = np.linspace(-1, 1, 5)
        for rule, deadzone, want in (("expected", 0.0, 0.2), ("expected", 0.25, 0.0), ("argmax", 0.0, 0.0), ("rounded", 0.0, 0.0)):
            rules = ChunkEnsemble(values, 50, rule, deadzone)
            rules.add(0, probabilities, np.tile([0.0, 1.0, 0.0], (4, 1)))
            self.assertAlmostEqual(rules.action(0)[0], want, msg=rule)
        with self.assertRaises(ValueError):
            ChunkEnsemble(values, 50, "median")
        modulator = SigmaDelta()
        presses = [modulator.step(-0.3) for _ in range(100)]
        self.assertEqual(sum(left for left, _ in presses), 30)
        self.assertFalse(any(right for _, right in presses))


class SteerFilterTests(unittest.TestCase):
    def test_small_opposite_corrections_go_straight(self):
        f = SteerFilter(50, switch=0.4, deadzone=0.2)
        self.assertAlmostEqual(f(0.5), 0.5)    # steering right
        self.assertEqual(f(-0.3), 0.0)         # small left tap suppressed
        self.assertAlmostEqual(f(-0.6), -0.6)  # a real left turn is taken

    def test_side_released_after_quiet_time(self):
        f = SteerFilter(50, switch=0.4, deadzone=0.2, release_ms=100)
        f(0.5); f(0.0); f(0.0)                 # 100 ms inside the deadzone
        self.assertAlmostEqual(f(-0.3), -0.3)  # left is allowed again

    def test_smoothing_averages(self):
        f = SteerFilter(50, smooth_ms=50)      # alpha 0.5
        self.assertAlmostEqual(f(1.0), 0.5)
        self.assertAlmostEqual(f(-1.0), -0.25)


if __name__ == "__main__":
    unittest.main()
