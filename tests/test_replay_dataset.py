import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trackmania_vla.replay_dataset import (
    ChunkDataset,
    duty_to_bin,
    load_episodes,
    mild_class_weights,
    shift_image,
    split_episodes,
    steer_class,
    steer_duty,
    telemetry_text,
    throttle_class,
)

TELEMETRY = {
    "race_time_ms": 0,
    "display_speed_kmh": 312,
    "speed_mps": 86.7,
    "rpm": 8500.4,
    "gear": 4,
    "local_velocity_mps": [0.42, 0.0, 86.7],
    "local_angular_velocity_rps": [0.0, -0.12, 0.0],
    "wheels_ground_contact": [True, True, False, True],
    "wheels_sliding": [False] * 4,
    "car_sliding": False,
    "lateral_contact": True,
}


def write_episode(root: Path, name: str, steers: list[str]) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    n = len(steers)
    (directory / "frames.rgb").write_bytes(np.arange(n, dtype=np.uint8).repeat(4 * 2 * 3).tobytes())
    with (directory / "steps.jsonl").open("w") as steps:
        for i, steer in enumerate(steers):
            controls = {"left": steer == "L", "right": steer == "R", "accelerate": True, "brake": i == 3}
            steps.write(json.dumps({"race_time_ms": 50 * i, "controls": controls, "telemetry": TELEMETRY}) + "\n")
    (directory / "manifest.json").write_text(json.dumps({
        "status": "complete", "schema_version": 2, "behavior_mode": "fast",
        "source": {"replay_id": name}, "capture": {"frame_count": n, "width": 4, "height": 2},
    }))


class ReplayDatasetTests(unittest.TestCase):
    def test_class_mapping_and_telemetry_text(self):
        self.assertEqual(steer_class({"left": True, "right": False}), 0)
        self.assertEqual(steer_class({"left": False, "right": True}), 2)
        self.assertEqual(steer_class({"left": True, "right": True}), 1)
        self.assertEqual(throttle_class({"accelerate": True, "brake": True}), 2)
        self.assertEqual(throttle_class({"accelerate": False, "brake": False}), 0)
        text = telemetry_text(TELEMETRY)
        self.assertIn("Speed 312 km/h", text)
        self.assertIn("sideways +0.4 m/s", text)
        self.assertIn("yaw rate -0.12 rad/s", text)
        self.assertIn("wheels on ground 3/4", text)
        self.assertIn("wall contact yes", text)

    def test_samples_pair_frames_with_future_action_chunk(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_episode(root, "a", list("SSLLRRSS"))
            dataset = ChunkDataset(load_episodes(root), stride=2, chunk=4)
            # t ranges over 2..4 for 8 frames.
            self.assertEqual(len(dataset), 3)
            sample = dataset[0]
            self.assertEqual(np.asarray(sample.frames[0])[0, 0, 0], 0)
            self.assertEqual(np.asarray(sample.frames[1])[0, 0, 0], 2)
            self.assertEqual(sample.steer.tolist(), [0, 0, 2, 2])
            self.assertEqual(sample.throttle.tolist(), [1, 2, 1, 1])
            steer_counts, _ = dataset.label_counts()
            self.assertEqual(int(steer_counts.sum()), 12)
            del sample, dataset  # release the memory map so Windows can delete the file

    def test_split_is_by_whole_episode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for k in range(10):
                write_episode(root, f"e{k}", list("SSSSSS"))
            splits = split_episodes(load_episodes(root), validation=2, test=2)
            ids = [{e.replay_id for e in splits[name]} for name in ("train", "validation", "test")]
            self.assertEqual([len(s) for s in ids], [6, 2, 2])
            self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])

    def test_mild_weights_are_sqrt_capped_and_normalized(self):
        counts = np.array([100, 10_000, 1])
        uncapped = mild_class_weights(counts, cap=1e9)
        self.assertAlmostEqual(uncapped[0] / uncapped[1], 10.0, places=3)  # sqrt of a 100x ratio
        capped = mild_class_weights(counts, cap=10.0)
        frequency = counts / counts.sum()
        self.assertAlmostEqual(float((capped * frequency).sum()), 1.0, places=5)
        self.assertAlmostEqual(capped[0], capped[2], places=5)  # both rare classes hit the cap

    def test_steer_duty_is_a_centered_window_mean(self):
        signs = np.array([0, -1, -1, 0, 1, 1, 1], float)
        duty = steer_duty(signs, 3)
        self.assertAlmostEqual(duty[0], -0.5)  # clipped window: frames 0..1
        self.assertAlmostEqual(duty[2], -2 / 3)
        self.assertAlmostEqual(duty[5], 1.0)
        self.assertEqual(duty_to_bin(np.array([-1.0, -0.4, 0.1, 0.6, 1.0]), 5).tolist(), [0, 1, 2, 3, 4])

    def test_v2_samples_have_history_and_steering_amount(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_episode(root, "a", list("SSLLLLSSRRRRSS"))
            dataset = ChunkDataset(load_episodes(root), stride=2, chunk=4, history=4, steer_window=5, steer_bins=5, turn_hint=True)
            # t starts at (4-1)*2 = 6 and ends at 14-4 = 10.
            self.assertEqual(len(dataset), 5)
            sample = dataset[0]
            self.assertEqual([np.asarray(f)[0, 0, 0] for f in sample.frames], [0, 2, 4, 6])
            # duty at 6..9 (windows 4..8, 5..9, ...): -0.2, +0.2, +0.6, +0.8
            self.assertEqual(sample.steer.tolist(), [2, 2, 3, 4])
            self.assertIn("turning", sample.telemetry)
            del sample, dataset

    def test_shift_augmentation_moves_frames_and_corrects_steering(self):
        image = Image.fromarray(np.arange(8, dtype=np.uint8).reshape(1, 8, 1).repeat(3, axis=2), "RGB")
        self.assertEqual(np.asarray(shift_image(image, 2))[0, :, 0].tolist(), [0, 0, 0, 1, 2, 3, 4, 5])
        self.assertEqual(np.asarray(shift_image(image, -2))[0, :, 0].tolist(), [2, 3, 4, 5, 6, 7, 7, 7])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_episode(root, "a", list("SSSSSSSSSS"))
            # Every frame shows columns 10, 20, 30, 40 so the shift is visible.
            columns = np.array([10, 20, 30, 40], np.uint8)
            (root / "a" / "frames.rgb").write_bytes(np.tile(columns[None, None, :, None], (10, 2, 1, 3)).tobytes())
            dataset = ChunkDataset(
                load_episodes(root), stride=2, chunk=4, history=2, steer_window=5, steer_bins=5,
                shift_fraction=0.5, shift_probability=1.0, shift_gain=2.0, seed=1,
            )
            levels = {}
            for _ in range(60):
                sample = dataset[0]
                row = np.asarray(sample.frames[1])[0, :, 0]
                dx = int((row == 10).sum() - 1) if row[0] == 10 else -int((row == 40).sum() - 1)
                self.assertTrue((np.asarray(sample.frames[0]) == np.asarray(sample.frames[1])).all())
                levels[dx] = int(sample.steer[0])
                self.assertTrue(all(int(s) == levels[dx] for s in sample.steer))
            # Width 4, gain 2: correction dx/2 -> content right steers right.
            self.assertEqual(levels, {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4})
            del sample, dataset

class ExpertDutyLabelTests(unittest.TestCase):
    def test_lookup_labels_replace_keys_and_missing_labels_are_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_episode(root, "a", list("SSSSSSSSSSSS"))
            steps_path = root / "a" / "steps.jsonl"
            rows = [json.loads(line) for line in steps_path.read_text().splitlines()]
            for i, row in enumerate(rows):
                row["expert_duty"] = None if i == 9 else 1.0  # full right, except one unlabeled frame
            steps_path.write_text("\n".join(json.dumps(r) for r in rows))
            dataset = ChunkDataset(load_episodes(root), stride=2, chunk=4, history=2, steer_window=5, steer_bins=5)
            # t in 2..8; chunks touching frame 9 (t = 6, 7, 8) have no label.
            self.assertEqual(sorted(t for _, t in dataset.index), [2, 3, 4, 5])
            self.assertEqual(dataset[0].steer.tolist(), [4, 4, 4, 4])
            del dataset


if __name__ == "__main__":
    unittest.main()
