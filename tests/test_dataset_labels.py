import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trackmania_vla.config import behavior_prompt
from trackmania_vla.dataset_labels import mode_from_finish_time, retag_episode_manifests


class DatasetLabelTests(unittest.TestCase):
    def test_threshold_is_strict(self):
        self.assertEqual(mode_from_finish_time(25_999), "fast")
        self.assertEqual(mode_from_finish_time(26_000), "safe")

    def test_retags_mode_and_language_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, finish_ms in (("quick", 25_950), ("steady", 26_000)):
                episode = root / name
                episode.mkdir()
                (episode / "manifest.json").write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "behavior_mode": "safe",
                            "language_instruction": behavior_prompt("safe"),
                            "finish_detection": {
                                "race_time_ms_at_last_sample": finish_ms,
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            summary = retag_episode_manifests(root)

            self.assertEqual((summary.total, summary.fast, summary.safe, summary.changed), (2, 1, 1, 2))
            quick = json.loads((root / "quick" / "manifest.json").read_text(encoding="utf-8"))
            steady = json.loads((root / "steady" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(quick["behavior_mode"], "fast")
            self.assertEqual(quick["language_instruction"], behavior_prompt("fast"))
            self.assertEqual(
                quick["behavior_labeling"],
                {"method": "finish_time_threshold", "fast_if_below_ms": 26_000},
            )
            self.assertEqual(steady["behavior_mode"], "safe")

    def test_manual_override_survives_retag(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary) / "crashed_fast_lap"
            episode.mkdir()
            labeling = {"method": "manual_override", "reason": "fast until a late crash"}
            (episode / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "behavior_mode": "fast",
                        "language_instruction": behavior_prompt("fast"),
                        "behavior_labeling": labeling,
                        "finish_detection": {"race_time_ms_at_last_sample": 29_500},
                    }
                ),
                encoding="utf-8",
            )

            summary = retag_episode_manifests(Path(temporary))

            manifest = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual((summary.fast, summary.changed), (1, 0))
            self.assertEqual(manifest["behavior_mode"], "fast")
            self.assertEqual(manifest["behavior_labeling"], labeling)


if __name__ == "__main__":
    unittest.main()
