import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trackmania_vla.config import PolicyConfig, behavior_prompt


class ConfigTests(unittest.TestCase):
    def test_modes_have_distinct_prompts(self):
        prompts = [behavior_prompt(mode) for mode in ("fast", "safe", "recover")]
        self.assertEqual(len(set(prompts)), 3)
        self.assertIn("fast lap", prompts[0])

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            behavior_prompt("anything")

    def test_history_requires_motion(self):
        with self.assertRaises(ValueError):
            PolicyConfig(history=1)


if __name__ == "__main__":
    unittest.main()
