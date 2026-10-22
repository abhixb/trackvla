"""Reject laps whose start frame shows an opponent ghost in the HUD.

Every replay of A01-Race starts from the same spawn with the same camera, so
frame 0 is pixel-identical across clean laps. A selected opponent (Nadeo or a
medal ghost) adds a second entry to the top-left HUD list, which shows up as
a large difference in that strip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

REFERENCE = Path(__file__).with_name("a01_start_hud_row2.png")
# Second HUD row of a 160x120 capture, where an opponent entry appears.
HUD_ROW2 = (slice(10, 24), slice(0, 60))
# Clean laps differ by ~0; opponent entries by ~34-37 (mean absolute RGB).
MAX_MEAN_ABS_DIFF = 5.0


def start_hud_difference(episode_dir: Path, width: int = 160, height: int = 120) -> float:
    size = width * height * 3
    with open(episode_dir / "frames.rgb", "rb") as frames:
        first = np.frombuffer(frames.read(size), np.uint8).reshape(height, width, 3)
    reference = np.asarray(Image.open(REFERENCE).convert("RGB"), np.int16)
    return float(np.abs(first[HUD_ROW2].astype(np.int16) - reference).mean())


def opponent_ghost_at_start(episode_dir: Path) -> bool:
    return start_hud_difference(episode_dir) > MAX_MEAN_ABS_DIFF
