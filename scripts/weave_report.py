"""Weaving and jump take-off metrics for solo-lap evaluations.

  python scripts/weave_report.py EPISODE_GLOB [EPISODE_GLOB ...]

Each glob selects episode directories (containing steps.jsonl) of one run;
the reference expert laps are reported first for comparison. Metrics:
finish rate; steering-key share and wall-contact share on the opening
straights; speed at 200 m; speed and distance off the racing line at the
take-off of jump 1 (330 m) and jump 2 (1680 m).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trackmania_vla.line_follower import RacingLine  # noqa: E402

REFERENCE = PROJECT_ROOT / "datasets" / "reference_laps" / "episodes"
STRAIGHTS_M = (0, 350)
SCRAPE_M = (150, 250)
JUMPS_M = {"jump 1": 330, "jump 2": 1680}


def lap_metrics(line: RacingLine, episode: Path) -> dict:
    steps = [json.loads(l) for l in (episode / "steps.jsonl").open(encoding="utf-8")]
    manifest = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
    hint = None
    rows = []
    for step in steps:
        position = np.asarray(step["telemetry"]["position_m"], float)
        index = line.nearest(position, hint)
        if np.linalg.norm(line.points[index] - position) > 25:
            index = line.nearest(position)
        hint = index
        offset = float(np.hypot(*(line.points[index] - position)[[0, 2]]))
        controls = step["controls"]
        rows.append((index, offset, step["telemetry"]["display_speed_kmh"],
                     controls["left"] or controls["right"], step["telemetry"]["lateral_contact"]))
    metres = np.array([r[0] for r in rows])
    out = {"finished": manifest.get("status") == "complete"}
    on_straights = (metres >= STRAIGHTS_M[0]) & (metres < STRAIGHTS_M[1])
    on_scrape = (metres >= SCRAPE_M[0]) & (metres < SCRAPE_M[1])
    out["steer_share"] = float(np.mean([r[3] for r, m in zip(rows, on_straights) if m])) if on_straights.any() else None
    out["wall_share"] = float(np.mean([r[4] for r, m in zip(rows, on_scrape) if m])) if on_scrape.any() else None
    for name, metre in {"speed_200": 200, **JUMPS_M}.items():
        hit = next((r for r in rows if r[0] >= metre), None)
        out[name] = None if hit is None else (hit[2], hit[1])
    return out


def summarize(name: str, laps: list[dict]) -> None:
    def mean(values):
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else float("nan")

    finished = sum(l["finished"] for l in laps)
    print(f"\n{name}: {len(laps)} laps, finished {finished}/{len(laps)} ({finished / max(1, len(laps)):.0%})")
    print(f"  steering key on 0-350 m: {mean(l['steer_share'] for l in laps):.0%}   "
          f"wall contact 150-250 m: {mean(l['wall_share'] for l in laps):.0%}   "
          f"speed at 200 m: {mean(l['speed_200'][0] if l['speed_200'] else None for l in laps):.0f} km/h")
    for jump in JUMPS_M:
        reached = [l[jump] for l in laps if l[jump]]
        print(f"  {jump} take-off: {mean(r[0] for r in reached):.0f} km/h, {mean(r[1] for r in reached):.1f} m off-line (n={len(reached)})")


def main() -> int:
    line = RacingLine.from_episode(sorted(REFERENCE.iterdir())[0])
    summarize("experts (reference laps)", [lap_metrics(line, d) for d in sorted(REFERENCE.iterdir())])
    for pattern in sys.argv[1:]:
        # Interrupted runs (no manifest) are not laps.
        episodes = sorted({Path(p).parent for p in glob.glob(str(Path(pattern) / "manifest.json"))})
        summarize(pattern, [lap_metrics(line, d) for d in episodes])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
