"""What the frames hide: the car's geometry against the experts' racing line,
and the physical outcome of each state (for the world-model head).

``TrackFeatures`` turns one telemetry sample into prompt text: signed sideways
offset and heading versus the racing line, distance to the next jump, the
experts' speed at this point, vertical speed, pitch and whether the car is
airborne. ``outcome_targets`` labels every frame of a recorded lap with what
happened next: the sideways offset and speed change 1 s and 2 s later, and
whether the lap failed within 2 s (the car reached the furthest point it would
ever reach: it crashed, fell or stalled there).

Offsets are measured from where the experts actually drive: the racing line
is one expert lap, and at some places (around 500 m and 1,385 m) other
experts run up to 20 m beside it. ``ExpertEnvelope`` holds, every 5 m, the
experts' median offset from the line, how far they spread around it and how
low they go.

Sign conventions (x/z horizontal, y up): positive offset or heading means the
car is right of / pointing right of the line.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from .line_follower import RacingLine

JUMPS_M = (330.0, 1680.0)
FRAME_MS = 50
HORIZONS = (20, 40)  # frames: 1 s and 2 s ahead
FAILURE_NEAR_M = 20.0  # a failed lap "fails" once within this of its furthest point
BIN_M = 5.0
AUX_SIZE = 2 * len(HORIZONS) + 1  # offsets, speed changes, crash
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_LAPS = PROJECT_ROOT / "datasets" / "reference_laps" / "episodes"
# Position-tagged expert laps (record_replays.py output); override with TRACKVLA_EXPERT_LAPS.
EXPERT_LAPS = Path(os.environ.get("TRACKVLA_EXPERT_LAPS", PROJECT_ROOT / "datasets" / "replay_runs" / "episodes"))
ENVELOPE_CACHE = PROJECT_ROOT / "datasets" / "track_envelope.npz"


def default_racing_line() -> RacingLine:
    """A01's racing line: the first reference expert lap."""
    return RacingLine.from_episode(sorted(REFERENCE_LAPS.iterdir())[0])


class ExpertEnvelope:
    def __init__(self, median: np.ndarray, spread: np.ndarray, lowest: np.ndarray):
        self.median, self.spread, self.lowest = median, spread, lowest

    @classmethod
    def build(cls, line: RacingLine, episodes: list[Path]) -> "ExpertEnvelope":
        bins = int(line.distance[-1] // BIN_M) + 1
        offsets: list[list[float]] = [[] for _ in range(bins)]
        heights: list[list[float]] = [[] for _ in range(bins)]
        for episode in episodes:
            track = TrackFeatures(line)
            for row in (episode / "steps.jsonl").open(encoding="utf-8"):
                index, offset, height = track.locate(json.loads(row)["telemetry"])
                b = int(line.distance[index] // BIN_M)
                offsets[b].append(offset)
                heights[b].append(height)
        median = np.array([np.median(o) if o else 0.0 for o in offsets])
        spread = np.array([np.percentile(np.abs(np.asarray(o) - m), 99) if o else 0.0 for o, m in zip(offsets, median)])
        lowest = np.array([np.percentile(h, 1) if h else 0.0 for h in heights])
        # Widen over neighbouring bins so sparse bins do not make a narrow envelope.
        spread = np.maximum.reduce([np.roll(spread, k) for k in (-1, 0, 1)])
        lowest = np.minimum.reduce([np.roll(lowest, k) for k in (-1, 0, 1)])
        return cls(median, spread, lowest)

    def at(self, metre: float) -> tuple[float, float, float]:
        b = min(int(metre // BIN_M), len(self.median) - 1)
        return float(self.median[b]), float(self.spread[b]), float(self.lowest[b])


def default_envelope(line: RacingLine | None = None, laps: int = 100) -> ExpertEnvelope:
    if ENVELOPE_CACHE.exists():
        data = np.load(ENVELOPE_CACHE)
        return ExpertEnvelope(data["median"], data["spread"], data["lowest"])
    line = line or default_racing_line()
    episodes = sorted(p for p in EXPERT_LAPS.iterdir() if (p / "steps.jsonl").exists())[:laps]
    envelope = ExpertEnvelope.build(line, episodes)
    np.savez(ENVELOPE_CACHE, median=envelope.median, spread=envelope.spread, lowest=envelope.lowest)
    return envelope


def default_track() -> "TrackFeatures":
    line = default_racing_line()
    return TrackFeatures(line, default_envelope(line))


def _side(direction: np.ndarray, vector: np.ndarray) -> float:
    """> 0 when ``vector`` points to the right of ``direction`` (horizontal)."""
    return float(direction[0] * vector[2] - direction[2] * vector[0])


class TrackFeatures:
    def __init__(self, line: RacingLine, envelope: ExpertEnvelope | None = None, jumps_m=JUMPS_M):
        self.line = line
        self.envelope = envelope
        self.jumps_m = tuple(jumps_m)
        tangent = np.gradient(line.points, axis=0)
        tangent[:, 1] = 0.0
        self.tangent = tangent / np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
        self.hint: int | None = None
        self.last_time = -1

    def fresh(self) -> "TrackFeatures":
        """Same line and envelope, no lap state (one per episode)."""
        return TrackFeatures(self.line, self.envelope, self.jumps_m)

    def locate(self, telemetry: dict) -> tuple[int, float, float]:
        """(line index, signed sideways offset m, height above the line m)."""
        if int(telemetry["race_time_ms"]) < self.last_time:
            self.hint = None  # a new lap
        self.last_time = int(telemetry["race_time_ms"])
        position = np.asarray(telemetry["position_m"], float)
        index = self.line.nearest(position, self.hint)
        if np.linalg.norm(self.line.points[index] - position) > 25.0:
            index = self.line.nearest(position)
        self.hint = index
        delta = position - self.line.points[index]
        offset, height = _side(self.tangent[index], delta), float(delta[1])
        if self.envelope is not None:
            median, _, _ = self.envelope.at(float(self.line.distance[index]))
            offset -= median
        return index, offset, height

    def features(self, telemetry: dict) -> dict:
        index, offset, height = self.locate(telemetry)
        forward = np.asarray(telemetry["forward_world"], float)
        flat = np.array([forward[0], 0.0, forward[2]])
        flat_norm = np.linalg.norm(flat)
        heading = 0.0
        if flat_norm > 1e-6:
            flat /= flat_norm
            heading = math.degrees(math.atan2(_side(self.tangent[index], flat), float(flat @ self.tangent[index])))
        pitch = math.asin(max(-1.0, min(1.0, float(forward[1]))))
        _, up, ahead = telemetry["local_velocity_mps"]
        metre = float(self.line.distance[index])
        next_jump = next((j - metre for j in self.jumps_m if j >= metre), None)
        return {
            "metre": metre, "offset_m": offset, "height_m": height, "heading_deg": heading,
            "next_jump_m": next_jump, "jump_number": None if next_jump is None else self.jumps_m.index(metre + next_jump) + 1,
            "expert_kmh": float(self.line.speed_kmh[index]),
            "vertical_mps": up * math.cos(pitch) + ahead * math.sin(pitch),
            "pitch_deg": math.degrees(pitch),
            "airborne": not any(telemetry["wheels_ground_contact"]),
        }

    def text(self, telemetry: dict) -> str:
        f = self.features(telemetry)
        side = "right" if f["offset_m"] >= 0 else "left"
        turn = "right" if f["heading_deg"] >= 0 else "left"
        jump = (f"{f['next_jump_m']:.0f} m to jump {f['jump_number']}" if f["next_jump_m"] is not None
                else "past the last jump")
        return (
            f"{abs(f['offset_m']):.1f} m {side} of the experts' line, heading {abs(f['heading_deg']):.0f} deg {turn} of it, "
            f"{jump}, experts do {f['expert_kmh']:.0f} km/h here. "
            f"Vertical speed {f['vertical_mps']:+.1f} m/s, pitch {f['pitch_deg']:+.0f} deg, "
            f"{'airborne' if f['airborne'] else 'on the ground'}."
        )


def outcome_targets(track: TrackFeatures, steps: list[dict], finished: bool) -> np.ndarray:
    """[frames, AUX_SIZE]: offset 1 s/2 s ahead (/10 m), speed change 1 s/2 s
    ahead (/100 km/h), crash within 2 s (0/1). NaN where the lap has ended."""
    track = track.fresh()
    offsets, metres, speeds = [], [], []
    for step in steps:
        index, offset, _ = track.locate(step["telemetry"])
        offsets.append(offset)
        metres.append(float(track.line.distance[index]))
        speeds.append(float(step["telemetry"]["display_speed_kmh"]))
    offsets, metres, speeds = np.asarray(offsets), np.asarray(metres), np.asarray(speeds)
    n = len(steps)
    out = np.full((n, AUX_SIZE), np.nan)
    for k, horizon in enumerate(HORIZONS):
        valid = np.arange(n) + horizon < n
        out[valid, k] = np.clip(offsets[horizon:], -20, 20) / 10.0
        out[valid, len(HORIZONS) + k] = (speeds[horizon:] - speeds[: n - horizon]) / 100.0
    crash = np.zeros(n)
    if not finished and n:
        # The failure: the first time the car got near the furthest point it
        # ever reached. Everything after it (falling, stalled, stuck) is failed too.
        failed_at = int(np.argmax(metres >= metres.max() - FAILURE_NEAR_M))
        crash[max(0, failed_at - HORIZONS[-1]):] = 1.0
    out[:, -1] = crash
    return out
