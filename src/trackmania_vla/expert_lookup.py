"""Expert lookup: label any state with what the human experts did in the most
similar states of their replays.

Every frame of the position-tagged expert replays becomes a (state, action)
pair: position, horizontal heading and speed, with the experts' steering
amount (left/right duty over a centered 5-frame window, like the training
labels) and throttle. A query finds the ``k`` nearest expert frames within
``window_m`` metres along the racing line, by a scaled distance:

    metres between positions + ``heading_m`` per radian of heading difference
    + ``speed_m`` per 10 km/h of speed difference

and returns their distance-weighted steering amount and majority throttle.
Queries with no expert frame within ``max_distance`` (fallen off the track,
facing backwards) have no label. Unlike the hand-tuned followers this keeps
the experts' own tap timing and speed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .line_follower import RacingLine


def _heading(forward) -> np.ndarray:
    vector = np.array([forward[0], forward[2]], float)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-6 else np.array([1.0, 0.0])


class ExpertLookup:
    def __init__(self, line: RacingLine, episodes: list[Path], *, k: int = 8, window_m: float = 40.0,
                 heading_m: float = 20.0, speed_m: float = 0.2, max_distance: float = 25.0):
        self.line = line
        self.k = k
        self.window_m = window_m
        self.heading_m = heading_m
        self.speed_m = speed_m
        self.max_distance = max_distance
        positions, headings, speeds, duties, throttles, metres = [], [], [], [], [], []
        for episode in episodes:
            steps = [json.loads(row) for row in (episode / "steps.jsonl").open(encoding="utf-8")]
            if not steps or "position_m" not in steps[0]["telemetry"]:
                continue
            signs = np.array([(1 if s["controls"]["right"] else 0) - (1 if s["controls"]["left"] else 0) for s in steps], float)
            duty = np.convolve(signs, np.ones(5) / 5, mode="same")
            hint = None
            for step, value in zip(steps, duty):
                telemetry = step["telemetry"]
                position = np.asarray(telemetry["position_m"], float)
                index = line.nearest(position, hint)
                if np.linalg.norm(line.points[index] - position) > 25.0:
                    index = line.nearest(position)
                hint = index
                controls = step["controls"]
                positions.append(position)
                headings.append(_heading(telemetry["forward_world"]))
                speeds.append(float(telemetry["display_speed_kmh"]))
                duties.append(value)
                throttles.append(2 if controls["brake"] else (1 if controls["accelerate"] else 0))
                metres.append(float(line.distance[index]))
        order = np.argsort(metres)
        self.metres = np.asarray(metres)[order]
        self.positions = np.asarray(positions)[order]
        self.headings = np.asarray(headings)[order]
        self.speeds = np.asarray(speeds)[order]
        self.duties = np.asarray(duties)[order]
        self.throttles = np.asarray(throttles, np.int64)[order]
        self.hint: int | None = None
        self.last_duty: float | None = None
        self.last_distance: float | None = None
        self._error = 0.0

    def __len__(self) -> int:
        return len(self.metres)

    def query(self, telemetry: dict) -> tuple[float, int, float] | None:
        """(steering amount, throttle class, match distance) or None if no expert state is close."""
        position = np.asarray(telemetry["position_m"], float)
        index = self.line.nearest(position, self.hint)
        if np.linalg.norm(self.line.points[index] - position) > 25.0:
            index = self.line.nearest(position)
        self.hint = index
        metre = float(self.line.distance[index])
        lo = np.searchsorted(self.metres, metre - self.window_m)
        hi = np.searchsorted(self.metres, metre + self.window_m)
        if hi <= lo:
            return None
        heading = _heading(telemetry["forward_world"])
        cos = np.clip(self.headings[lo:hi] @ heading, -1.0, 1.0)
        distance = (
            np.linalg.norm(self.positions[lo:hi] - position, axis=1)
            + self.heading_m * np.arccos(cos)
            + self.speed_m * np.abs(self.speeds[lo:hi] - float(telemetry["display_speed_kmh"])) / 10.0
        )
        k = min(self.k, len(distance))
        nearest = np.argpartition(distance, k - 1)[:k]
        if distance[nearest].min() > self.max_distance:
            return None
        weights = 1.0 / (distance[nearest] + 0.5)
        duty = float(np.clip((self.duties[lo:hi][nearest] * weights).sum() / weights.sum(), -1.0, 1.0))
        throttle = int(np.bincount(self.throttles[lo:hi][nearest], weights=weights, minlength=3).argmax())
        return duty, throttle, float(distance[nearest].min())

    def action(self, telemetry: dict) -> tuple[int, int]:
        """Labeler interface: keys for this 50 ms frame (sigma-delta of the amount).

        The continuous amount is kept in ``last_duty`` (None when no expert
        state is close) so the driver can record it as the training label.
        """
        result = self.query(telemetry)
        if result is None:
            self.last_duty, self.last_distance = None, None
            return 1, 1
        duty, throttle, distance = result
        self.last_duty, self.last_distance = duty, distance
        self._error += abs(duty)
        press = self._error >= 0.5
        if press:
            self._error -= 1.0
        self._error = min(max(self._error, -1.0), 1.0)
        return ((0 if duty < 0 else 2) if press else 1), throttle
