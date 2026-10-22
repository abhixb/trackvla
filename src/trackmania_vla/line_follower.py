"""Racing-line follower: the automatic expert that labels DAgger states.

The racing line is the car's path in an expert replay lap, resampled every
metre with the expert's speed at each point. From any state the follower
looks a speed-dependent distance ahead along the line (pure pursuit, Coulter
1992) and steers toward that point with the three digital steer classes;
throttle tracks a fraction of the expert speed at the nearest line point.
Unlike a replay it gives a sensible action off the line, which is exactly
what the policy never saw.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np


@dataclass
class RacingLine:
    points: np.ndarray  # [N, 3] world positions, ~1 m apart
    speed_kmh: np.ndarray  # [N]
    distance: np.ndarray  # [N] cumulative metres

    @classmethod
    def from_episode(cls, episode: Path, spacing_m: float = 1.0) -> "RacingLine":
        steps = [json.loads(line) for line in (episode / "steps.jsonl").open(encoding="utf-8")]
        raw = np.array([step["telemetry"]["position_m"] for step in steps], float)
        speed = np.array([step["telemetry"]["display_speed_kmh"] for step in steps], float)
        keep = np.concatenate([[True], np.linalg.norm(np.diff(raw, axis=0), axis=1) > 1e-3])
        raw, speed = raw[keep], speed[keep]
        cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))])
        samples = np.arange(0.0, cumulative[-1], spacing_m)
        points = np.stack([np.interp(samples, cumulative, raw[:, k]) for k in range(3)], axis=1)
        return cls(points, np.interp(samples, cumulative, speed), samples)

    def nearest(self, position: np.ndarray, hint: int | None = None, window: int = 120) -> int:
        if hint is None:
            lo, hi = 0, len(self.points)
        else:
            lo, hi = max(0, hint - window // 4), min(len(self.points), hint + window)
        offsets = np.linalg.norm(self.points[lo:hi] - position, axis=1)
        return lo + int(offsets.argmin())


@dataclass
class FollowerConfig:
    lookahead_s: float = 0.4  # look this far ahead in time at current speed
    min_lookahead_m: float = 8.0
    max_lookahead_m: float = 45.0
    deadband_deg: float = 2.0  # |angle| below this -> straight
    left_sign: int = 1  # +1 if a positive heading-to-target angle means "turn left"
    speed_factor: float = 1.0  # fraction of expert speed to aim for
    coast_margin_kmh: float = 15.0
    brake_margin_kmh: float = 40.0


class LineFollower:
    def __init__(self, line: RacingLine, config: FollowerConfig | None = None):
        self.line = line
        self.config = config or FollowerConfig()
        self.hint: int | None = None

    def reset(self) -> None:
        self.hint = None

    def target_angle_deg(self, telemetry: dict) -> tuple[float, int]:
        position = np.asarray(telemetry["position_m"], float)
        forward = np.asarray(telemetry["forward_world"], float)
        index = self.line.nearest(position, self.hint)
        # Fall back to a global search if the windowed match is far (respawn, cut).
        if np.linalg.norm(self.line.points[index] - position) > 25.0:
            index = self.line.nearest(position)
        self.hint = index
        speed = float(telemetry["speed_mps"])
        lookahead = min(max(speed * self.config.lookahead_s, self.config.min_lookahead_m), self.config.max_lookahead_m)
        target_distance = self.line.distance[index] + lookahead
        target_index = min(int(np.searchsorted(self.line.distance, target_distance)), len(self.line.points) - 1)
        to_target = self.line.points[target_index] - position
        # Horizontal plane is x/z (y is up).
        cross = forward[0] * to_target[2] - forward[2] * to_target[0]
        dot = forward[0] * to_target[0] + forward[2] * to_target[2]
        return math.degrees(math.atan2(cross, dot)), index

    def action(self, telemetry: dict) -> tuple[int, int]:
        """(steer class 0=left 1=straight 2=right, throttle 0=coast 1=gas 2=brake)."""
        angle, index = self.target_angle_deg(telemetry)
        signed = angle * self.config.left_sign
        if signed > self.config.deadband_deg:
            steer = 0
        elif signed < -self.config.deadband_deg:
            steer = 2
        else:
            steer = 1
        target_speed = self.line.speed_kmh[index] * self.config.speed_factor
        speed = float(telemetry["display_speed_kmh"])
        if speed > target_speed + self.config.brake_margin_kmh:
            throttle = 2
        elif speed > target_speed + self.config.coast_margin_kmh:
            throttle = 0
        else:
            throttle = 1
        return steer, throttle



class GuidedFollower:
    """Expert-guided automatic driver: the expert laps' own steering along the
    line (feed-forward) plus a proportional pull back toward the line.

    The feed-forward is the reference laps' steering amount (left/right duty
    over a centered 5-frame window) averaged per metre of the racing line, so
    corrections keep the experts' timing; the pure-pursuit heading error adds
    ``gain_per_deg`` of steering per degree beyond ``dead_deg``. Every key
    tap scrubs speed, so small errors are deliberately left alone. The amount is played as key
    presses on the 50 ms frames by a sigma-delta modulator, like the policy.
    """

    def __init__(self, line: RacingLine, reference_episodes: list[Path], config: FollowerConfig | None = None,
                 gain_per_deg: float = 0.08, dead_deg: float = 0.0, smooth_m: int = 5):
        self.line = line
        self.inner = LineFollower(line, config)
        self.gain_per_deg = gain_per_deg
        self.dead_deg = dead_deg
        total = np.zeros(len(line.points))
        count = np.zeros(len(line.points))
        for episode in reference_episodes:
            steps = [json.loads(row) for row in (episode / "steps.jsonl").open(encoding="utf-8")]
            signs = np.array([(1 if s["controls"]["right"] else 0) - (1 if s["controls"]["left"] else 0) for s in steps], float)
            kernel = np.ones(5) / 5
            duty = np.convolve(signs, kernel, mode="same")
            hint = None
            for step, value in zip(steps, duty):
                index = line.nearest(np.asarray(step["telemetry"]["position_m"], float), hint)
                hint = index
                total[index] += value
                count[index] += 1
        known = count > 0
        profile = np.interp(np.arange(len(total)), np.flatnonzero(known), total[known] / count[known])
        self.profile = np.convolve(profile, np.ones(smooth_m) / smooth_m, mode="same")
        self.error = 0.0

    def duty(self, telemetry: dict) -> float:
        angle, index = self.inner.target_angle_deg(telemetry)
        toward_left = angle * self.inner.config.left_sign  # > 0: the line is to the left
        excess = math.copysign(max(abs(toward_left) - self.dead_deg, 0.0), toward_left)
        return float(np.clip(self.profile[index] - self.gain_per_deg * excess, -1.0, 1.0))

    def action(self, telemetry: dict) -> tuple[int, int]:
        """(steer class 0=left 1=straight 2=right, throttle class) for this 50 ms frame."""
        duty = self.duty(telemetry)
        _, throttle = self.inner.action(telemetry)
        self.error += abs(duty)
        press = self.error >= 0.5
        if press:
            self.error -= 1.0
        self.error = min(max(self.error, -1.0), 1.0)
        steer = (0 if duty < 0 else 2) if press else 1
        return steer, throttle
