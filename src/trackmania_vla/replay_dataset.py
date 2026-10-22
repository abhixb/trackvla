"""Training samples from replay-recorded episodes (plan v5, Phase 1).

A sample is ``history`` RGB frames (``stride`` captures apart, ending at the
current one), the current telemetry rendered as text, and the next ``chunk``
actions as factorized classes. Steering is either the held key (3 classes) or,
with ``steer_window``, the steering amount: the left/right duty over a centered
window of frames, binned into ``steer_bins`` levels from full left to full
right. Expert taps are too short to time from a frame; the duty is not. Episodes are split whole, so validation and test laps are
never seen in training. Frames are read lazily from each ``frames.rgb``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image

STEER_CLASSES = ("left", "straight", "right")
IGNORE = -100  # cross-entropy ignore_index: no driving label
THROTTLE_CLASSES = ("coast", "gas", "brake")


def steer_class(controls: dict) -> int:
    if controls["left"] and not controls["right"]:
        return 0
    if controls["right"] and not controls["left"]:
        return 2
    return 1


def throttle_class(controls: dict) -> int:
    # Rare simultaneous gas+brake maps to brake, as in the earlier head.
    if controls["brake"]:
        return 2
    return 1 if controls["accelerate"] else 0


def telemetry_text(telemetry: dict, *, turn_hint: bool = False, track=None) -> str:
    """Exact numeric state for the prompt; vision handles geometry.

    ``turn_hint`` adds the lateral acceleration (forward speed x yaw rate) in g,
    a speed-scaled measure of how hard the car is already turning.
    """
    lateral, _, forward = telemetry["local_velocity_mps"]
    yaw_rate = telemetry["local_angular_velocity_rps"][1]
    wheels = sum(bool(contact) for contact in telemetry["wheels_ground_contact"])
    hint = f"turning {forward * yaw_rate / 9.81:+.2f} g, " if turn_hint else ""
    # ``track`` (a TrackFeatures) adds the racing-line geometry the frames hide.
    line = f" {track.text(telemetry)}" if track is not None else ""
    return (
        f"Speed {telemetry['display_speed_kmh']} km/h, RPM {telemetry['rpm']:.0f}, "
        f"gear {telemetry['gear']}, forward {forward:.1f} m/s, sideways {lateral:+.1f} m/s, "
        f"yaw rate {yaw_rate:+.2f} rad/s, {hint}wheels on ground {wheels}/4, "
        f"sliding {'yes' if telemetry['car_sliding'] else 'no'}, "
        f"wall contact {'yes' if telemetry['lateral_contact'] else 'no'}.{line}"
    )


@dataclass(frozen=True)
class Episode:
    directory: Path
    replay_id: str
    mode: str
    frame_count: int
    width: int
    height: int


def load_episodes(root: Path) -> list[Episode]:
    episodes = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or manifest.get("schema_version") != 2:
            continue
        capture = manifest["capture"]
        episodes.append(
            Episode(
                directory=manifest_path.parent,
                replay_id=str((manifest.get("source") or {}).get("replay_id", manifest_path.parent.name)),
                mode=str(manifest["behavior_mode"]),
                frame_count=int(capture["frame_count"]),
                width=int(capture["width"]),
                height=int(capture["height"]),
            )
        )
    return episodes


def load_dagger_episodes(root: Path, *, min_frames: int = 20, finished_only: bool = False,
                         pattern: str = "round_*/*/*/manifest.json", map_id: str | None = None) -> list[Episode]:
    """Closed-loop DAgger laps (finished or not) whose frames carry expert labels.

    Also loads the line follower's own "safe" laps (``finished_only``) and,
    with a recursive ``pattern``, any policy laps (e.g. evaluations)."""
    episodes = []
    for manifest_path in sorted(root.glob(pattern)):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("source") or {}).get("kind") != "policy_closed_loop":
            continue
        if manifest.get("reason") == "opponent_ghost_visible":
            continue
        if map_id is not None and manifest.get("map_id") != map_id:
            continue
        if finished_only and manifest.get("status") != "complete":
            continue
        capture = manifest["capture"]
        if int(capture["frame_count"]) < min_frames:
            continue
        episodes.append(
            Episode(
                directory=manifest_path.parent,
                replay_id=manifest_path.parent.name,
                mode=str(manifest["behavior_mode"]),
                frame_count=int(capture["frame_count"]),
                width=int(capture["width"]),
                height=int(capture["height"]),
            )
        )
    return episodes


def has_positions(episode: Episode) -> bool:
    """Laps recorded before world positions were logged cannot get line features."""
    with (episode.directory / "steps.jsonl").open(encoding="utf-8") as steps:
        first = steps.readline()
    return bool(first) and "position_m" in json.loads(first)["telemetry"]


def split_episodes(
    episodes: list[Episode], *, validation: int = 20, test: int = 20, seed: int = 0
) -> dict[str, list[Episode]]:
    shuffled = list(episodes)
    random.Random(seed).shuffle(shuffled)
    return {
        "validation": shuffled[:validation],
        "test": shuffled[validation : validation + test],
        "train": shuffled[validation + test :],
    }


def steer_sign(controls: dict) -> int:
    """-1 left, 0 none, +1 right."""
    return steer_class(controls) - 1


def steer_duty(signs: np.ndarray, window: int) -> np.ndarray:
    """Mean steer sign over a centered window (clipped at the lap ends)."""
    half = window // 2
    padded = np.concatenate([[0.0], np.cumsum(signs, dtype=np.float64)])
    n = len(signs)
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    return (padded[hi] - padded[lo]) / (hi - lo)


def steer_bin_values(bins: int) -> np.ndarray:
    return np.linspace(-1.0, 1.0, bins)


def duty_to_bin(duty, bins: int) -> np.ndarray:
    values = steer_bin_values(bins)
    return np.abs(np.asarray(duty, np.float64)[..., None] - values).argmin(-1).astype(np.int64)


def shift_image(image: Image.Image, dx: int) -> Image.Image:
    """Move content ``dx`` pixels right (left if negative), repeating the edge column."""
    if dx == 0:
        return image
    array = np.asarray(image)
    if dx > 0:
        array = np.concatenate([np.repeat(array[:, :1], dx, axis=1), array[:, :-dx]], axis=1)
    else:
        array = np.concatenate([array[:, -dx:], np.repeat(array[:, -1:], -dx, axis=1)], axis=1)
    return Image.fromarray(np.ascontiguousarray(array), "RGB")


class _EpisodeData:
    def __init__(self, episode: Episode, *, steer_window: int = 0, steer_bins: int = 3, turn_hint: bool = False,
                 track=None, line_hint: bool = False, aux: bool = False, aux_only: bool = False):
        self.episode = episode
        steps = [json.loads(line) for line in (episode.directory / "steps.jsonl").open(encoding="utf-8")]
        if len(steps) != episode.frame_count:
            raise ValueError(f"{episode.directory}: steps do not match frame_count")
        # DAgger frames are labelled by the automatic expert, not by whatever
        # the policy actually pressed.
        labels = [step.get("expert_controls") or step["controls"] for step in steps]
        has_duty = bool(steps) and "expert_duty" in steps[0]
        if steer_window:
            signs = np.array([steer_sign(controls) for controls in labels], np.float64)
            self.duty = steer_duty(signs, steer_window)
            if has_duty:
                # Expert-lookup laps carry the experts' steering amount directly.
                self.duty = np.array([np.nan if s["expert_duty"] is None else s["expert_duty"] for s in steps], np.float64)
            self.steer = duty_to_bin(np.nan_to_num(self.duty), steer_bins)
        else:
            self.duty = None
            self.steer = np.array([steer_class(controls) for controls in labels], np.int64)
        self.throttle = np.array([throttle_class(controls) for controls in labels], np.int64)
        self.usable = np.array(
            [not (step["race_time_ms"] > 2_000 and step["telemetry"]["display_speed_kmh"] < 10) for step in steps],
            bool,
        )
        if has_duty and steer_window:
            # No expert state was close (e.g. fallen off): no label for the chunk.
            labelled = ~np.isnan(self.duty)
            self.usable &= np.array([labelled[t:t + 4].all() for t in range(len(steps))])
        hint_track = track.fresh() if line_hint else None
        self.telemetry = [telemetry_text(step["telemetry"], turn_hint=turn_hint, track=hint_track) for step in steps]
        self.aux = None
        if aux:
            from .track_features import outcome_targets

            manifest = json.loads((episode.directory / "manifest.json").read_text(encoding="utf-8"))
            self.aux = outcome_targets(track, steps, manifest.get("status") == "complete")
        if aux_only:
            # Watched laps without expert labels: only the world-model head learns from them.
            self.steer = np.full(len(steps), IGNORE, np.int64)
            self.throttle = np.full(len(steps), IGNORE, np.int64)
        self.frames = np.memmap(
            episode.directory / "frames.rgb",
            dtype=np.uint8,
            mode="r",
            shape=(episode.frame_count, episode.height, episode.width, 3),
        )

    def image(self, index: int) -> Image.Image:
        return Image.fromarray(np.ascontiguousarray(self.frames[index]), "RGB")


@dataclass(frozen=True)
class Sample:
    frames: tuple[Image.Image, ...]  # oldest first, current last
    mode: str
    telemetry: str
    steer: np.ndarray  # [chunk] class ids
    throttle: np.ndarray  # [chunk] class ids
    aux: np.ndarray | None = None  # outcome targets (track_features.outcome_targets), NaN = unknown


class ChunkDataset:
    """Every frame t with t-(history-1)*stride >= 0 and t+chunk <= frame_count.

    ``shift_fraction`` (training only) moves every frame of a sample sideways
    by up to that fraction of the width with probability ``shift_probability``
    and corrects the steering duty by ``shift_gain`` per unit fraction: content
    moved right looks like the car yawed left, so it should steer right. This
    is the PilotNet side-camera trick for off-line views without new laps.
    """

    def __init__(
        self,
        episodes: list[Episode],
        *,
        stride: int = 2,
        chunk: int = 4,
        history: int = 2,
        steer_window: int = 0,
        steer_bins: int = 3,
        turn_hint: bool = False,
        shift_fraction: float = 0.0,
        shift_probability: float = 0.5,
        shift_gain: float = 8.0,
        seed: int = 0,
        line_hint: bool = False,
        aux: bool = False,
        aux_only: bool = False,
        track=None,
    ):
        if shift_fraction and not steer_window:
            raise ValueError("shift augmentation needs steering-amount labels (steer_window)")
        self.stride = stride
        self.chunk = chunk
        self.history = history
        self.steer_bins = steer_bins
        self.shift_fraction = shift_fraction
        self.shift_probability = shift_probability
        self.shift_gain = shift_gain
        self.rng = random.Random(seed)
        if (line_hint or aux) and track is None:
            from .track_features import default_track

            track = default_track()
        self.episodes = [
            _EpisodeData(episode, steer_window=steer_window, steer_bins=steer_bins, turn_hint=turn_hint,
                         track=track, line_hint=line_hint, aux=aux, aux_only=aux_only)
            for episode in episodes
        ]
        first = (history - 1) * stride
        # Skip samples where the car is stuck (stopped after the start).
        self.index = [
            (episode_index, t)
            for episode_index, data in enumerate(self.episodes)
            for t in range(first, data.episode.frame_count - chunk + 1)
            if data.usable[t]
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> Sample:
        episode_index, t = self.index[item]
        data = self.episodes[episode_index]
        frames = tuple(data.image(t - k * self.stride) for k in reversed(range(self.history)))
        steer = data.steer[t : t + self.chunk]
        if self.shift_fraction and self.rng.random() < self.shift_probability:
            dx = round(self.rng.uniform(-self.shift_fraction, self.shift_fraction) * data.episode.width)
            if dx:
                frames = tuple(shift_image(frame, dx) for frame in frames)
                duty = np.clip(data.duty[t : t + self.chunk] + self.shift_gain * dx / data.episode.width, -1.0, 1.0)
                steer = duty_to_bin(duty, self.steer_bins)
        return Sample(
            frames=frames,
            mode=data.episode.mode,
            telemetry=data.telemetry[t],
            steer=steer,
            throttle=data.throttle[t : t + self.chunk],
            aux=None if data.aux is None else data.aux[t],
        )

    def label_counts(self) -> tuple[np.ndarray, np.ndarray]:
        """Class counts over every chunk position of every sample."""
        steer = np.zeros(self.steer_bins, np.int64)
        throttle = np.zeros(3, np.int64)
        for episode_index, t in self.index:
            data = self.episodes[episode_index]
            steer += np.bincount(data.steer[t : t + self.chunk], minlength=self.steer_bins)
            throttle += np.bincount(data.throttle[t : t + self.chunk], minlength=3)
        return steer, throttle

    def stratified_indices(self, per_class: int, *, seed: int = 0) -> list[int]:
        """Validation subset balanced over current steer class, plus every
        sample whose current throttle is rare (coast/brake)."""
        rng = random.Random(seed)
        by_steer: dict[int, list[int]] = {c: [] for c in range(self.steer_bins)}
        rare_throttle: list[int] = []
        for item, (episode_index, t) in enumerate(self.index):
            data = self.episodes[episode_index]
            by_steer[int(data.steer[t])].append(item)
            if data.throttle[t] != 1:
                rare_throttle.append(item)
        chosen: set[int] = set(rare_throttle)
        for items in by_steer.values():
            chosen.update(rng.sample(items, min(per_class, len(items))))
        return sorted(chosen)


def mild_class_weights(counts: np.ndarray, cap: float = 10.0) -> np.ndarray:
    """sqrt(inverse frequency), capped, normalized to expected weight 1.

    Plan v5: raw inverse-frequency weights caused an asymmetric collapse.
    """
    frequency = counts / counts.sum()
    weight = np.minimum(np.sqrt(1.0 / np.maximum(frequency, 1e-6)), cap)
    weight = weight / (weight * frequency).sum()
    return weight.astype(np.float32)
