"""Post-training evaluation and bias calibration for the Qwen driving policy.

1. Score every frame of the validation and test laps (frame t -> action t,
   i.e. the first chunk position, which is what the controller executes).
2. Fit an additive per-class logit bias on the VALIDATION laps that maximizes
   balanced (mean per-class) recall, for steer and throttle separately. This
   counters the majority-class pull without retraining.
3. Report strict and +/-1 frame (50 ms) tolerant per-class recall on the
   untouched TEST laps, before and after calibration.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trackmania_vla.qwen_policy import QwenDrivingPolicy, QwenPolicyConfig
from trackmania_vla.replay_dataset import STEER_CLASSES, THROTTLE_CLASSES, ChunkDataset, load_episodes, split_episodes


@torch.no_grad()
def lap_logits(policy, dataset: ChunkDataset, batch_size: int):
    """Per lap: first-position logits and labels for every scored frame."""
    policy.eval()
    laps = []
    for episode_index in range(len(dataset.episodes)):
        items = [i for i, (e, _) in enumerate(dataset.index) if e == episode_index]
        steer_logits, throttle_logits = [], []
        for start in range(0, len(items), batch_size):
            samples = [dataset[i] for i in items[start : start + batch_size]]
            inputs = policy.encode([s.frames for s in samples], [policy.prompt(s.mode, s.telemetry) for s in samples])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s, t = policy.logits(inputs)
            steer_logits.append(s[:, 0].float().cpu().numpy())
            throttle_logits.append(t[:, 0].float().cpu().numpy())
        data = dataset.episodes[episode_index]
        frames = [dataset.index[i][1] for i in items]
        laps.append({
            "steer_logits": np.concatenate(steer_logits),
            "throttle_logits": np.concatenate(throttle_logits),
            "steer": data.steer[frames],
            "throttle": data.throttle[frames],
        })
        print(f"  scored lap {episode_index + 1}/{len(dataset.episodes)}", flush=True)
    return laps


def recall(true, pred, names, tolerance=0):
    """Per-class recall; with tolerance k a frame counts if the class is
    predicted anywhere within +/-k frames (same lap)."""
    out = {}
    for c, name in enumerate(names):
        hits = total = 0
        for t_lap, p_lap in zip(true, pred):
            idx = np.flatnonzero(t_lap == c)
            total += len(idx)
            for i in idx:
                window = p_lap[max(0, i - tolerance) : i + tolerance + 1]
                hits += bool((window == c).any())
        out[name] = hits / total if total else None
    present = [v for v in out.values() if v is not None]
    out["balanced"] = float(np.mean(present))
    return out


def fit_bias(logits, labels, names):
    """Grid-search an additive bias (majority class fixed at 0) maximizing
    balanced recall on validation."""
    majority = int(np.bincount(np.concatenate(labels), minlength=3).argmax())
    others = [c for c in range(3) if c != majority]
    grid = np.arange(-2.0, 6.01, 0.25)
    best = (-1.0, np.zeros(3))
    for a, b in itertools.product(grid, grid):
        bias = np.zeros(3)
        bias[others[0]], bias[others[1]] = a, b
        preds = [(l + bias).argmax(-1) for l in logits]
        score = recall(labels, preds, names)["balanced"]
        if score > best[0]:
            best = (score, bias)
    return best[1]


def report(laps, key, names, bias):
    logits = [lap[f"{key}_logits"] for lap in laps]
    labels = [lap[key] for lap in laps]
    raw = [l.argmax(-1) for l in logits]
    cal = [(l + bias).argmax(-1) for l in logits]
    return {
        "raw": recall(labels, raw, names),
        "raw_tolerant_50ms": recall(labels, raw, names, tolerance=1),
        "calibrated": recall(labels, cal, names),
        "calibrated_tolerant_50ms": recall(labels, cal, names, tolerance=1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="checkpoint directory, e.g. checkpoints/qwen3b/<run>/best")
    parser.add_argument("--episodes", type=Path, default=PROJECT_ROOT / "datasets" / "replay_runs" / "episodes")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = QwenPolicyConfig()
    splits = split_episodes(load_episodes(args.episodes), seed=args.seed)
    policy = QwenDrivingPolicy.from_pretrained(config)
    manifest = policy.load(args.checkpoint)
    print(f"loaded {args.checkpoint} (step {manifest.get('step')})", flush=True)

    print("scoring validation laps", flush=True)
    val = lap_logits(policy, ChunkDataset(splits["validation"], stride=config.frame_stride, chunk=config.chunk), args.batch_size)
    print("scoring test laps", flush=True)
    test = lap_logits(policy, ChunkDataset(splits["test"], stride=config.frame_stride, chunk=config.chunk), args.batch_size)

    steer_bias = fit_bias([l["steer_logits"] for l in val], [l["steer"] for l in val], STEER_CLASSES)
    throttle_bias = fit_bias([l["throttle_logits"] for l in val], [l["throttle"] for l in val], THROTTLE_CLASSES)
    result = {
        "checkpoint": str(args.checkpoint),
        "step": manifest.get("step"),
        "steer_bias": steer_bias.tolist(),
        "throttle_bias": throttle_bias.tolist(),
        "validation": {"steer": report(val, "steer", STEER_CLASSES, steer_bias),
                       "throttle": report(val, "throttle", THROTTLE_CLASSES, throttle_bias)},
        "test": {"steer": report(test, "steer", STEER_CLASSES, steer_bias),
                 "throttle": report(test, "throttle", THROTTLE_CLASSES, throttle_bias)},
    }
    (args.checkpoint / "calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for split in ("validation", "test"):
        for key in ("steer", "throttle"):
            for variant, scores in result[split][key].items():
                shown = {k: (round(v, 3) if v is not None else None) for k, v in scores.items()}
                print(f"{split:10s} {key:8s} {variant:26s} {shown}")
    print(f"steer bias {np.round(steer_bias, 2).tolist()}  throttle bias {np.round(throttle_bias, 2).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
