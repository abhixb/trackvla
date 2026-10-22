"""Train the Qwen2.5-VL-3B driving policy on replay-recorded A01 laps.

  --phase0                one training step + inference latency, then exit
  --overfit-episodes 1    Phase 2 gate: train and evaluate on one lap
  (default)               Phase 3 full SFT with stratified per-class validation
  --v2                    steering amount (5 levels), 4 frames, turn hint,
                          sideways-shift augmentation; best = lowest steer error
  --v3                    v2 + racing-line geometry in the prompt + world-model
                          head trained on every recorded lap (--aux-root)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataclasses import replace

from trackmania_vla.qwen_policy import V2_CONFIG, V3_CONFIG, QwenDrivingPolicy, QwenPolicyConfig
from trackmania_vla.replay_dataset import (
    STEER_CLASSES,
    steer_bin_values,
    THROTTLE_CLASSES,
    ChunkDataset,
    has_positions,
    load_dagger_episodes,
    load_episodes,
    mild_class_weights,
    split_episodes,
)


def batch_inputs(policy, dataset, indices):
    samples = [dataset[i] for i in indices]
    inputs = policy.encode(
        [s.frames for s in samples], [policy.prompt(s.mode, s.telemetry) for s in samples]
    )
    steer = torch.as_tensor(np.stack([s.steer for s in samples]))
    throttle = torch.as_tensor(np.stack([s.throttle for s in samples]))
    return inputs, steer, throttle


def batch_inputs_mixed(policy, dataset, indices, extra=None, extra_indices=(), watched=None, watched_indices=()):
    samples = ([dataset[i] for i in indices] + [extra[i] for i in extra_indices]
               + [watched[i] for i in watched_indices])
    inputs = policy.encode(
        [s.frames for s in samples], [policy.prompt(s.mode, s.telemetry) for s in samples]
    )
    steer = torch.as_tensor(np.stack([s.steer for s in samples]))
    throttle = torch.as_tensor(np.stack([s.throttle for s in samples]))
    aux = torch.as_tensor(np.stack([s.aux for s in samples])) if samples[0].aux is not None else None
    return inputs, steer, throttle, aux


@torch.no_grad()
def evaluate_aux(policy, dataset, indices, batch_size) -> dict:
    """World-model head on held-out watched laps: crash flag and 2 s offset."""
    policy.eval()
    predictions, targets = [], []
    for start in range(0, len(indices), batch_size):
        inputs, _, _, aux = batch_inputs_mixed(policy, dataset, indices[start : start + batch_size])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens, mask = policy.tokens(inputs)
            predictions.append(policy.aux_head(tokens, mask).float().cpu().numpy())
        targets.append(aux.numpy())
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    crash_true, crash_pred = target[:, -1] > 0.5, prediction[:, -1] > 0
    known = ~np.isnan(target[:, 1])
    return {
        "samples": len(indices),
        "crash_share": float(crash_true.mean()),
        "crash_recall": float((crash_pred & crash_true).sum() / max(1, crash_true.sum())),
        "crash_precision": float((crash_pred & crash_true).sum() / max(1, crash_pred.sum())),
        "offset_2s_mae_m": float(np.abs(prediction[known, 1] - target[known, 1]).mean() * 10) if known.any() else None,
        "offset_2s_baseline_mae_m": float(np.abs(target[known, 1]).mean() * 10) if known.any() else None,
    }


def per_class_recall(true: np.ndarray, pred: np.ndarray, names) -> dict[str, float | None]:
    return {
        name: (float((pred[true == c] == c).mean()) if (true == c).any() else None)
        for c, name in enumerate(names)
    }


def steer_names(bins: int) -> tuple[str, ...]:
    if bins == 3:
        return STEER_CLASSES
    return tuple(f"{v:+.1f}" for v in steer_bin_values(bins))


@torch.no_grad()
def evaluate(policy, dataset, indices, batch_size) -> dict:
    policy.eval()
    bins = policy.config.steer_bins
    names = steer_names(bins)
    values = steer_bin_values(bins)
    steer_true, steer_pred, thr_true, thr_pred, expected = [], [], [], [], []
    for start in range(0, len(indices), batch_size):
        inputs, steer, throttle, _ = batch_inputs_mixed(policy, dataset, indices[start : start + batch_size])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s_logits, t_logits = policy.logits(inputs)
        s_logits = s_logits.float()
        steer_true.append(steer.numpy()); steer_pred.append(s_logits.argmax(-1).cpu().numpy())
        thr_true.append(throttle.numpy()); thr_pred.append(t_logits.argmax(-1).cpu().numpy())
        expected.append((s_logits.softmax(-1).cpu().numpy() @ values))
    st, sp = np.concatenate(steer_true), np.concatenate(steer_pred)
    tt, tp = np.concatenate(thr_true), np.concatenate(thr_pred)
    steer_recall = per_class_recall(st.ravel(), sp.ravel(), names)
    throttle_recall = per_class_recall(tt.ravel(), tp.ravel(), THROTTLE_CLASSES)
    # Steering-amount error: expected amount vs the label's level (v1: -1/0/+1).
    error = np.abs(np.concatenate(expected) - values[st])
    confusion = np.zeros((bins, bins), int)
    np.add.at(confusion, (st.ravel(), sp.ravel()), 1)
    present = [v for v in steer_recall.values() if v is not None]
    return {
        "samples": len(indices),
        "steer_recall": steer_recall,
        "steer_recall_now": per_class_recall(st[:, 0], sp[:, 0], names),
        "steer_mean_abs_error": float(error.mean()),
        "steer_within_one_level": float((np.abs(sp - st) <= 1).mean()),
        "steer_balanced_accuracy": float(np.mean(present)),
        "steer_min_class_recall": float(np.min(present)),
        "throttle_recall": throttle_recall,
        "exact_chunk_accuracy": float(((sp == st) & (tp == tt)).all(axis=1).mean()),
        "steer_confusion_true_by_pred": confusion.tolist(),
        "predicted_steer_share": (np.bincount(sp.ravel(), minlength=bins) / sp.size).round(3).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=Path, default=PROJECT_ROOT / "datasets" / "replay_runs" / "episodes")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "checkpoints" / "qwen3b")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=15_000)
    parser.add_argument("--lr", type=float, default=1e-4, help="LoRA learning rate")
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=2500)
    parser.add_argument("--val-per-class", type=int, default=250)
    parser.add_argument("--overfit-episodes", type=int, default=0)
    parser.add_argument("--phase0", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-checkpoint", type=Path, help="start from these LoRA/head weights")
    parser.add_argument("--dagger-root", type=Path, nargs="+", help="datasets/dagger: add closed-loop laps with expert labels")
    parser.add_argument("--dagger-fraction", type=float, default=0.3, help="share of each batch drawn from DAgger frames")
    parser.add_argument("--safe-root", type=Path, help="datasets/safe: add the line follower's finished safe-mode laps to the DAgger pool")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--v2", action="store_true", help="steering amount, 4 frames, turn hint, shift augmentation")
    parser.add_argument("--v3", action="store_true", help="v2 + racing-line hint + world-model head")
    parser.add_argument("--aux-root", type=Path, nargs="+",
                        help="v3: recorded policy laps (evaluations) the world-model head watches; no driving labels")
    parser.add_argument("--aux-exclude", nargs="*", default=["respawn"], help="skip watched laps whose path contains these")
    parser.add_argument("--aux-fraction", type=float, default=0.125, help="share of each batch drawn from watched laps")
    parser.add_argument("--aux-weight", type=float, default=0.5, help="world-model loss weight")
    parser.add_argument("--shift-fraction", type=float, default=0.05, help="v2: max sideways shift (fraction of width)")
    parser.add_argument("--shift-gain", type=float, default=8.0, help="v2: steering correction per unit shift fraction")
    args = parser.parse_args()

    if sys.platform == "win32":
        import ctypes

        # Keep the PC from idle-sleeping during long runs; released on exit.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    config = V3_CONFIG if args.v3 else (V2_CONFIG if args.v2 else QwenPolicyConfig())
    data_kwargs = config.dataset_kwargs()
    augment = ({"shift_fraction": args.shift_fraction, "shift_gain": args.shift_gain, "seed": args.seed}
               if args.v2 or args.v3 else {})
    splits = split_episodes(load_episodes(args.episodes), seed=args.seed)
    if args.overfit_episodes:
        splits["train"] = splits["train"][: args.overfit_episodes]
    train = ChunkDataset(splits["train"], **data_kwargs, **augment)
    if args.overfit_episodes:
        val, val_indices = ChunkDataset(splits["train"], **data_kwargs), list(range(len(train)))
    else:
        val = ChunkDataset(splits["validation"], **data_kwargs)
        val_indices = val.stratified_indices(args.val_per_class, seed=args.seed)
    dagger = None
    extra_episodes = []
    for root in args.dagger_root or []:
        extra_episodes += load_dagger_episodes(root)
    if config.line_hint or config.aux:
        extra_episodes = [e for e in extra_episodes if has_positions(e)]
    if args.safe_root is not None:
        safe_episodes = load_dagger_episodes(args.safe_root, finished_only=True)
        print(f"safe laps: {len(safe_episodes)}", flush=True)
        extra_episodes += safe_episodes
    if extra_episodes:
        dagger = ChunkDataset(extra_episodes, **data_kwargs, **augment)
        print(f"dagger {len(dagger.episodes)} laps / {len(dagger)} samples (batch share {args.dagger_fraction:.0%})", flush=True)
    watched = watched_val = None
    watched_val_indices: list[int] = []
    if args.aux_root and config.aux:
        laps = []
        for root in args.aux_root:
            laps += [e for e in load_dagger_episodes(root, pattern="**/manifest.json", map_id="A01-Race")
                     if not any(word in str(e.directory) for word in args.aux_exclude) and has_positions(e)]
        laps = split_episodes(laps, validation=max(1, len(laps) // 10), test=0, seed=args.seed)
        watched = ChunkDataset(laps["train"], **data_kwargs, aux_only=True)
        watched_val = ChunkDataset(laps["validation"], **data_kwargs, aux_only=True)
        watched_val_indices = random.Random(args.seed).sample(range(len(watched_val)), min(600, len(watched_val)))
        print(f"watched laps {len(watched.episodes)} / {len(watched)} samples (batch share {args.aux_fraction:.0%}); "
              f"held out {len(watched_val.episodes)} laps", flush=True)
    steer_counts, throttle_counts = train.label_counts()
    if dagger is not None:
        dagger_steer, dagger_throttle = dagger.label_counts()
        # Weights follow the mix actually seen in batches.
        share = args.dagger_fraction
        steer_counts = (1 - share) * steer_counts / steer_counts.sum() + share * dagger_steer / max(1, dagger_steer.sum())
        throttle_counts = (1 - share) * throttle_counts / throttle_counts.sum() + share * dagger_throttle / max(1, dagger_throttle.sum())
    print(f"train {len(splits['train'])} laps / {len(train)} samples; eval {len(val_indices)} samples", flush=True)

    policy = QwenDrivingPolicy.from_pretrained(config)
    policy.set_class_weights(mild_class_weights(steer_counts), mild_class_weights(throttle_counts))
    if args.init_checkpoint is not None:
        weights = (policy.steer_weight.clone(), policy.throttle_weight.clone())
        source = QwenPolicyConfig.from_checkpoint(args.init_checkpoint)
        # A v2 checkpoint fits a v3 policy whole, except the new world-model head.
        partial = source != config and replace(source, line_hint=config.line_hint, aux=config.aux) == config
        backbone_only = source != config and not partial
        loaded = policy.load(args.init_checkpoint, backbone_only=backbone_only, partial=partial)
        print(
            f"initialized from {args.init_checkpoint} (step {loaded.get('step')})"
            + (" - LoRA only, fresh head" if backbone_only else "")
            + (" - fresh world-model head" if partial else ""),
            flush=True,
        )
        policy.set_class_weights(*weights)  # keep weights for this data mix
    head_params = list(policy.head.parameters())
    head_ids = {id(p) for p in head_params}
    lora_params = [p for p in policy.parameters() if p.requires_grad and id(p) not in head_ids]
    print(f"trainable: LoRA {sum(p.numel() for p in lora_params)/1e6:.1f}M, head {sum(p.numel() for p in head_params)/1e6:.1f}M", flush=True)
    optimizer = torch.optim.AdamW(
        [{"params": lora_params, "lr": args.lr}, {"params": head_params, "lr": args.head_lr}], weight_decay=0.01
    )
    base_lrs = [args.lr, args.head_lr]

    def train_step(indices, dagger_indices=(), watched_indices=()) -> tuple[float, dict]:
        policy.train()
        inputs, steer, throttle, aux = batch_inputs_mixed(policy, train, indices, dagger, dagger_indices, watched, watched_indices)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, parts = policy.loss(inputs, steer, throttle, aux, args.aux_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], 1.0)
        optimizer.step()
        return float(loss), parts

    if args.phase0:
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        loss, _ = train_step(list(range(args.batch_size)))
        torch.cuda.synchronize(); step_s = time.perf_counter() - start
        grads = sum(1 for p in lora_params if p.grad is not None and p.grad.abs().sum() > 0)
        policy.eval()
        inputs, _, _, _ = batch_inputs_mixed(policy, train, [0])
        latencies = []
        for _ in range(6):
            torch.cuda.synchronize(); t = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                policy.predict(inputs)
            torch.cuda.synchronize(); latencies.append(time.perf_counter() - t)
        tokens = int(inputs["input_ids"].shape[1])
        print(json.dumps({
            "phase0": "ok", "batch_size": args.batch_size, "loss": round(loss, 4), "train_step_s": round(step_s, 2),
            "lora_tensors_with_grad": grads, "lora_tensors": len(lora_params),
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "tokens_per_sample": tokens, "plan_latency_ms_median": round(1000 * sorted(latencies[1:])[2], 1),
        }, indent=2))
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = "overfit-" if args.overfit_episodes else (f"{args.run_name}-" if args.run_name else "")
    run_dir = args.output_root / f"{prefix}{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "run.json").write_text(json.dumps({
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_laps": [e.replay_id for e in splits["train"]],
        "validation_laps": [e.replay_id for e in splits["validation"]],
        "test_laps": [e.replay_id for e in splits["test"]],
        "dagger_laps": [e.episode.replay_id for e in dagger.episodes] if dagger is not None else [],
        "steer_counts": steer_counts.tolist(), "throttle_counts": throttle_counts.tolist(),
        "steer_weight": policy.steer_weight.tolist(), "throttle_weight": policy.throttle_weight.tolist(),
    }, indent=2, default=str), encoding="utf-8")
    metrics_path = run_dir / "metrics.jsonl"

    def log(record: dict) -> None:
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

    order: list[int] = []
    dagger_order: list[int] = []
    dagger_per_batch = round(args.batch_size * args.dagger_fraction) if dagger is not None and len(dagger) else 0
    watched_order: list[int] = []
    watched_per_batch = round(args.batch_size * args.aux_fraction) if watched is not None and len(watched) else 0
    # v1 keeps the best balanced steer accuracy; v2 the lowest steering-amount error.
    best = -1.0 if config.steer_bins == 3 else -float("inf")
    started = time.perf_counter()
    running = []
    for step in range(1, args.max_steps + 1):
        # Natural sampling without replacement; reshuffle each epoch.
        expert_per_batch = args.batch_size - dagger_per_batch - watched_per_batch
        if len(order) < expert_per_batch:
            order = random.sample(range(len(train)), len(train))
        batch, order = order[:expert_per_batch], order[expert_per_batch:]
        dagger_batch: list[int] = []
        if dagger_per_batch:
            if len(dagger_order) < dagger_per_batch:
                dagger_order = random.sample(range(len(dagger)), len(dagger))
            dagger_batch, dagger_order = dagger_order[:dagger_per_batch], dagger_order[dagger_per_batch:]
        watched_batch: list[int] = []
        if watched_per_batch:
            if len(watched_order) < watched_per_batch:
                watched_order = random.sample(range(len(watched)), len(watched))
            watched_batch, watched_order = watched_order[:watched_per_batch], watched_order[watched_per_batch:]
        warm = min(1.0, step / args.warmup_steps)
        decay = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / args.max_steps)))
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base * warm * decay
        loss, parts = train_step(batch, dagger_batch, watched_batch)
        if not math.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        running.append(loss)
        if step % 50 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"step {step}/{args.max_steps} loss {np.mean(running):.4f} "
                f"({elapsed / step:.2f}s/step, eta {(args.max_steps - step) * elapsed / step / 3600:.1f}h, "
                f"vram {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB)",
                flush=True,
            )
            log({"step": step, "train_loss": float(np.mean(running)), **parts})
            running = []
        if step % args.eval_every == 0 or step == args.max_steps:
            result = evaluate(policy, val, val_indices, args.batch_size)
            if watched_val is not None:
                result["world_model"] = evaluate_aux(policy, watched_val, watched_val_indices, args.batch_size)
                print(f"EVAL step {step} world model: {result['world_model']}", flush=True)
            log({"step": step, "eval": result})
            print(
                f"EVAL step {step}: steer error {result['steer_mean_abs_error']:.3f} "
                f"within-one-level {result['steer_within_one_level']:.3f} "
                f"balanced steer {result['steer_balanced_accuracy']:.3f} "
                f"(min class {result['steer_min_class_recall']:.3f}) recall {result['steer_recall']} "
                f"throttle {result['throttle_recall']} predicted share {result['predicted_steer_share']}",
                flush=True,
            )
            score = result["steer_balanced_accuracy"] if config.steer_bins == 3 else -result["steer_mean_abs_error"]
            if score > best:
                best = score
                best_dir = run_dir / "best"
                if best_dir.exists():
                    for f in best_dir.iterdir():
                        f.unlink()
                    best_dir.rmdir()
                policy.save(best_dir, {"step": step, "eval": result})
        if step % args.save_every == 0:
            policy.save(run_dir / f"step-{step:06d}", {"step": step})
    measure = "balanced steer accuracy" if config.steer_bins == 3 else "steer error"
    print(f"Training complete: {run_dir} (best {measure} {abs(best):.3f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
