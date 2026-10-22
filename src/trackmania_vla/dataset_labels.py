"""Auditable behavior-mode labeling for recorded human episodes."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

from .config import behavior_prompt


@dataclass(frozen=True)
class RetagSummary:
    total: int
    fast: int
    safe: int
    changed: int


def mode_from_finish_time(finish_time_ms: int, fast_threshold_ms: int = 26_000) -> str:
    """Return ``fast`` only for laps strictly below the threshold."""
    if finish_time_ms < 0:
        raise ValueError("finish_time_ms cannot be negative")
    if fast_threshold_ms <= 0:
        raise ValueError("fast_threshold_ms must be positive")
    return "fast" if finish_time_ms < fast_threshold_ms else "safe"


def retag_episode_manifests(
    episodes_root: Path,
    *,
    fast_threshold_ms: int = 26_000,
) -> RetagSummary:
    """Retag every completed episode, writing each validated manifest atomically."""
    if fast_threshold_ms <= 0:
        raise ValueError("fast_threshold_ms must be positive")

    manifest_paths = sorted(episodes_root.glob("*/manifest.json"))
    updates: list[tuple[Path, dict[str, object], str, bool]] = []
    fast = 0
    safe = 0

    # Validate the complete collection before changing any file.
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"non-complete run found in episodes: {manifest_path}")
        try:
            finish_time_ms = int(manifest["finish_detection"]["race_time_ms_at_last_sample"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"missing valid finish time in {manifest_path}") from exc

        existing_labeling = manifest.get("behavior_labeling") or {}
        if existing_labeling.get("method") == "manual_override":
            # Human-reviewed labels (e.g. a fast lap ruined by a late crash)
            # are authoritative; the finish-time rule must not revert them.
            mode = str(manifest["behavior_mode"])
            prompt = behavior_prompt(mode)
            labeling = existing_labeling
        else:
            mode = mode_from_finish_time(finish_time_ms, fast_threshold_ms)
            prompt = behavior_prompt(mode)
            labeling = {
                "method": "finish_time_threshold",
                "fast_if_below_ms": fast_threshold_ms,
            }
        changed = (
            manifest.get("behavior_mode") != mode
            or manifest.get("language_instruction") != prompt
            or manifest.get("behavior_labeling") != labeling
        )
        manifest["behavior_mode"] = mode
        manifest["language_instruction"] = prompt
        manifest["behavior_labeling"] = labeling
        updates.append((manifest_path, manifest, mode, changed))
        fast += int(mode == "fast")
        safe += int(mode == "safe")

    for manifest_path, manifest, _mode, _changed in updates:
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, manifest_path)

    return RetagSummary(
        total=len(updates),
        fast=fast,
        safe=safe,
        changed=sum(int(item[3]) for item in updates),
    )
