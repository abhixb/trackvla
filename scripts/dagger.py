"""DAgger data collection with the racing-line follower as the automatic expert.

  follow   the follower drives alone (checks it is a usable teacher)
  collect  the policy drives; the follower labels every frame and takes the
           wheel with probability --beta per planning window, or, with
           --sections, only inside those stretches of the racing line when the
           car is more than --drift-m off it or sliding, until it is back
           within --release-m for --hold-ms

Every lap uses the same unattended game setup as replay recording. Episodes
store the executed controls and the expert's ``expert_controls`` per frame;
training on DAgger rounds uses the expert labels.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from record_replays import connect_ready_game, quarantine  # noqa: E402
from trackmania_vla.closed_loop import PolicyDriver  # noqa: E402
from trackmania_vla.human_recording import RecordingConfig  # noqa: E402
from trackmania_vla.expert_lookup import ExpertLookup  # noqa: E402
from trackmania_vla.line_follower import FollowerConfig, GuidedFollower, LineFollower, RacingLine  # noqa: E402
from trackmania_vla.start_frame_check import opponent_ghost_at_start  # noqa: E402
from trackmania_vla.tmnf_window import OpponentDialogController, close_trackmania  # noqa: E402

REFERENCE_LAP = PROJECT_ROOT / "datasets" / "reference_laps" / "episodes"
FOLLOWER = FollowerConfig(left_sign=-1, lookahead_s=0.6, deadband_deg=3.0)


class SectionTakeover:
    """Expert drives on mistakes, but only inside the given line sections."""

    def __init__(self, line: RacingLine, sections: list[tuple[float, float]], drift_m: float, release_m: float, hold_ms: int):
        self.line = line
        self.sections = sections
        self.drift_m = drift_m
        self.release_m = release_m
        self.hold_ms = hold_ms
        self.hint: int | None = None
        self.since: int | None = None
        self.back_since: int | None = None
        self.triggers = 0

    def __call__(self, telemetry: dict) -> bool:
        position = np.asarray(telemetry["position_m"], float)
        index = self.line.nearest(position, self.hint)
        if np.linalg.norm(self.line.points[index] - position) > 25.0:
            index = self.line.nearest(position)
        self.hint = index
        # Horizontal distance: over jumps the height legitimately differs.
        offset = float(np.hypot(*(self.line.points[index] - position)[[0, 2]]))
        metre = float(self.line.distance[index])
        inside = any(start <= metre < end for start, end in self.sections)
        t = int(telemetry["race_time_ms"])
        mistake = offset > self.drift_m or bool(telemetry.get("car_sliding"))
        if self.since is None:
            if inside and mistake:
                self.since, self.back_since = t, None
                self.triggers += 1
            return self.since is not None
        # Hand back once on the line for hold_ms (or when the section ends).
        if not inside:
            self.since = None
        elif offset < self.release_m and not telemetry.get("car_sliding"):
            self.back_since = t if self.back_since is None else self.back_since
            if t - self.back_since >= self.hold_ms:
                self.since = None
        else:
            self.back_since = None
        return self.since is not None


def load_policy_planner(checkpoint: Path):
    from trackmania_vla.qwen_policy import load_policy, make_planner

    policy, _ = load_policy(checkpoint)
    return make_planner(policy), policy.config.history


def main() -> int:
    local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("follow", "collect"))
    parser.add_argument("--checkpoint", type=Path, help="policy checkpoint (collect mode)")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--laps", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.5, help="probability the expert drives each window")
    parser.add_argument("--expert", choices=("line", "guided", "lookup"), default="line",
                        help="line: pure-pursuit follower; guided: expert laps' steering profile + correction toward the line; "
                             "lookup: what the experts did in the most similar recorded states")
    parser.add_argument("--expert-root", type=Path, default=PROJECT_ROOT / "datasets" / "replay_runs" / "episodes",
                        help="lookup expert: position-tagged expert replay episodes")
    parser.add_argument("--gain", type=float, default=0.08, help="guided expert: steering per degree of heading error")
    parser.add_argument("--gain-deadband", type=float, default=0.0, help="guided expert: ignore heading errors below this (degrees)")
    parser.add_argument("--sections", help="takeover only in these line sections, e.g. 150-550,1400-1700 (metres)")
    parser.add_argument("--drift-m", type=float, default=3.0)
    parser.add_argument("--release-m", type=float, default=1.5)
    parser.add_argument("--hold-ms", type=int, default=500)
    parser.add_argument("--replan-ms", type=int, default=200)
    parser.add_argument("--pwm-ms", type=int, default=50)
    parser.add_argument("--steer-rule", choices=("expected", "argmax", "rounded"), default="expected")
    parser.add_argument("--steer-deadzone", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=4.0)
    parser.add_argument("--max-race-s", type=float, default=60.0)
    parser.add_argument("--map-id", default="A01-Race")
    parser.add_argument("--map", dest="map_command", default='"Official Maps\\A01-Race.Challenge.Gbx"')
    parser.add_argument("--loader", type=Path, default=local_app_data / "TMLoader" / "TMLoader.exe")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--no-launch", action="store_true")
    args = parser.parse_args()

    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    line = RacingLine.from_episode(sorted(REFERENCE_LAP.iterdir())[0])
    if args.mode == "follow":
        planner = lambda *frames_and_telemetry: (np.ones(4, int), np.ones(4, int))  # never used
        history = 2
        beta, output = 1.0, PROJECT_ROOT / "datasets" / "dagger" / "follower_test"
    else:
        if args.checkpoint is None:
            parser.error("collect mode needs --checkpoint")
        planner, history = load_policy_planner(args.checkpoint)
        beta, output = args.beta, PROJECT_ROOT / "datasets" / "dagger" / f"round_{args.round}"
    print(f"{args.mode}: {args.laps} lap(s), expert share {beta:.2f}, racing line {line.distance[-1]:.0f} m -> {output}", flush=True)

    ghost_controller = OpponentDialogController()
    close_trackmania()
    rows = []
    lookup = None
    for lap in range(1, args.laps + 1):
        for attempt in range(1, 5):
            if args.expert == "lookup":
                if lookup is None:
                    lookup = ExpertLookup(line, sorted(p for p in args.expert_root.iterdir() if (p / "steps.jsonl").exists()))
                    print(f"expert lookup: {len(lookup)} expert frames", flush=True)
                follower = lookup
            elif args.expert == "guided":
                follower = GuidedFollower(line, sorted(REFERENCE_LAP.iterdir()), FOLLOWER,
                                          gain_per_deg=args.gain, dead_deg=args.gain_deadband)
            else:
                follower = LineFollower(line, FOLLOWER)
            takeover = None
            if args.mode == "collect" and args.sections:
                sections = [tuple(float(v) for v in part.split("-")) for part in args.sections.split(",")]
                takeover = SectionTakeover(line, sections, args.drift_m, args.release_m, args.hold_ms)
            recording = RecordingConfig(
                output_root=output, map_id=args.map_id, map_command=args.map_command,
                behavior_mode="fast", max_run_seconds=args.max_race_s + 10,
            )
            try:
                interface, game_pid = connect_ready_game(args)
                driver = PolicyDriver(
                    interface, recording, planner,
                    stride=2, history=history, game_speed=args.speed,
                    replan_ms=args.replan_ms, pwm_ms=args.pwm_ms,
                    steer_rule=args.steer_rule, steer_deadzone=args.steer_deadzone, takeover=takeover,
                    max_race_ms=int(args.max_race_s * 1000),
                    schedule_no_ghost_setup=ghost_controller.schedule_for_process,
                    label=f"{args.mode}-round{args.round}", labeler=follower.action,
                    expert_probability=beta, seed=1000 * args.round + 10 * lap + attempt,
                )
                result = driver.drive(game_pid=game_pid)
            except (OSError, RuntimeError, ValueError) as exc:
                print(f"  [lap {lap}] game problem ({type(exc).__name__}: {exc}); restarting", flush=True)
                close_trackmania()
                time.sleep(10)
                continue
            finally:
                close_trackmania()
            frames = sum(1 for _ in (Path(result.episode_dir) / "steps.jsonl").open())
            if frames and opponent_ghost_at_start(Path(result.episode_dir)):
                quarantine(Path(result.episode_dir), "opponent_ghost_visible")
                print(f"  [lap {lap}] opponent ghost visible; retrying", flush=True)
                continue
            if result.reason in {"interface_disconnected", "interface_shutdown", "run_restarted"}:
                print(f"  [lap {lap}] interrupted ({result.reason}); retrying", flush=True)
                continue
            break
        else:
            print(f"[lap {lap}] gave up after 4 attempts", flush=True)
            continue
        time_s = None if result.race_time_ms is None else result.race_time_ms / 1000
        rows.append({"lap": lap, "finished": result.finished, "reason": result.reason, "race_time_s": time_s,
                     "checkpoints": result.checkpoints, "frames": frames, "episode": str(result.episode_dir),
                     "takeovers": takeover.triggers if takeover is not None else None})
        status = f"FINISHED in {time_s:.2f}s" if result.finished else f"DNF ({result.reason}) at {time_s}s"
        print(f"[lap {lap}] {status}, checkpoints {result.checkpoints}/{result.checkpoint_target}, {frames} frames", flush=True)

    out = PROJECT_ROOT / "evaluations" / "dagger"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out / f"{args.mode}-round{args.round}-{stamp}.json").write_text(
        json.dumps({"mode": args.mode, "round": args.round, "beta": beta, "steer_rule": args.steer_rule,
                    "sections": args.sections, "drift_m": args.drift_m, "expert": args.expert, "gain": args.gain, "gain_deadband": args.gain_deadband,
                    "follower": vars(FOLLOWER), "laps": rows}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
