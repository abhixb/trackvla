"""Phase 4: let the trained Qwen policy drive A01 in the real game.

Each lap uses the same unattended game setup as replay recording (fresh game
-> windowed 16:9 -> profile -> no opponent ghost) and then hands control to
the policy. Settings compare the raw model with the validation-fitted logit
corrections from ``calibration.json``. Results: lap completion, race time,
checkpoints reached, and every frame/action saved under datasets/closed_loop.
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
from trackmania_vla.qwen_policy import load_policy, make_blend_planner, make_planner  # noqa: E402
from trackmania_vla.start_frame_check import opponent_ghost_at_start  # noqa: E402
from trackmania_vla.tmnf_window import OpponentDialogController, close_trackmania  # noqa: E402


def main() -> int:
    local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--settings", default="raw,steer,both",
        help="comma list of raw, steer, both (steer/both need calibration.json)",
    )
    parser.add_argument("--laps", type=int, default=2, help="laps per setting")
    parser.add_argument("--speed", type=float, default=4.0)
    parser.add_argument(
        "--replan-ms", type=int, default=200,
        help="plan interval; v1 at 50 executes only each chunk's first action, v2 averages overlapping chunks",
    )
    parser.add_argument("--pwm-ms", type=int, default=50, help="v2 steering tap resolution in ms (10 swerves; 50 matches expert taps)")
    parser.add_argument("--steer-rule", choices=("expected", "argmax", "rounded"), default="expected",
                        help="v2: how the steering amount is read from the averaged chunk probabilities")
    parser.add_argument("--steer-deadzone", type=float, default=0.0, help="v2: steering amounts below this are 0")
    parser.add_argument("--steer-smooth-ms", type=int, default=0, help="v2: moving-average time constant on the steering amount")
    parser.add_argument("--steer-switch", type=float, default=0.0, help="v2: opposite steering needs at least this amount (hysteresis)")
    parser.add_argument("--prior-correct", action="store_true",
                        help="subtract log(training class weight) from steer logits: undoes the weighted loss's push toward steering")
    parser.add_argument("--max-race-s", type=float, default=60.0)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "datasets" / "closed_loop")
    parser.add_argument("--map-id", default="A01-Race")
    parser.add_argument("--map", dest="map_command", default='"Official Maps\\A01-Race.Challenge.Gbx"')
    parser.add_argument("--loader", type=Path, default=local_app_data / "TMLoader" / "TMLoader.exe")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument("--blend-early", type=Path,
                        help="this checkpoint drives until --blend-at-m, then CHECKPOINT takes over (same backbone)")
    parser.add_argument("--blend-at-m", type=float, default=1400.0, help="racing-line metre of the hand-over")
    parser.add_argument("--none-row-y", type=float, help="Select Opponent 'None' row height (fraction); A03: 0.62")
    parser.add_argument("--no-ghost-check", action="store_true",
                        help="skip the opponent-ghost start check (its reference frame is A01's spawn)")
    args = parser.parse_args()

    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    policy, manifest = load_policy(args.checkpoint)
    config = policy.config
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    prior = -np.log(policy.steer_weight.float().cpu().numpy()) if args.prior_correct else None
    biases = {"raw": (prior, None)}
    if set(settings) - {"raw"}:
        calibration = json.loads((args.checkpoint / "calibration.json").read_text(encoding="utf-8"))
        biases["steer"] = (np.array(calibration["steer_bias"]), None)
        biases["both"] = (np.array(calibration["steer_bias"]), np.array(calibration["throttle_bias"]))
    print(f"policy step {manifest.get('step')}; settings {args.settings}; {args.laps} lap(s) each", flush=True)

    def planner_for(setting: str):
        if args.blend_early is not None:
            return make_blend_planner(policy, args.blend_early, args.blend_at_m)
        return make_planner(policy, *biases[setting])

    ghost_controller = OpponentDialogController(none_row_y=args.none_row_y)
    close_trackmania()
    results = []
    for setting in settings:
        for lap in range(1, args.laps + 1):
            for attempt in range(1, 5):
                recording = RecordingConfig(
                    output_root=args.output_root / (
                        f"{setting}-replan{args.replan_ms}-pwm{args.pwm_ms}-{args.steer_rule}"
                        + (f"-dz{args.steer_deadzone:g}" if args.steer_deadzone else "")
                        + ("-prior" if args.prior_correct else "")
                        + (f"-sm{args.steer_smooth_ms}" if args.steer_smooth_ms else "")
                        + (f"-sw{args.steer_switch:g}" if args.steer_switch else "")
                        + (f"-blend{args.blend_at_m:g}" if args.blend_early is not None else "")
                    ),
                    map_id=args.map_id,
                    map_command=args.map_command,
                    behavior_mode="fast",
                    max_run_seconds=args.max_race_s + 10,
                )
                try:
                    interface, game_pid = connect_ready_game(args)
                    driver = PolicyDriver(
                        interface, recording, planner_for(setting),
                        stride=config.frame_stride, history=config.history,
                        game_speed=args.speed, replan_ms=args.replan_ms, pwm_ms=args.pwm_ms,
                        steer_rule=args.steer_rule, steer_deadzone=args.steer_deadzone,
                        steer_smooth_ms=args.steer_smooth_ms, steer_switch=args.steer_switch,
                        max_race_ms=int(args.max_race_s * 1000),
                        schedule_no_ghost_setup=ghost_controller.schedule_for_process,
                        label=setting,
                    )
                    result = driver.drive(game_pid=game_pid)
                except (OSError, RuntimeError, ValueError) as exc:
                    print(f"  [{setting} lap {lap}] game problem ({type(exc).__name__}: {exc}); restarting", flush=True)
                    close_trackmania()
                    time.sleep(10)
                    continue
                finally:
                    close_trackmania()
                if result.plans and not args.no_ghost_check and opponent_ghost_at_start(Path(result.episode_dir)):
                    quarantine(Path(result.episode_dir), "opponent_ghost_visible")
                    print(f"  [{setting} lap {lap}] opponent ghost visible; retrying", flush=True)
                    continue
                if result.reason in {"interface_disconnected", "interface_shutdown", "run_restarted"}:
                    print(f"  [{setting} lap {lap}] interrupted ({result.reason}); retrying", flush=True)
                    continue
                break
            else:
                print(f"[{setting} lap {lap}] gave up after 4 attempts", flush=True)
                continue
            row = {
                "setting": setting, "replan_ms": args.replan_ms, "lap": lap, "finished": result.finished, "reason": result.reason,
                "race_time_s": None if result.race_time_ms is None else result.race_time_ms / 1000,
                "checkpoints": f"{result.checkpoints}/{result.checkpoint_target}",
                "plans": result.plans, "episode": str(result.episode_dir),
            }
            results.append(row)
            status = f"FINISHED in {row['race_time_s']:.2f}s" if result.finished else f"DNF ({result.reason}) at {row['race_time_s']}s"
            print(f"[{setting} lap {lap}] {status}, checkpoints {row['checkpoints']}", flush=True)

    out = PROJECT_ROOT / "evaluations" / "closed_loop"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out / f"{stamp}.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "policy_step": manifest.get("step"),
        "steer_rule": args.steer_rule, "prior_correct": args.prior_correct,
        "steer_smooth_ms": args.steer_smooth_ms, "steer_switch": args.steer_switch, "steer_deadzone": args.steer_deadzone, "pwm_ms": args.pwm_ms,
        "expert_reference_s": [23.93, 24.51], "results": results,
    }, indent=2), encoding="utf-8")
    print(f"summary: evaluations/closed_loop/{stamp}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
