"""Unattended dataset recording: replay downloaded keyboard laps in TMNF.

Each replay is driven through TMInterface while frames and telemetry are
captured, using the same episode format as human recordings. A lap is kept
only if its checkpoint times reproduce the original replay. The first run
calibrates the replay-time -> race-time input offset and saves it.
"""

from __future__ import annotations

import argparse
import json
import select
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trackmania_vla.human_recording import RecordingConfig, launch_and_connect
from trackmania_vla.replay_inputs import load_keyboard_replay
from trackmania_vla.replay_recording import ReplayRunRecorder
from trackmania_vla.start_frame_check import opponent_ghost_at_start
from trackmania_vla.tmnf_window import (
    OpponentDialogController,
    click_first_profile,
    close_trackmania,
    ensure_windowed_trackmania,
    resize_trackmania_client,
)

CALIBRATION_CANDIDATES_MS = (-10, 0, -20, 10)
MAX_ATTEMPTS_PER_LAP = 4
SILENCE_TIMEOUT_S = 90.0
CONNECT_ANNOUNCE_WAIT_S = 10.0
PROFILE_CLICK_ATTEMPTS = 4
MAX_CONSECUTIVE_GAME_FAILURES = 5
RECOVERY_WAIT_S = 10.0
# Failures caused by the game/session rather than the replay itself.
TRANSIENT_REASONS = frozenset({
    "interface_disconnected",
    "interface_shutdown",
    "opponent_dialog_setup_not_completed",
    "run_restarted",
    "race_time_went_backwards",
})


def finished_replay_ids(output_root: Path, offset: int | None) -> set[str]:
    """Replays to skip: already accepted, or proven non-reproducible at the
    calibrated offset. Other failures (connection, dialog) are retried."""
    done: set[str] = set()
    for folder in ("episodes", "failed"):
        for manifest_path in (output_root / folder).glob("*/manifest.json"):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source = manifest.get("source") or {}
            unreproducible = (
                offset is not None
                and source.get("input_offset_ms") == offset
                and manifest.get("reason") in {"replay_not_reproduced", "checkpoint_count_mismatch"}
            )
            if folder == "episodes" or unreproducible:
                done.add(str(source.get("replay_id")))
    return done


def quarantine(episode_dir: Path, reason: str) -> None:
    manifest_path = episode_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(status="failed", reason=reason)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(episode_dir, episode_dir.parent.parent / "failed" / episode_dir.name)


def _has_pending_data(interface, timeout_s: float) -> bool:
    """True once TMInterface has sent something; nothing is consumed."""
    assert interface.sock is not None
    readable, _, _ = select.select([interface.sock], [], [], timeout_s)
    return bool(readable)


def connect_ready_game(args):
    """Launch or reuse TMNF and return a connection TMInterface has accepted.

    Startup needs no human: the game (which always opens fullscreen) is
    switched to windowed with Alt+Enter, so it keeps rendering in the
    background, and the offline profile is picked on "Select your Profile".
    TMInterface only announces a connection after that screen.
    """
    interface, game_pid = launch_and_connect(
        args.loader,
        args.profile,
        args.port,
        launch=not args.no_launch,
        require_windowed=False,
    )
    try:
        if game_pid is None:
            raise RuntimeError("the TrackMania process id is unknown")
        ensure_windowed_trackmania(game_pid)
        # 16:9 like the fullscreen recordings; the ghost-dialog clicks and the
        # profile click are calibrated for this aspect ratio.
        resize_trackmania_client(game_pid)
        for _ in range(PROFILE_CLICK_ATTEMPTS):
            if _has_pending_data(interface, CONNECT_ANNOUNCE_WAIT_S):
                # Same limit catches a frozen game mid-lap.
                interface.sock.settimeout(SILENCE_TIMEOUT_S)
                return interface, game_pid
            click_first_profile(game_pid)
        raise RuntimeError("TMInterface never accepted the connection (stuck before the main menu?)")
    except BaseException:
        interface.close()
        raise


def main() -> int:
    local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, default=PROJECT_ROOT / "datasets" / "replays" / "tmx_a01")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "datasets" / "replay_runs")
    parser.add_argument("--count", type=int, default=0, help="laps to record this session (0 = all remaining)")
    parser.add_argument("--speed", type=float, default=4.0, help="game speed between frame captures")
    parser.add_argument("--offset-ms", type=int, default=None, help="skip calibration and use this offset")
    parser.add_argument("--map-id", default="A01-Race")
    parser.add_argument("--map", dest="map_command", default='"Official Maps\\A01-Race.Challenge.Gbx"')
    parser.add_argument("--loader", type=Path, default=local_app_data / "TMLoader" / "TMLoader.exe")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument(
        "--keep-game-open",
        action="store_true",
        help="reuse one game for all laps instead of exiting it after every lap",
    )
    args = parser.parse_args()

    index_path = args.replays / "index.json"
    if not index_path.exists():
        print(f"No replay index at {index_path}; run scripts/fetch_tmx_replays.py first.", file=sys.stderr)
        return 1
    index = json.loads(index_path.read_text(encoding="utf-8"))
    calibration_path = args.output_root / "calibration.json"
    offset = args.offset_ms
    if offset is None and calibration_path.exists():
        offset = json.loads(calibration_path.read_text(encoding="utf-8"))["input_offset_ms"]
    done = finished_replay_ids(args.output_root, offset)
    queue = [replay_id for replay_id, entry in index.items() if entry["usable"] and replay_id not in done]
    if args.count:
        queue = queue[: args.count]
    print(f"{len(queue)} replays to record ({len(done)} already done).", flush=True)
    print("Per lap: open TrackMania -> windowed -> Admin profile -> no ghosts -> record -> exit.", flush=True)

    if sys.platform == "win32":
        # Keep the PC from idle-sleeping while this process records
        # (ES_CONTINUOUS | ES_SYSTEM_REQUIRED); released when it exits.
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    ghost_controller = OpponentDialogController()
    if not args.keep_game_open and close_trackmania():
        print("Closed the running TrackMania so the first lap starts from a fresh game.", flush=True)

    def record_with_recovery(replay, config, input_offset_ms):
        """One lap per fresh game: open -> windowed -> profile -> no ghosts ->
        record -> exit. Any game-side failure exits the game and retries."""
        for attempt in range(1, MAX_ATTEMPTS_PER_LAP + 1):
            try:
                interface, game_pid = connect_ready_game(args)
                recorder = ReplayRunRecorder(
                    interface,
                    config,
                    replay,
                    input_offset_ms=input_offset_ms,
                    game_speed=args.speed,
                    schedule_no_ghost_setup=ghost_controller.schedule_for_process,
                )
                result = recorder.record_connected(game_pid=game_pid)
            except FileNotFoundError:
                raise  # TMLoader missing: not recoverable by retrying
            except (OSError, RuntimeError, ValueError) as exc:
                # Game closed, crashed, hung, or stuck in a menu: exit it and
                # start the next attempt from a fresh launch.
                print(f"  game problem on attempt {attempt}/{MAX_ATTEMPTS_PER_LAP} ({type(exc).__name__}: {exc}); restarting the game...", flush=True)
                close_trackmania()
                time.sleep(RECOVERY_WAIT_S)
                continue
            if not args.keep_game_open:
                close_trackmania()
            if result.complete and opponent_ghost_at_start(Path(result.episode_dir)):
                # The opponent dialog picked a ghost; never keep such a lap.
                quarantine(Path(result.episode_dir), "opponent_ghost_visible")
                print(f"  opponent ghost visible at the start on attempt {attempt}/{MAX_ATTEMPTS_PER_LAP}; retrying...", flush=True)
                time.sleep(RECOVERY_WAIT_S)
                continue
            if result.reason in TRANSIENT_REASONS:
                print(f"  lap interrupted ({result.reason}) on attempt {attempt}/{MAX_ATTEMPTS_PER_LAP}; retrying...", flush=True)
                time.sleep(RECOVERY_WAIT_S)
                continue
            return result
        return None

    accepted = rejected = consecutive_failures = consecutive_game_failures = 0
    for position, replay_id in enumerate(queue, start=1):
        entry = index[replay_id]
        replay = load_keyboard_replay(
            args.replays / f"{replay_id}.Replay.Gbx", replay_id=replay_id, player=entry["player"]
        )
        config = RecordingConfig(
            output_root=args.output_root,
            map_id=args.map_id,
            map_command=args.map_command,
            behavior_mode="fast" if replay.race_time_ms < 26_000 else "safe",
            max_run_seconds=max(180.0, replay.race_time_ms / 1000 + 5),
        )
        candidates = (offset,) if offset is not None else CALIBRATION_CANDIDATES_MS
        result = None
        try:
            for candidate in candidates:
                result = record_with_recovery(replay, config, candidate)
                if result is None:
                    break
                if result.complete and offset is None:
                    offset = candidate
                    calibration_path.parent.mkdir(parents=True, exist_ok=True)
                    calibration_path.write_text(
                        json.dumps({"input_offset_ms": offset, "verified_with_replay": replay_id}, indent=2),
                        encoding="utf-8",
                    )
                    print(f"Calibrated input offset: {offset} ms (verified on replay {replay_id}).", flush=True)
                if result.complete or offset is not None:
                    break
                print(f"  offset {candidate} ms did not reproduce the lap ({result.reason}); trying next.", flush=True)
        except KeyboardInterrupt:
            print("Stopped with Ctrl+C; the partial lap was moved to failed/. Run again to continue.")
            return 130
        except FileNotFoundError as exc:
            print(f"Cannot launch the game: {exc}", file=sys.stderr)
            return 1

        if result is None:
            consecutive_game_failures += 1
            print(f"[{position}/{len(queue)}] SKIP {replay_id}: the game failed {MAX_ATTEMPTS_PER_LAP} times; will retry on the next run.", flush=True)
            if consecutive_game_failures >= MAX_CONSECUTIVE_GAME_FAILURES:
                print("The game keeps failing; stopping. Run the same command again to continue.", file=sys.stderr)
                return 2
            continue
        consecutive_game_failures = 0
        if result.complete:
            accepted += 1
            consecutive_failures = 0
            print(
                f"[{position}/{len(queue)}] OK   {replay_id} {entry['player'][:20]:20s} "
                f"{replay.race_time_ms / 1000:.2f}s frames={result.frame_count}",
                flush=True,
            )
        else:
            rejected += 1
            consecutive_failures += 1
            print(
                f"[{position}/{len(queue)}] FAIL {replay_id} {entry['player'][:20]:20s} "
                f"{result.reason} checkpoint_errors_ms={list(result.checkpoint_errors_ms)}",
                flush=True,
            )
            if offset is None:
                print("No input offset reproduced the first replay; stopping for diagnosis.", file=sys.stderr)
                return 2
            if consecutive_failures >= 5:
                print("Five laps in a row did not reproduce; stopping for diagnosis.", file=sys.stderr)
                return 2

    print(f"Done: {accepted} laps accepted, {rejected} rejected. Output: {args.output_root}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
