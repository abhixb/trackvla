"""Record frames + telemetry by driving TMNF with a replay's keyboard inputs.

Unlike the human recorder, this one owns the controls: it sets the replay's
held keys on every 10 ms engine step and pauses the game (speed 0) after each
50 ms frame request, so no frame can be missed even at high game speed. The
frame's telemetry is read when the frame arrives, so image, telemetry, and the
control label all describe the same simulation state. A run is accepted only
if every checkpoint time reproduces the original replay.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import time
from typing import Callable

from .config import behavior_prompt
from .dataset_labels import mode_from_finish_time
from .human_recording import EpisodeWriter, RecordingConfig, bgra_to_rgb
from .replay_inputs import ReplayInputs
from .telemetry import TELEMETRY_FIELDS, read_telemetry
from .tmnf_protocol import MessageType, TMInterface

ENGINE_STEP_MS = 10
# Checkpoint callbacks fire inside the step that crosses the line, so the
# last observed step time can trail the replay's checkpoint time by one step.
CHECKPOINT_TOLERANCE_MS = ENGINE_STEP_MS


@dataclass(frozen=True)
class ReplayRecordingResult:
    complete: bool
    reason: str
    episode_dir: object
    frame_count: int
    checkpoint_errors_ms: tuple[int, ...]


class ReplayRunRecorder:
    def __init__(
        self,
        interface: TMInterface,
        config: RecordingConfig,
        replay: ReplayInputs,
        *,
        input_offset_ms: int,
        game_speed: float = 4.0,
        frame_converter: Callable[[bytes, int, int], bytes] = bgra_to_rgb,
        schedule_no_ghost_setup: Callable[[int], Future[bool]] | None = None,
        telemetry_reader: Callable[[TMInterface], dict[str, object]] = read_telemetry,
    ):
        if game_speed <= 0:
            raise ValueError("game_speed must be positive")
        self.interface = interface
        self.config = config
        self.replay = replay
        self.input_offset_ms = input_offset_ms
        self.timeline = replay.timeline(input_offset_ms)
        self.game_speed = game_speed
        self.frame_converter = frame_converter
        self.schedule_no_ghost_setup = schedule_no_ghost_setup
        self.telemetry_reader = telemetry_reader

    def _send_controls(self, race_time_ms: int) -> None:
        controls = self.timeline.state_at(race_time_ms)
        self.interface.set_input_state(
            controls.left, controls.right, controls.accelerate, controls.brake
        )

    def record_connected(self, *, game_pid: int | None = None) -> ReplayRecordingResult:
        try:
            writer = EpisodeWriter(self.config)
        except BaseException:
            self.interface.close()
            raise
        started_at = datetime.now(timezone.utc).isoformat()
        frame_pending = False
        capture_tick: int | None = None
        saw_countdown = False
        saw_race = False
        last_race_time_ms: int | None = None
        checkpoint_current: int | None = None
        checkpoint_target: int | None = None
        checkpoint_times: list[int] = []
        missed_frames = 0
        max_frame_lag_ms = 0
        complete = False
        reason = "interface_disconnected"
        raw_inputs_saved = False
        opponent_dialog_configured = False
        ghost_setup_complete = self.schedule_no_ghost_setup is None
        ghost_future: Future[bool] | None = None
        interface_closed = False
        caught: BaseException | None = None
        max_race_ms = min(
            int(self.config.max_run_seconds * 1000),
            self.replay.race_time_ms + 2_000,
        )

        try:
            while True:
                event = MessageType(self.interface.read_int32())

                if event == MessageType.SC_ON_CONNECT_SYNC:
                    try:
                        self.interface.set_timeout(self.config.interface_timeout_ms)
                        # Every engine step: replay inputs change at 10 ms resolution.
                        self.interface.set_on_step_period(ENGINE_STEP_MS)
                        self.interface.execute_command(f"cam {self.config.camera_number}")
                        self.interface.execute_command("set disable_forced_camera true")
                        self.interface.execute_command("set unfocused_fps_limit false")
                        self.interface.execute_command("set autorewind false")
                        self.interface.execute_command("set skip_map_load_screens false")
                        self.interface.set_speed(1.0)
                        self.interface.execute_command(f"map {self.config.map_command}")
                        if self.schedule_no_ghost_setup is not None:
                            if game_pid is None:
                                raise RuntimeError("game PID is required for ghost suppression")
                            ghost_future = self.schedule_no_ghost_setup(game_pid)
                            writer.add_event("opponent_dialog_setup_scheduled")
                        writer.add_event("connected_and_map_requested", map_command=self.config.map_command)
                    finally:
                        self.interface.respond(event)

                elif event == MessageType.SC_RUN_STEP_SYNC:
                    stop = False
                    try:
                        race_time_ms = self.interface.read_int32()
                        if ghost_future is not None and not ghost_setup_complete:
                            if ghost_future.done():
                                opponent_dialog_configured = ghost_future.result()
                                ghost_setup_complete = True
                            elif race_time_ms >= 0:
                                try:
                                    opponent_dialog_configured = ghost_future.result(timeout=1.0)
                                    ghost_setup_complete = True
                                except TimeoutError:
                                    pass
                            if ghost_setup_complete:
                                writer.add_event(
                                    "opponent_dialog_setup_completed",
                                    method="select_opponent_ghosts_dialog",
                                    configured=opponent_dialog_configured,
                                )
                        if race_time_ms < 0:
                            if saw_race:
                                reason = "run_restarted"
                                stop = True
                            else:
                                if not saw_countdown:
                                    writer.add_event("countdown_started", race_time_ms=race_time_ms)
                                    self.interface.set_speed(self.game_speed)
                                saw_countdown = True
                                self._send_controls(race_time_ms)
                        elif saw_countdown:
                            if not ghost_setup_complete:
                                reason = "opponent_dialog_setup_not_completed"
                                stop = True
                            elif not saw_race:
                                saw_race = True
                                writer.add_event("race_started", race_time_ms=race_time_ms)
                            if stop:
                                pass
                            elif last_race_time_ms is not None and race_time_ms < last_race_time_ms:
                                reason = "race_time_went_backwards"
                                stop = True
                            elif race_time_ms > max_race_ms:
                                reason = "run_timeout"
                                stop = True
                            else:
                                if frame_pending:
                                    # The pause should make this impossible.
                                    missed_frames += 1
                                self._send_controls(race_time_ms)
                                if race_time_ms % self.config.period_ms == 0 and not frame_pending:
                                    # Same sequence linesight uses: sync, pause, then
                                    # let the next render deliver this state's frame.
                                    self.interface.rewind_to_current_state()
                                    self._send_controls(race_time_ms)
                                    self.interface.set_speed(0.0)
                                    self.interface.request_frame(self.config.width, self.config.height)
                                    frame_pending = True
                                    capture_tick = race_time_ms
                            last_race_time_ms = race_time_ms
                    except BaseException:
                        stop = True
                        raise
                    finally:
                        if stop:
                            try:
                                if frame_pending:
                                    self.interface.unrequest_frame()
                                    frame_pending = False
                                self.interface.set_speed(1.0)
                            finally:
                                self.interface.close()
                                interface_closed = True
                        else:
                            self.interface.respond(event)
                    if stop:
                        break

                elif event == MessageType.SC_REQUESTED_FRAME_SYNC:
                    try:
                        raw = self.interface.get_frame_bgra(self.config.width, self.config.height)
                        if frame_pending and capture_tick is not None:
                            # Telemetry of the state actually on screen.
                            telemetry = self.telemetry_reader(self.interface)
                            frame_time = int(telemetry["race_time_ms"])
                            max_frame_lag_ms = max(max_frame_lag_ms, abs(frame_time - capture_tick))
                            controls = self.timeline.state_at(frame_time)
                            writer.add_frame(
                                self.frame_converter(raw, self.config.width, self.config.height),
                                {
                                    "frame_index": writer.frame_count,
                                    "race_time_ms": frame_time,
                                    "capture_tick_ms": capture_tick,
                                    "checkpoint_current": checkpoint_current,
                                    "checkpoint_target": checkpoint_target,
                                    "controls": asdict(controls),
                                    "action_gas_brake_steer": controls.normalized(),
                                    "telemetry": telemetry,
                                    "sample_wall_time_ns": time.time_ns(),
                                },
                            )
                            frame_pending = False
                            capture_tick = None
                        self.interface.set_speed(self.game_speed)
                    finally:
                        self.interface.respond(event)

                elif event == MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC:
                    finish_candidate = False
                    try:
                        checkpoint_current = self.interface.read_int32()
                        checkpoint_target = self.interface.read_int32()
                        if saw_race and last_race_time_ms is not None and checkpoint_current > 0:
                            checkpoint_times.append(last_race_time_ms)
                        writer.add_event(
                            "checkpoint",
                            race_time_ms=last_race_time_ms,
                            current=checkpoint_current,
                            target=checkpoint_target,
                        )
                        finish_candidate = (
                            saw_race
                            and checkpoint_target > 0
                            and checkpoint_current == checkpoint_target
                        )
                        if finish_candidate:
                            try:
                                writer.save_tm_inputs(self.interface.get_inputs_text())
                                raw_inputs_saved = True
                            except (OSError, UnicodeError, ConnectionError) as exc:
                                writer.add_event("tm_inputs_error", error=repr(exc))
                            # As in the human recorder: shut down inside this
                            # synchronous callback to avoid a dead-socket freeze.
                            if frame_pending:
                                self.interface.unrequest_frame()
                                frame_pending = False
                            self.interface.set_speed(1.0)
                            self.interface.close()
                            interface_closed = True
                    finally:
                        if not finish_candidate and not interface_closed:
                            self.interface.respond(event)

                    if finish_candidate:
                        writer.add_event("finish_disconnect_completed_in_callback")
                        reason = "finished"
                        break

                elif event == MessageType.SC_LAP_COUNT_CHANGED_SYNC:
                    try:
                        writer.add_event(
                            "lap",
                            current=self.interface.read_int32(),
                            target=self.interface.read_int32(),
                        )
                    finally:
                        self.interface.respond(event)

                elif event == MessageType.C_SHUTDOWN:
                    reason = "interface_shutdown"
                    break

                else:
                    raise RuntimeError(f"unexpected TMInterface event {event}")
        except BaseException as exc:
            caught = exc
            reason = f"error_{type(exc).__name__}"
            try:
                writer.add_event("recording_error", error=repr(exc))
            except OSError:
                pass
        finally:
            if not interface_closed:
                try:
                    if frame_pending:
                        self.interface.unrequest_frame()
                    self.interface.set_speed(1.0)
                except OSError:
                    pass
                finally:
                    self.interface.close()

        expected = self.replay.checkpoint_times_ms
        checkpoint_errors = tuple(
            observed - target for observed, target in zip(checkpoint_times, expected)
        )
        if reason == "finished":
            if len(checkpoint_times) != len(expected):
                reason = "checkpoint_count_mismatch"
            elif any(
                not -CHECKPOINT_TOLERANCE_MS <= error <= 0 for error in checkpoint_errors
            ):
                reason = "replay_not_reproduced"
            elif writer.frame_count < self.config.min_frames:
                reason = "insufficient_frames"
            elif missed_frames:
                reason = "missed_frames"
            else:
                complete = True

        mode = mode_from_finish_time(self.replay.race_time_ms, self.config.fast_threshold_ms)
        manifest: dict[str, object] = {
            "schema_version": 2,
            "status": "complete" if complete else "failed",
            "reason": reason,
            "run_id": writer.run_id,
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "game_pid": game_pid,
            "map_id": self.config.map_id,
            "map_command": self.config.map_command,
            "behavior_mode": mode,
            "language_instruction": behavior_prompt(mode),
            "behavior_labeling": {
                "method": "finish_time_threshold",
                "fast_if_below_ms": self.config.fast_threshold_ms,
            },
            "source": {
                "kind": "replay",
                "replay_id": self.replay.replay_id,
                "player": self.replay.player,
                "replay_race_time_ms": self.replay.race_time_ms,
                "replay_checkpoint_times_ms": list(expected),
                "observed_checkpoint_times_ms": checkpoint_times,
                "checkpoint_errors_ms": list(checkpoint_errors),
                "input_offset_ms": self.input_offset_ms,
                "game_speed": self.game_speed,
            },
            "capture": {
                "width": self.config.width,
                "height": self.config.height,
                "channels": 3,
                "dtype": "uint8",
                "layout": "consecutive RGB frames in frames.rgb",
                "period_ms": self.config.period_ms,
                "frame_count": writer.frame_count,
                "max_frame_lag_ms": max_frame_lag_ms,
            },
            "controls": {
                "source": "replay keyboard timeline (labels are the replayed inputs)",
                "normalized_order": ["gas", "brake", "steer"],
                "authoritative_tminterface_timeline": raw_inputs_saved,
            },
            "telemetry": {
                "source": "TMInterface C_GET_SIMULATION_STATE when the frame arrives",
                "fields": list(TELEMETRY_FIELDS),
                "car_frame_axes": "x lateral, y up, z forward",
            },
            "visual_environment": {
                "ghost_suppression_method": (
                    "select_opponent_ghosts_dialog"
                    if self.schedule_no_ghost_setup is not None
                    else None
                ),
                "ghost_setup_completed_before_race": ghost_setup_complete,
                "opponent_dialog_configured": opponent_dialog_configured,
            },
            "finish_detection": {
                "source": "TMInterface SC_CHECKPOINT_COUNT_CHANGED_SYNC",
                "checkpoint_current": checkpoint_current,
                "checkpoint_target": checkpoint_target,
                "race_time_ms_at_last_sample": last_race_time_ms,
            },
            "quality": {
                "saw_countdown": saw_countdown,
                "saw_race": saw_race,
                "missed_frames": missed_frames,
                "minimum_frames": self.config.min_frames,
            },
            "files": {
                "frames": "frames.rgb",
                "steps": "steps.jsonl",
                "events": "events.jsonl",
                "authoritative_inputs": "tm_inputs.txt" if raw_inputs_saved else None,
            },
        }
        episode_dir = writer.finalize(manifest, complete)
        result = ReplayRecordingResult(
            complete, reason, episode_dir, writer.frame_count, checkpoint_errors
        )
        if caught is not None:
            raise caught
        return result
