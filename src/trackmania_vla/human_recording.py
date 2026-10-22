"""Safe, single-episode human demonstration recording for TMNF.

The recorder is intentionally separate from the policy environment: it never
sends driving inputs. It samples the human's keyboard, requests matching game
frames, and also stores TMInterface's authoritative input-event timeline.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import ctypes
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Callable, Protocol
from uuid import uuid4

from .config import BEHAVIOR_MODES, behavior_prompt
from .dataset_labels import mode_from_finish_time
from .telemetry import TELEMETRY_FIELDS, read_telemetry
from .tmnf_protocol import MessageType, TMInterface
from .tmnf_window import wait_for_windowed_trackmania


@dataclass(frozen=True)
class RecordingConfig:
    output_root: Path
    map_id: str = "A01-Race"
    map_command: str = '"Official Maps\\A01-Race.Challenge.Gbx"'
    behavior_mode: str = "safe"
    width: int = 160
    height: int = 120
    period_ms: int = 50
    camera_number: int = 2
    interface_timeout_ms: int = 60_000
    max_run_seconds: float = 180.0
    min_frames: int = 20
    fast_threshold_ms: int = 26_000

    def __post_init__(self) -> None:
        if self.behavior_mode not in BEHAVIOR_MODES:
            raise ValueError(f"behavior_mode must be one of {tuple(BEHAVIOR_MODES)}")
        if not self.map_id.strip() or not self.map_command.strip():
            raise ValueError("map_id and map_command cannot be empty")
        if "\n" in self.map_command or "\r" in self.map_command:
            raise ValueError("map_command must fit on one line")
        if self.width < 1 or self.height < 1:
            raise ValueError("frame dimensions must be positive")
        if self.period_ms < 10 or self.period_ms % 10:
            raise ValueError("period_ms must be a multiple of TMNF's 10 ms engine step")
        if self.max_run_seconds <= 0 or self.min_frames < 1:
            raise ValueError("run limit and minimum frame count must be positive")
        if self.fast_threshold_ms <= 0:
            raise ValueError("fast_threshold_ms must be positive")


@dataclass(frozen=True)
class ControlState:
    left: bool
    right: bool
    accelerate: bool
    brake: bool

    @property
    def conflict(self) -> bool:
        return self.left and self.right

    def normalized(self) -> list[float]:
        steer = -1.0 if self.left else (1.0 if self.right else 0.0)
        return [1.0 if self.accelerate else -1.0, 1.0 if self.brake else -1.0, steer]


class KeyboardSampler(Protocol):
    def sample(self) -> ControlState: ...


class WindowsKeyboardSampler:
    """Sample the default TMNF arrow/WASD bindings without controlling them."""

    VK_LEFT = 0x25
    VK_UP = 0x26
    VK_RIGHT = 0x27
    VK_DOWN = 0x28
    VK_A = 0x41
    VK_D = 0x44
    VK_S = 0x53
    VK_W = 0x57

    def __init__(self) -> None:
        if os.name != "nt":
            raise RuntimeError("human keyboard sampling is only supported on Windows")
        self._get_key = ctypes.windll.user32.GetAsyncKeyState  # type: ignore[attr-defined]

    def _down(self, virtual_key: int) -> bool:
        return bool(self._get_key(virtual_key) & 0x8000)

    def sample(self) -> ControlState:
        return ControlState(
            left=self._down(self.VK_LEFT) or self._down(self.VK_A),
            right=self._down(self.VK_RIGHT) or self._down(self.VK_D),
            accelerate=self._down(self.VK_UP) or self._down(self.VK_W),
            brake=self._down(self.VK_DOWN) or self._down(self.VK_S),
        )


@dataclass(frozen=True)
class RecordingResult:
    complete: bool
    reason: str
    episode_dir: Path
    frame_count: int
    finish_race_time_ms: int | None


def bgra_to_rgb(raw: bytes, width: int, height: int) -> bytes:
    """Convert one tightly packed BGRA frame to tightly packed RGB."""
    if len(raw) != width * height * 4:
        raise ValueError("incomplete TMInterface BGRA frame")
    rgb = bytearray(width * height * 3)
    rgb[0::3] = raw[2::4]
    rgb[1::3] = raw[1::4]
    rgb[2::3] = raw[0::4]
    return bytes(rgb)


class EpisodeWriter:
    """Write an episode transactionally; only validated runs enter episodes/."""

    def __init__(self, config: RecordingConfig):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{stamp}_{config.map_id}_{config.behavior_mode}_{uuid4().hex[:8]}"
        self.root = config.output_root.resolve()
        self.working_dir = self.root / "_incomplete" / self.run_id
        self.working_dir.mkdir(parents=True, exist_ok=False)
        self.frames = (self.working_dir / "frames.rgb").open("wb")
        self.steps = (self.working_dir / "steps.jsonl").open("w", encoding="utf-8", newline="\n")
        self.events = (self.working_dir / "events.jsonl").open("w", encoding="utf-8", newline="\n")
        self.frame_count = 0
        self._closed = False

    def add_frame(self, rgb: bytes, metadata: dict[str, object]) -> None:
        self.frames.write(rgb)
        self.steps.write(json.dumps(metadata, separators=(",", ":")) + "\n")
        self.frame_count += 1

    def add_event(self, kind: str, **fields: object) -> None:
        payload = {"event": kind, "wall_time_ns": time.time_ns(), **fields}
        self.events.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.events.flush()

    def save_tm_inputs(self, text: str) -> None:
        (self.working_dir / "tm_inputs.txt").write_text(text, encoding="utf-8")

    def finalize(self, manifest: dict[str, object], complete: bool) -> Path:
        if self._closed:
            raise RuntimeError("episode writer was already finalized")
        self._closed = True
        for stream in (self.frames, self.steps, self.events):
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()

        manifest_path = self.working_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        destination_parent = self.root / ("episodes" if complete else "failed")
        destination_parent.mkdir(parents=True, exist_ok=True)
        destination = destination_parent / self.run_id
        self.working_dir.replace(destination)
        return destination


class HumanRunRecorder:
    def __init__(
        self,
        interface: TMInterface,
        config: RecordingConfig,
        keyboard: KeyboardSampler,
        *,
        frame_converter: Callable[[bytes, int, int], bytes] = bgra_to_rgb,
        schedule_no_ghost_setup: Callable[[int], Future[bool]] | None = None,
        telemetry_reader: Callable[[TMInterface], dict[str, object]] = read_telemetry,
    ):
        self.interface = interface
        self.telemetry_reader = telemetry_reader
        self.config = config
        self.keyboard = keyboard
        self.frame_converter = frame_converter
        self.schedule_no_ghost_setup = schedule_no_ghost_setup

    def record_connected(self, *, game_pid: int | None = None) -> RecordingResult:
        """Record one run from an already-connected TMInterface socket."""
        try:
            writer = EpisodeWriter(self.config)
        except BaseException:
            self.interface.close()
            raise
        started_at = datetime.now(timezone.utc).isoformat()
        pending: dict[str, object] | None = None
        frame_pending = False
        saw_countdown = False
        saw_race = False
        last_race_time_ms: int | None = None
        checkpoint_current: int | None = None
        checkpoint_target: int | None = None
        missed_frames = 0
        input_conflicts = 0
        max_telemetry_time_offset_ms = 0
        complete = False
        reason = "interface_disconnected"
        raw_inputs_saved = False
        opponent_dialog_configured = False
        ghost_setup_complete = self.schedule_no_ghost_setup is None
        ghost_future: Future[bool] | None = None
        interface_closed = False
        caught: BaseException | None = None

        try:
            while True:
                event = MessageType(self.interface.read_int32())

                if event == MessageType.SC_ON_CONNECT_SYNC:
                    try:
                        self.interface.set_timeout(self.config.interface_timeout_ms)
                        self.interface.set_on_step_period(self.config.period_ms)
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
                                # The OK click can produce race time zero just
                                # before the dialog thread publishes its result.
                                # It no longer waits on rendering, so this short
                                # synchronization cannot deadlock the game loop.
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
                                saw_countdown = True
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
                            elif race_time_ms > int(self.config.max_run_seconds * 1000):
                                reason = "run_timeout"
                                stop = True
                            elif frame_pending:
                                missed_frames += 1
                            else:
                                controls = self.keyboard.sample()
                                input_conflicts += int(controls.conflict)
                                # Read inside this same synchronous step so the
                                # telemetry shares the frame/control timestamp.
                                telemetry = self.telemetry_reader(self.interface)
                                max_telemetry_time_offset_ms = max(
                                    max_telemetry_time_offset_ms,
                                    abs(int(telemetry["race_time_ms"]) - race_time_ms),
                                )
                                pending = {
                                    "frame_index": writer.frame_count,
                                    "race_time_ms": race_time_ms,
                                    "checkpoint_current": checkpoint_current,
                                    "checkpoint_target": checkpoint_target,
                                    "controls": asdict(controls),
                                    "action_gas_brake_steer": controls.normalized(),
                                    "telemetry": telemetry,
                                    "sample_wall_time_ns": time.time_ns(),
                                }
                                self.interface.request_frame(self.config.width, self.config.height)
                                frame_pending = True
                            last_race_time_ms = race_time_ms
                    except BaseException:
                        # A synchronous callback must be answered or shut down
                        # before its socket is closed, otherwise TMInterface can
                        # leave the game blocked on the abandoned callback.
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
                        if pending is not None:
                            writer.add_frame(
                                self.frame_converter(raw, self.config.width, self.config.height), pending
                            )
                            pending = None
                            frame_pending = False
                    finally:
                        self.interface.respond(event)

                elif event == MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC:
                    finish_candidate = False
                    try:
                        checkpoint_current = self.interface.read_int32()
                        checkpoint_target = self.interface.read_int32()
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
                            # C_SHUTDOWN must be consumed while the plugin is still
                            # waiting in this synchronous callback. A normal event
                            # response followed by shutdown races the next callback
                            # and can leave the game blocked on a dead socket.
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
                        if writer.frame_count < self.config.min_frames:
                            reason = "insufficient_frames"
                        elif missed_frames:
                            reason = "missed_frames"
                        elif input_conflicts:
                            reason = "left_right_input_conflict"
                        elif not raw_inputs_saved:
                            reason = "authoritative_inputs_unavailable"
                        else:
                            complete = True
                            reason = "finished"
                        break

                elif event == MessageType.SC_LAP_COUNT_CHANGED_SYNC:
                    try:
                        current_lap = self.interface.read_int32()
                        target_laps = self.interface.read_int32()
                        writer.add_event("lap", current=current_lap, target=target_laps)
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

        effective_behavior_mode = self.config.behavior_mode
        behavior_labeling: dict[str, object] = {"method": "requested_mode"}
        if complete and last_race_time_ms is not None:
            effective_behavior_mode = mode_from_finish_time(
                last_race_time_ms,
                self.config.fast_threshold_ms,
            )
            behavior_labeling = {
                "method": "finish_time_threshold",
                "fast_if_below_ms": self.config.fast_threshold_ms,
            }

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
            "behavior_mode": effective_behavior_mode,
            "language_instruction": behavior_prompt(effective_behavior_mode),
            "behavior_labeling": behavior_labeling,
            "capture": {
                "width": self.config.width,
                "height": self.config.height,
                "channels": 3,
                "dtype": "uint8",
                "layout": "consecutive RGB frames in frames.rgb",
                "period_ms": self.config.period_ms,
                "frame_count": writer.frame_count,
            },
            "controls": {
                "sampled_bindings": ["arrow keys", "WASD"],
                "normalized_order": ["gas", "brake", "steer"],
                "authoritative_tminterface_timeline": raw_inputs_saved,
            },
            "telemetry": {
                "source": "TMInterface C_GET_SIMULATION_STATE in the frame's SC_RUN_STEP_SYNC callback",
                "fields": list(TELEMETRY_FIELDS),
                "car_frame_axes": "x lateral, y up, z forward",
                "max_race_time_offset_ms": max_telemetry_time_offset_ms,
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
                "left_right_input_conflicts": input_conflicts,
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
        result = RecordingResult(complete, reason, episode_dir, writer.frame_count, last_race_time_ms)
        if caught is not None:
            raise caught
        return result


@dataclass(frozen=True)
class LaunchInfo:
    loader_pid: int
    game_pid: int


def _tmforever_pids() -> set[int]:
    import psutil

    pids: set[int] = set()
    for process in psutil.process_iter(("pid", "name")):
        try:
            if (process.info["name"] or "").startswith("TmForever"):
                pids.add(int(process.info["pid"]))
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return pids


def launch_trackmania(
    loader_path: Path,
    profile: str,
    port: int,
    *,
    timeout_s: float = 60.0,
) -> LaunchInfo:
    """Launch via TMLoader, but never focus, resize, or kill the game."""
    if not loader_path.is_file():
        raise FileNotFoundError(f"TMLoader was not found at {loader_path}")
    existing = _tmforever_pids()
    if existing:
        raise RuntimeError(
            f"TmForever is already running (PID(s) {sorted(existing)}); use --no-launch to attach explicitly"
        )
    command = [
        str(loader_path),
        "run",
        "TmForever",
        profile,
        f"/configstring=set custom_port {port}",
    ]
    loader = subprocess.Popen(command, cwd=loader_path.parent)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        candidates = _tmforever_pids() - existing
        if len(candidates) == 1:
            return LaunchInfo(loader.pid, candidates.pop())
        if len(candidates) > 1:
            raise RuntimeError(f"TMLoader created multiple TmForever processes: {sorted(candidates)}")
        time.sleep(0.1)
    raise TimeoutError("TMLoader started, but no TmForever process appeared before the launch timeout")


def connect_tminterface(port: int, timeout_s: float = 60.0) -> TMInterface:
    """Retry localhost connection while the TMInterface plugin initializes."""
    deadline = time.monotonic() + timeout_s
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        interface = TMInterface(port)
        try:
            interface.register(timeout_s=min(2.0, max(0.1, deadline - time.monotonic())))
            assert interface.sock is not None
            interface.sock.settimeout(max(5.0, timeout_s))
            return interface
        except OSError as exc:
            last_error = exc
            time.sleep(0.25)
    raise TimeoutError(f"could not connect to TMInterface on port {port}: {last_error}")


def launch_and_connect(
    loader_path: Path,
    profile: str,
    port: int,
    *,
    launch: bool = True,
    timeout_s: float = 60.0,
    require_windowed: bool = True,
) -> tuple[TMInterface, int | None]:
    existing = _tmforever_pids()
    if len(existing) > 1:
        raise RuntimeError(f"multiple TmForever processes are running: {sorted(existing)}")
    if existing:
        game_pid = next(iter(existing))
    elif launch:
        game_pid = launch_trackmania(loader_path, profile, port, timeout_s=timeout_s).game_pid
    else:
        raise RuntimeError("--no-launch was requested, but no TmForever process is running")

    wait_for_windowed_trackmania(timeout_s=timeout_s, require_windowed=require_windowed)
    try:
        return connect_tminterface(port, timeout_s), game_pid
    except (OSError, TimeoutError):
        # Deliberately leave the game open so a crash never strands the user.
        raise
