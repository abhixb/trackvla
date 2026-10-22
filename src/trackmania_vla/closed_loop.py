"""Closed-loop driving of A01 by the Qwen policy through TMInterface (Phase 4).

A frame is captured every 50 ms of race time; the game is paused (speed 0)
until it arrives, and its telemetry is read at that moment, exactly as in the
replay recorder. Every ``replan_ms`` the policy sees the last ``history``
frames (``stride`` captures apart) and the telemetry, and predicts a chunk of
four 50 ms actions; inference also happens while the game is paused, so the
run is causally valid but slower than real time.

Planners return either class ids (v1: the chunk's actions are applied on every
10 ms engine step until the next plan) or probabilities (v2). With
probabilities, every overlapping chunk that covers a 50 ms slot is averaged
(ACT temporal ensembling), throttle is the argmax of the average, and steering
is read from the averaged distribution by ``steer_rule`` (see ChunkEnsemble),
played as left/right taps by a sigma-delta
modulator so the held fraction matches; ``pwm_ms`` is the tap resolution
(10 = every engine step, 50 = whole frames, like the experts' taps).
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import time

import numpy as np
from PIL import Image

from .human_recording import ControlState, EpisodeWriter, RecordingConfig, bgra_to_rgb
from .replay_dataset import telemetry_text
from .telemetry import read_telemetry
from .tmnf_protocol import MessageType, TMInterface

ENGINE_STEP_MS = 10
STEER_TO_KEYS = {0: (True, False), 1: (False, False), 2: (False, True)}  # left, right
THROTTLE_TO_KEYS = {0: (False, False), 1: (True, False), 2: (False, True)}  # accelerate, brake


def chunk_controls(steer: np.ndarray, throttle: np.ndarray) -> list[ControlState]:
    controls = []
    for s, t in zip(steer.tolist(), throttle.tolist()):
        left, right = STEER_TO_KEYS[s]
        accelerate, brake = THROTTLE_TO_KEYS[t]
        controls.append(ControlState(left, right, accelerate, brake))
    return controls


@dataclass(frozen=True)
class DriveResult:
    finished: bool
    reason: str
    race_time_ms: int | None
    checkpoints: int
    checkpoint_target: int | None
    plans: int
    episode_dir: object


# planner(*frames oldest first, telemetry dict) -> (steer[K], throttle[K]) class
# ids, or (steer[K, bins], throttle[K, 3]) probabilities with a ``steer_values``
# attribute giving each steer bin's amount.
Planner = Callable[..., tuple[np.ndarray, np.ndarray]]


STEER_RULES = ("expected", "argmax", "rounded")


class ChunkEnsemble:
    """Average of every stored chunk's probabilities for one 50 ms slot.

    Steering amount from the averaged distribution:
      expected  probability-weighted mean of the levels (never exactly 0 on a
                straight, so it keeps tapping), zeroed below ``deadzone``
      argmax    the single most likely level
      rounded   the expected amount snapped to the nearest level
    """

    def __init__(self, steer_values: np.ndarray, period: int, rule: str = "expected", deadzone: float = 0.0):
        if rule not in STEER_RULES:
            raise ValueError(f"unknown steer rule {rule!r}")
        self.steer_values = np.asarray(steer_values, float)
        self.period = period
        self.rule = rule
        self.deadzone = deadzone
        self.chunks: list[tuple[int, np.ndarray, np.ndarray]] = []

    def add(self, start: int, steer: np.ndarray, throttle: np.ndarray) -> None:
        self.chunks.append((start, steer, throttle))

    def clear(self) -> None:
        self.chunks.clear()

    def action(self, t: int) -> tuple[float, int] | None:
        steer, throttle = [], []
        keep = []
        for start, s, th in self.chunks:
            k = (t - start) // self.period
            if 0 <= k < len(s):
                steer.append(s[k]); throttle.append(th[k])
            if k < len(s) - 1:
                keep.append((start, s, th))
        self.chunks = keep
        if not steer:
            return None
        probabilities = np.mean(steer, axis=0)
        if self.rule == "argmax":
            duty = float(self.steer_values[probabilities.argmax()])
        else:
            duty = float(probabilities @ self.steer_values)
            if self.rule == "rounded":
                duty = float(self.steer_values[np.abs(self.steer_values - duty).argmin()])
        if abs(duty) < self.deadzone:
            duty = 0.0
        return duty, int(np.mean(throttle, axis=0).argmax())


class SteerFilter:
    """Anti-weave filter on the steering amount, applied every 50 ms slot.

    ``smooth_ms``  exponential moving average with this time constant.
    ``switch``     hysteresis: once steering one way, the other way is only
                   taken when it reaches this amount; smaller opposite values
                   go straight. The side is released after the amount has been
                   inside ``deadzone`` for ``release_ms``.
    """

    def __init__(self, period_ms: int, *, smooth_ms: int = 0, switch: float = 0.0,
                 deadzone: float = 0.0, release_ms: int = 100):
        self.alpha = 1.0 if smooth_ms <= 0 else period_ms / (smooth_ms + period_ms)
        self.period_ms = period_ms
        self.switch = switch
        self.deadzone = deadzone
        self.release_ms = release_ms
        self.value = 0.0
        self.side = 0
        self.quiet_ms = 0

    def __call__(self, duty: float) -> float:
        self.value += self.alpha * (duty - self.value)
        value = self.value
        if abs(value) <= self.deadzone:
            self.quiet_ms += self.period_ms
            if self.quiet_ms >= self.release_ms:
                self.side = 0
            return 0.0
        self.quiet_ms = 0
        if not self.switch:
            return value
        side = 1 if value > 0 else -1
        if self.side in (0, side) or abs(value) >= self.switch:
            self.side = side
            return value
        return 0.0  # small opposite correction: go straight instead


class SigmaDelta:
    """Turns a steering amount into per-step key presses with that duty."""

    def __init__(self):
        self.error = 0.0

    def step(self, duty: float) -> tuple[bool, bool]:
        self.error += abs(duty)
        press = self.error >= 0.5
        if press:
            self.error -= 1.0
        self.error = min(max(self.error, -1.0), 1.0)
        return press and duty < 0, press and duty > 0


# labeler(telemetry dict) -> (steer class, throttle class) from the automatic expert
Labeler = Callable[[dict], tuple[int, int]]


class PolicyDriver:
    def __init__(
        self,
        interface: TMInterface,
        config: RecordingConfig,
        planner: Planner,
        *,
        stride: int = 2,
        history: int = 2,
        replan_ms: int = 200,
        pwm_ms: int = ENGINE_STEP_MS,
        steer_rule: str = "expected",
        steer_deadzone: float = 0.0,
        steer_smooth_ms: int = 0,
        steer_switch: float = 0.0,
        game_speed: float = 4.0,
        max_race_ms: int = 60_000,
        stuck_kmh: float = 10.0,
        stuck_ms: int = 3_000,
        schedule_no_ghost_setup: Callable[[int], Future[bool]] | None = None,
        telemetry_reader: Callable[[TMInterface], dict] = read_telemetry,
        label: str = "",
        labeler: Labeler | None = None,
        expert_probability: float = 0.0,
        takeover: Callable[[dict], bool] | None = None,
        seed: int = 0,
    ):
        self.interface = interface
        self.config = config
        self.planner = planner
        self.stride = stride
        self.history = history
        self.replan_ms = replan_ms
        self.pwm_ms = pwm_ms
        self.steer_rule = steer_rule
        self.steer_deadzone = steer_deadzone
        self.steer_smooth_ms = steer_smooth_ms
        self.steer_switch = steer_switch
        self.game_speed = game_speed
        self.max_race_ms = max_race_ms
        self.stuck_kmh = stuck_kmh
        self.stuck_ms = stuck_ms
        self.schedule_no_ghost_setup = schedule_no_ghost_setup
        self.telemetry_reader = telemetry_reader
        self.label = label
        # DAgger: every frame is labelled by the expert; for each planning
        # window the expert drives with this probability (beta), else the policy.
        self.labeler = labeler
        self.expert_probability = expert_probability
        # Optional trigger checked every frame instead of the random beta:
        # True while the expert should drive (e.g. off-line in a hard section).
        self.takeover = takeover
        self.rng = np.random.default_rng(seed)

    def drive(self, *, game_pid: int | None = None) -> DriveResult:
        writer = EpisodeWriter(self.config)
        started_at = datetime.now(timezone.utc).isoformat()
        period = self.config.period_ms
        frames: dict[int, Image.Image] = {}  # race time -> frame
        plan_start: int | None = None
        plan: list[ControlState] = []
        plans = 0
        current = ControlState(False, False, True, False)  # hold gas until the first plan
        frame_pending = False
        saw_countdown = saw_race = False
        last_race_time: int | None = None
        checkpoint_current = 0
        checkpoint_target: int | None = None
        slow_since: int | None = None
        reason = "interface_disconnected"
        finished = False
        interface_closed = False
        ghost_future: Future[bool] | None = None
        ghost_done = self.schedule_no_ghost_setup is None
        plan_seconds = 0.0
        expert_driving = False
        expert_frames = 0
        steer_values = getattr(self.planner, "steer_values", None)
        ensemble = (
            ChunkEnsemble(steer_values, period, self.steer_rule, self.steer_deadzone)
            if steer_values is not None else None
        )
        modulator = SigmaDelta()
        steer_filter = (
            SteerFilter(period, smooth_ms=self.steer_smooth_ms, switch=self.steer_switch, deadzone=self.steer_deadzone)
            if self.steer_smooth_ms or self.steer_switch else None
        )
        duty: float | None = None  # v2 steering amount in force; None = discrete keys
        pedals = (True, False)

        def modulated() -> ControlState:
            left, right = modulator.step(duty)
            return ControlState(left, right, *pedals)

        def send(controls: ControlState) -> None:
            self.interface.set_input_state(controls.left, controls.right, controls.accelerate, controls.brake)

        try:
            while True:
                event = MessageType(self.interface.read_int32())
                if event == MessageType.SC_ON_CONNECT_SYNC:
                    try:
                        self.interface.set_timeout(self.config.interface_timeout_ms)
                        self.interface.set_on_step_period(ENGINE_STEP_MS)
                        self.interface.execute_command(f"cam {self.config.camera_number}")
                        self.interface.execute_command("set disable_forced_camera true")
                        self.interface.execute_command("set unfocused_fps_limit false")
                        self.interface.execute_command("set autorewind false")
                        self.interface.execute_command("set skip_map_load_screens false")
                        self.interface.set_speed(1.0)
                        self.interface.execute_command(f"map {self.config.map_command}")
                        if self.schedule_no_ghost_setup is not None:
                            ghost_future = self.schedule_no_ghost_setup(game_pid)
                    finally:
                        self.interface.respond(event)

                elif event == MessageType.SC_RUN_STEP_SYNC:
                    stop = False
                    try:
                        t = self.interface.read_int32()
                        if ghost_future is not None and not ghost_done and (ghost_future.done() or t >= 0):
                            try:
                                ghost_future.result(timeout=1.0)
                                ghost_done = True
                            except TimeoutError:
                                pass
                        if t < 0:
                            if saw_race:
                                reason, stop = "run_restarted", True
                            elif not saw_countdown:
                                saw_countdown = True
                                self.interface.set_speed(self.game_speed)
                            send(current)
                        elif saw_countdown:
                            saw_race = True
                            if t > self.max_race_ms:
                                reason, stop = "timeout", True
                            else:
                                if duty is not None:
                                    # Frame steps get their action from the frame handler.
                                    if t % period != 0 and t % self.pwm_ms == 0:
                                        current = modulated()
                                elif plan_start is not None and plan:
                                    index = min((t - plan_start) // period, len(plan) - 1)
                                    current = plan[index]
                                send(current)
                                if t % period == 0 and not frame_pending:
                                    self.interface.rewind_to_current_state()
                                    send(current)
                                    self.interface.set_speed(0.0)
                                    self.interface.request_frame(self.config.width, self.config.height)
                                    frame_pending = True
                            last_race_time = t
                    except BaseException:
                        stop = True
                        raise
                    finally:
                        if stop:
                            try:
                                if frame_pending:
                                    self.interface.unrequest_frame()
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
                        if frame_pending:
                            frame_pending = False
                            telemetry = self.telemetry_reader(self.interface)
                            t = int(telemetry["race_time_ms"])
                            rgb = bgra_to_rgb(raw, self.config.width, self.config.height)
                            frames[t] = Image.frombytes("RGB", (self.config.width, self.config.height), rgb)
                            frames.pop(t - ((self.history - 1) * self.stride + 2) * period, None)
                            new_plan = False
                            expert = self.labeler(telemetry) if self.labeler is not None else None
                            window = [frames.get(t - k * self.stride * period) for k in reversed(range(self.history))]
                            if self.takeover is not None and expert is not None:
                                expert_driving = bool(self.takeover(telemetry))
                            if t % self.replan_ms == 0 and all(frame is not None for frame in window):
                                if self.takeover is None:
                                    expert_driving = expert is not None and self.rng.random() < self.expert_probability
                                if not expert_driving:
                                    begin = time.perf_counter()
                                    steer, throttle = self.planner(*window, telemetry)
                                    plan_seconds += time.perf_counter() - begin
                                    plans, new_plan = plans + 1, True
                                    if ensemble is not None:
                                        ensemble.add(t, steer, throttle)
                                    else:
                                        plan, plan_start = chunk_controls(steer, throttle), t
                                        current = plan[0]
                            if ensemble is not None and not expert_driving:
                                action = ensemble.action(t)
                                if action is not None:
                                    duty, throttle_id = action
                                    if steer_filter is not None:
                                        duty = steer_filter(duty)
                                    pedals = THROTTLE_TO_KEYS[throttle_id]
                                    current = modulated()
                            if expert_driving and expert is not None:
                                current = chunk_controls(np.array([expert[0]]), np.array([expert[1]]))[0]
                                plan, plan_start = [current], t
                                duty = None
                                if ensemble is not None:
                                    ensemble.clear()
                                expert_frames += 1
                            send(current)
                            # Labelers with a continuous steering amount (expert lookup)
                            # also record it; None means "no expert label here".
                            labeler_owner = getattr(self.labeler, "__self__", None)
                            extra_labels = (
                                {"expert_duty": labeler_owner.last_duty}
                                if expert is not None and hasattr(labeler_owner, "last_duty") else {}
                            )
                            expert_controls = (
                                asdict(chunk_controls(np.array([expert[0]]), np.array([expert[1]]))[0])
                                if expert is not None else None
                            )
                            writer.add_frame(rgb, {
                                "frame_index": writer.frame_count,
                                "race_time_ms": t,
                                "controls": asdict(current),
                                "action_gas_brake_steer": current.normalized(),
                                "new_plan": new_plan,
                                "steer_duty": duty,
                                "expert_driving": expert_driving,
                                "expert_controls": expert_controls,
                                "telemetry": telemetry,
                                **extra_labels,
                                "sample_wall_time_ns": time.time_ns(),
                            })
                            speed = float(telemetry["display_speed_kmh"])
                            if t > 2_000 and speed < self.stuck_kmh:
                                slow_since = t if slow_since is None else slow_since
                            else:
                                slow_since = None
                        self.interface.set_speed(self.game_speed)
                    finally:
                        self.interface.respond(event)
                    if slow_since is not None and last_race_time is not None and last_race_time - slow_since >= self.stuck_ms:
                        reason = "stuck"
                        self.interface.set_speed(1.0)
                        self.interface.close()
                        interface_closed = True
                        break

                elif event == MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC:
                    is_finish = False
                    try:
                        checkpoint_current = self.interface.read_int32()
                        checkpoint_target = self.interface.read_int32()
                        writer.add_event("checkpoint", race_time_ms=last_race_time, current=checkpoint_current, target=checkpoint_target)
                        is_finish = saw_race and checkpoint_target > 0 and checkpoint_current == checkpoint_target
                        if is_finish:
                            if frame_pending:
                                self.interface.unrequest_frame()
                            self.interface.set_speed(1.0)
                            self.interface.close()
                            interface_closed = True
                    finally:
                        if not is_finish and not interface_closed:
                            self.interface.respond(event)
                    if is_finish:
                        finished, reason = True, "finished"
                        break

                elif event == MessageType.SC_LAP_COUNT_CHANGED_SYNC:
                    try:
                        self.interface.read_int32(); self.interface.read_int32()
                    finally:
                        self.interface.respond(event)
                elif event == MessageType.C_SHUTDOWN:
                    reason = "interface_shutdown"
                    break
                else:
                    raise RuntimeError(f"unexpected TMInterface event {event}")
        finally:
            if not interface_closed:
                try:
                    self.interface.set_speed(1.0)
                except OSError:
                    pass
                finally:
                    self.interface.close()
            manifest = {
                "schema_version": 2,
                "status": "complete" if finished else "failed",
                "reason": reason,
                "run_id": writer.run_id,
                "started_at_utc": started_at,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "map_id": self.config.map_id,
                "source": {"kind": "policy_closed_loop", "label": self.label},
                "behavior_mode": self.config.behavior_mode,
                "capture": {"width": self.config.width, "height": self.config.height, "period_ms": period, "frame_count": writer.frame_count},
                "result": {
                    "finished": finished,
                    "race_time_ms": last_race_time,
                    "checkpoints": checkpoint_current,
                    "checkpoint_target": checkpoint_target,
                    "plans": plans,
                    "mean_plan_ms": round(1000 * plan_seconds / plans, 1) if plans else None,
                    "expert_frames": expert_frames,
                    "expert_probability": self.expert_probability,
                    "steer_rule": self.steer_rule,
                    "steer_deadzone": self.steer_deadzone,
                    "steer_smooth_ms": self.steer_smooth_ms,
                    "steer_switch": self.steer_switch,
                },
            }
            episode_dir = writer.finalize(manifest, finished)
        return DriveResult(finished, reason, last_race_time, checkpoint_current, checkpoint_target, plans, episode_dir)
