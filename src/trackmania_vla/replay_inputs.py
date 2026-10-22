"""Keyboard input timelines extracted from TMNF ``.Replay.Gbx`` files.

TMNF replays store only control events and the simulation is deterministic, so
feeding a replay's key timeline back through TMInterface reproduces the lap.
Only digital (keyboard) replays are accepted: analog gamepad steering cannot be
represented by the policy's left/straight/right action space.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

from .human_recording import ControlState
from .lzo1x import install as _install_lzo

KEY_EVENTS = {
    "SteerLeft": "left",
    "SteerRight": "right",
    "Accelerate": "accelerate",
    "Brake": "brake",
}
ANALOG_EVENTS = frozenset({"Steer", "Gas", "AccelerateReal", "BrakeReal"})
IGNORED_EVENTS = frozenset({"_FakeIsRaceRunning", "_FakeFinishLine", "_FakeDontInverseAxis", "Horn"})


class ReplayRejected(ValueError):
    """The replay cannot be reproduced with four digital controls."""


@dataclass(frozen=True)
class ReplayInputs:
    replay_id: str
    player: str
    race_time_ms: int
    checkpoint_times_ms: tuple[int, ...]
    # (replay time in ms, control name, pressed), sorted by time.
    events: tuple[tuple[int, str, bool], ...]

    def timeline(self, offset_ms: int) -> "InputTimeline":
        return InputTimeline(self.events, offset_ms)


class InputTimeline:
    """Held-key state at any race time; replay time + offset = race time."""

    def __init__(self, events: tuple[tuple[int, str, bool], ...], offset_ms: int):
        self.offset_ms = offset_ms
        self._times: list[int] = []
        self._states: list[ControlState] = []
        held = {"left": False, "right": False, "accelerate": False, "brake": False}
        for time_ms, control, pressed in events:
            held[control] = pressed
            state = ControlState(**held)
            effective = time_ms + offset_ms
            if self._times and self._times[-1] == effective:
                self._states[-1] = state
            else:
                self._times.append(effective)
                self._states.append(state)

    def state_at(self, race_time_ms: int) -> ControlState:
        index = bisect_right(self._times, race_time_ms) - 1
        if index < 0:
            return ControlState(False, False, False, False)
        return self._states[index]


def load_keyboard_replay(path: Path, *, replay_id: str | None = None, player: str = "") -> ReplayInputs:
    _install_lzo()
    from pygbx import Gbx, GbxType

    ghosts = Gbx(str(path)).get_classes_by_ids([GbxType.CTN_GHOST, GbxType.CTN_GHOST_OLD])
    if not ghosts:
        raise ReplayRejected("no ghost in replay")
    ghost = ghosts[0]
    events: list[tuple[int, str, bool]] = []
    finished = False
    for entry in ghost.control_entries:
        name = entry.event_name
        if name in KEY_EVENTS:
            events.append((int(entry.time), KEY_EVENTS[name], entry.enabled != 0))
        elif name in ANALOG_EVENTS:
            raise ReplayRejected(f"analog input ({name})")
        elif name == "Respawn":
            # Countdown restarts are harmless; a mid-race respawn is not
            # reproducible with the four digital controls.
            if entry.time >= 0:
                raise ReplayRejected("respawn during the race")
        elif name == "_FakeFinishLine":
            finished = True
        elif name not in IGNORED_EVENTS:
            raise ReplayRejected(f"unsupported input event {name!r}")
    if not finished:
        raise ReplayRejected("replay has no finish")
    if not any(control in ("left", "right") for _, control, _ in events):
        raise ReplayRejected("no keyboard steering events")
    events.sort(key=lambda event: event[0])
    return ReplayInputs(
        replay_id=replay_id or path.name.split(".")[0],
        player=player or str(getattr(ghost, "login", "") or ""),
        race_time_ms=int(ghost.race_time),
        checkpoint_times_ms=tuple(int(time_ms) for time_ms in ghost.cp_times),
        events=tuple(events),
    )
