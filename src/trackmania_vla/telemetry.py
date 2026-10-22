"""Per-step vehicle telemetry from TMInterface's simulation state.

The state is fetched inside the same ``SC_RUN_STEP_SYNC`` callback that samples
the human controls and requests the frame, so telemetry, controls, and image
share one simulation timestamp. Parsing uses the ``tminterface`` struct layout
(the same one linesight uses for TMNF).
"""

from __future__ import annotations

import math

# Imported eagerly: a lazy first import inside the synchronous step callback
# stalled the game ~0.5 s at race start, which TMNF then skipped through,
# dropping frames.
from tminterface.constants import SIM_HAS_DYNA, SIM_HAS_PLAYER_INFO
from tminterface.structs import SimStateData

from .tmnf_protocol import TMInterface

TELEMETRY_FIELDS = (
    "race_time_ms",
    "display_speed_kmh",
    "speed_mps",
    "rpm",
    "gear",
    "local_velocity_mps",
    "local_angular_velocity_rps",
    "wheels_ground_contact",
    "wheels_sliding",
    "car_sliding",
    "lateral_contact",
    "position_m",
    "forward_world",
)


def _rotate_into_car_frame(rotation: list[list[float]], vector: list[float]) -> list[float]:
    # TMNF stores the car-to-world rotation; its transpose maps world vectors
    # into the car frame (x: lateral, y: up, z: forward), matching linesight.
    return [
        sum(rotation[row][column] * vector[row] for row in range(3))
        for column in range(3)
    ]


def parse_simulation_state(raw: bytes) -> dict[str, object]:
    """Extract the plan-v5 telemetry fields from a raw SimStateData buffer."""
    state = SimStateData(raw)
    required = SIM_HAS_DYNA | SIM_HAS_PLAYER_INFO
    if (state.flags & required) != required:
        raise ValueError(f"simulation state is missing dyna/player info (flags={state.flags:#x})")

    dyna = state.dyna.current_state
    rotation = [[float(value) for value in row] for row in dyna.rotation.to_numpy()]
    velocity = [float(value) for value in dyna.linear_speed]
    angular = [float(value) for value in dyna.angular_speed]
    mobil = state.scene_mobil
    wheels = [state.simulation_wheels[index].real_time_state for index in range(4)]
    telemetry = {
        "race_time_ms": int(state.race_time),
        "display_speed_kmh": int(state.display_speed),
        "speed_mps": math.sqrt(sum(component * component for component in velocity)),
        "rpm": float(mobil.engine.actual_rpm),
        "gear": int(mobil.engine.gear),
        "local_velocity_mps": _rotate_into_car_frame(rotation, velocity),
        "local_angular_velocity_rps": _rotate_into_car_frame(rotation, angular),
        "wheels_ground_contact": [bool(wheel.has_ground_contact) for wheel in wheels],
        "wheels_sliding": [bool(wheel.is_sliding) for wheel in wheels],
        "car_sliding": bool(mobil.is_sliding),
        "lateral_contact": bool(mobil.has_any_lateral_contact),
        # World position and the car's forward axis (third rotation column),
        # used by the racing-line follower that labels DAgger states.
        "position_m": [float(value) for value in dyna.position],
        "forward_world": [rotation[row][2] for row in range(3)],
    }
    for key, value in telemetry.items():
        numbers = value if isinstance(value, list) else [value]
        if not all(math.isfinite(float(number)) for number in numbers):
            raise ValueError(f"telemetry field {key} is not finite")
    return telemetry


def read_telemetry(interface: TMInterface) -> dict[str, object]:
    return parse_simulation_state(interface.get_simulation_state_bytes())
