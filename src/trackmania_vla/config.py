from dataclasses import dataclass


BEHAVIOR_MODES = {
    "fast": "Drive a fast lap. Prioritize progress and speed while staying on track.",
    "safe": "Drive a safe lap. Prioritize staying on track and finishing cleanly.",
    "recover": "Recover from a wall or bad position. Regain control and rejoin the route.",
}


@dataclass(frozen=True)
class PolicyConfig:
    model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    history: int = 4
    chunk_size: int = 4
    # Legacy manifest field. The categorical policy does not sample a flow.
    flow_steps: int = 8
    head_width: int = 512
    lora_rank: int = 8
    lora_alpha: int = 16

    def __post_init__(self) -> None:
        if self.history < 2:
            raise ValueError("history must include at least two frames")
        if self.chunk_size < 1 or self.flow_steps < 1:
            raise ValueError("chunk_size and legacy flow_steps must be positive")
        if self.head_width < 1 or self.lora_rank < 1 or self.lora_alpha < 1:
            raise ValueError("head_width and LoRA settings must be positive")


def behavior_prompt(mode: str) -> str:
    try:
        instruction = BEHAVIOR_MODES[mode]
    except KeyError as exc:
        raise ValueError(f"unknown behavior mode {mode!r}; choose {tuple(BEHAVIOR_MODES)}") from exc
    return (
        "You control a Trackmania racing car from chronological front-camera frames. "
        f"{instruction} Predict the next steering, throttle, and brake action chunk."
    )
