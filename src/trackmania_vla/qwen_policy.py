"""Qwen2.5-VL-3B driving policy (plan v5): bf16 LoRA backbone + categorical head.

``history`` frames and the telemetry/behavior prompt go through Qwen2.5-VL
together. Four learned action queries cross-attend over every fused image/text
token and feed factorized steer and throttle (coast/gas/brake) softmaxes, one
pair per 50 ms chunk position. Steer is the held key (3 classes, v1) or the
steering amount in ``steer_bins`` levels from full left to full right (v2).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import BEHAVIOR_MODES
from .replay_dataset import IGNORE, steer_bin_values, telemetry_text

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


@dataclass(frozen=True)
class QwenPolicyConfig:
    model_id: str = DEFAULT_MODEL_ID
    chunk: int = 4
    frame_stride: int = 2
    head_width: int = 512
    lora_rank: int = 16
    lora_alpha: int = 32
    # Upscale the 160x120 capture so the ViT's 28 px merged patches resolve
    # track edges: 320x240 -> ~108 visual tokens per frame.
    image_min_pixels: int = 320 * 240
    image_max_pixels: int = 320 * 240
    # v2: frames per sample, steering-amount labels and the telemetry turn hint.
    history: int = 2
    steer_bins: int = 3
    steer_window: int = 0
    turn_hint: bool = False
    # v3: racing-line geometry in the prompt, and the world-model head that
    # predicts the outcome of the current state (track_features).
    line_hint: bool = False
    aux: bool = False

    @classmethod
    def from_checkpoint(cls, directory: Path) -> "QwenPolicyConfig":
        manifest = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
        return cls(**manifest["config"])

    def dataset_kwargs(self) -> dict:
        return {
            "stride": self.frame_stride, "chunk": self.chunk, "history": self.history,
            "steer_window": self.steer_window, "steer_bins": self.steer_bins, "turn_hint": self.turn_hint,
            "line_hint": self.line_hint, "aux": self.aux,
        }


V2_CONFIG = QwenPolicyConfig(history=4, steer_bins=5, steer_window=5, turn_hint=True)
V3_CONFIG = QwenPolicyConfig(history=4, steer_bins=5, steer_window=5, turn_hint=True, line_hint=True, aux=True)


def driving_prompt(mode: str, telemetry: str, history: int = 2, frame_ms: int = 100) -> str:
    if history == 2:
        images = "The first image is 100 ms ago, the second is now."
    else:
        ages = ", ".join(f"{frame_ms * k} ms ago" for k in reversed(range(1, history)))
        images = f"The images are {ages}, and now."
    return (
        f"{BEHAVIOR_MODES[mode]} {images} "
        f"Telemetry: {telemetry} Predict the next four 50 ms steer and throttle actions."
    )


class ActionHead(nn.Module):
    def __init__(self, hidden: int, config: QwenPolicyConfig):
        super().__init__()
        width = config.head_width
        self.queries = nn.Parameter(torch.randn(config.chunk, width) * 0.02)
        self.key = nn.Linear(hidden, width)
        self.value = nn.Linear(hidden, width)
        self.attention = nn.MultiheadAttention(width, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.trunk = nn.Sequential(nn.Linear(width, width), nn.SiLU())
        self.steer = nn.Linear(width, config.steer_bins)
        self.throttle = nn.Linear(width, 3)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = tokens.float()
        queries = self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        attended, _ = self.attention(
            queries, self.key(tokens), self.value(tokens), key_padding_mask=~mask, need_weights=False
        )
        features = self.trunk(self.norm(queries + attended))
        return self.steer(features), self.throttle(features)


class AuxHead(nn.Module):
    """World model: one learned query over the fused tokens predicts the
    outcome of the current state (offsets, speed changes, crash logit)."""

    def __init__(self, hidden: int, config: QwenPolicyConfig):
        super().__init__()
        from .track_features import AUX_SIZE

        width = config.head_width
        self.query = nn.Parameter(torch.randn(1, width) * 0.02)
        self.key = nn.Linear(hidden, width)
        self.value = nn.Linear(hidden, width)
        self.attention = nn.MultiheadAttention(width, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, AUX_SIZE))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = tokens.float()
        query = self.query.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        attended, _ = self.attention(query, self.key(tokens), self.value(tokens), key_padding_mask=~mask, need_weights=False)
        return self.out(self.norm(query + attended))[:, 0]


def aux_loss(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Smooth L1 on the known regression targets + BCE on the crash flag (NaN = unknown)."""
    target = target.to(prediction.device, torch.float32)
    known = ~torch.isnan(target)
    zero = prediction.sum() * 0
    reg_known = known[:, :-1]
    regression = F.smooth_l1_loss(prediction[:, :-1][reg_known], target[:, :-1][reg_known]) if reg_known.any() else zero
    crash_known = known[:, -1]
    crash = (F.binary_cross_entropy_with_logits(prediction[:, -1][crash_known], target[:, -1][crash_known])
             if crash_known.any() else zero)
    return regression + crash, {"aux_regression": float(regression.detach()), "aux_crash": float(crash.detach())}


class QwenDrivingPolicy(nn.Module):
    def __init__(self, backbone: nn.Module, processor, config: QwenPolicyConfig):
        super().__init__()
        self.backbone = backbone
        self.processor = processor
        self.config = config
        self.head = ActionHead(backbone.config.text_config.hidden_size, config)
        self.aux_head = AuxHead(backbone.config.text_config.hidden_size, config) if config.aux else None
        self.register_buffer("steer_weight", torch.ones(config.steer_bins))
        self.register_buffer("throttle_weight", torch.ones(3))

    @classmethod
    def from_pretrained(cls, config: QwenPolicyConfig, device: str = "cuda") -> "QwenDrivingPolicy":
        from peft import LoraConfig
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            config.model_id,
            min_pixels=config.image_min_pixels,
            max_pixels=config.image_max_pixels,
        )
        processor.tokenizer.padding_side = "right"
        # Load the weights straight onto the GPU: staging the ~7.5 GB model in
        # CPU RAM first has hit the Windows commit limit (access violations).
        backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_id, dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map={"": device}, low_cpu_mem_usage=True,
        )
        backbone.requires_grad_(False)
        # Language-model attention projections only (the vision tower names
        # its projections qkv/proj, so these names do not match it).
        backbone.add_adapter(
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            )
        )
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        backbone.enable_input_require_grads()
        policy = cls(backbone, processor, config)
        return policy.to(device)

    def set_class_weights(self, steer: Sequence[float], throttle: Sequence[float]) -> None:
        self.steer_weight.copy_(torch.as_tensor(steer, dtype=torch.float32))
        self.throttle_weight.copy_(torch.as_tensor(throttle, dtype=torch.float32))

    def prompt(self, mode: str, telemetry: str) -> str:
        return driving_prompt(mode, telemetry, self.config.history, 50 * self.config.frame_stride)

    def encode(self, frame_groups, prompts: Sequence[str]) -> dict[str, torch.Tensor]:
        texts = [
            self.processor.apply_chat_template(
                [{"role": "user", "content": [*({"type": "image"} for _ in group), {"type": "text", "text": prompt}]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for group, prompt in zip(frame_groups, prompts)
        ]
        images = [image for group in frame_groups for image in group]
        return self.processor(text=texts, images=images, return_tensors="pt", padding=True)

    def logits(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head(*self.tokens(inputs))

    def tokens(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.head.queries.device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = self.backbone.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            use_cache=False,
            return_dict=True,
        )
        return outputs.last_hidden_state, inputs["attention_mask"].bool()

    def loss(self, inputs, steer: torch.Tensor, throttle: torch.Tensor, aux=None, aux_weight: float = 0.5) -> tuple[torch.Tensor, dict]:
        """Driving cross-entropy (IGNORE labels skipped) + ``aux_weight`` x world-model loss."""
        tokens, mask = self.tokens(inputs)
        steer_logits, throttle_logits = self.head(tokens, mask)
        steer = steer.to(steer_logits.device)
        throttle = throttle.to(throttle_logits.device)
        steer_loss = F.cross_entropy(steer_logits.flatten(0, 1), steer.flatten(), weight=self.steer_weight, ignore_index=IGNORE)
        throttle_loss = F.cross_entropy(throttle_logits.flatten(0, 1), throttle.flatten(), weight=self.throttle_weight, ignore_index=IGNORE)
        total = steer_loss + throttle_loss
        parts = {"steer_loss": float(steer_loss.detach()), "throttle_loss": float(throttle_loss.detach())}
        if self.aux_head is not None and aux is not None:
            world, world_parts = aux_loss(self.aux_head(tokens, mask), aux)
            total = total + aux_weight * world
            parts.update(world_parts)
        return total, parts

    @torch.no_grad()
    def predict(self, inputs) -> tuple[torch.Tensor, torch.Tensor]:
        steer_logits, throttle_logits = self.logits(inputs)
        return steer_logits.argmax(-1), throttle_logits.argmax(-1)

    def trainable_state(self) -> dict[str, torch.Tensor]:
        return {name: p.detach().cpu() for name, p in self.named_parameters() if p.requires_grad}

    def save(self, directory: Path, extra: dict) -> None:
        from safetensors.torch import save_file

        directory.mkdir(parents=True, exist_ok=False)
        state = self.trainable_state()
        state["steer_weight"] = self.steer_weight.cpu()
        state["throttle_weight"] = self.throttle_weight.cpu()
        save_file({k: v.contiguous() for k, v in state.items()}, str(directory / "trainable.safetensors"))
        (directory / "manifest.json").write_text(
            json.dumps({"format": "trackmania-qwen-policy-v1", "config": asdict(self.config), **extra}, indent=2),
            encoding="utf-8",
        )

    def load(self, directory: Path, *, backbone_only: bool = False, partial: bool = False) -> dict:
        """Load a checkpoint; ``backbone_only`` takes just the LoRA weights
        (e.g. a v1 checkpoint into a v2 policy with a fresh head); ``partial``
        takes everything a v2 checkpoint has into a v3 policy (fresh aux head)."""
        from safetensors.torch import load_file

        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != "trackmania-qwen-policy-v1":
            raise ValueError("checkpoint format does not match")
        if not backbone_only and not partial and QwenPolicyConfig(**manifest["config"]) != self.config:
            raise ValueError("checkpoint config does not match")
        state = load_file(str(directory / "trainable.safetensors"))
        steer_weight, throttle_weight = state.pop("steer_weight"), state.pop("throttle_weight")
        params = dict(self.named_parameters())
        if backbone_only:
            state = {name: tensor for name, tensor in state.items() if not name.startswith("head.")}
            params = {name: p for name, p in params.items() if not name.startswith("head.")}
        else:
            self.set_class_weights(steer_weight, throttle_weight)
        expected = {name: p for name, p in params.items() if not (partial and name.startswith("aux_head."))}
        missing = [name for name, p in expected.items() if p.requires_grad and name not in state]
        if missing or set(state) - set(params):
            raise ValueError("checkpoint parameter names do not match")
        with torch.no_grad():
            for name, tensor in state.items():
                params[name].copy_(tensor.to(params[name].device, params[name].dtype))
        return manifest


def load_policy(directory: Path) -> tuple[QwenDrivingPolicy, dict]:
    """Policy with the architecture recorded in the checkpoint, in eval mode."""
    policy = QwenDrivingPolicy.from_pretrained(QwenPolicyConfig.from_checkpoint(directory))
    manifest = policy.load(directory)
    policy.eval()
    return policy, manifest


def make_planner(policy: QwenDrivingPolicy, steer_bias=None, throttle_bias=None, mode: str = "fast"):
    """planner(*frames oldest first, telemetry) for PolicyDriver.

    v1 (3 steer classes) returns class ids per chunk position; v2 returns
    softmax probabilities so the driver can average overlapping chunks.
    """
    config = policy.config
    track = None
    if config.line_hint:
        from .track_features import default_track

        track = default_track()

    @torch.no_grad()
    def plan(*args):
        *frames, telemetry = args
        text = telemetry_text(telemetry, turn_hint=config.turn_hint, track=track)
        inputs = policy.encode([tuple(frames)], [policy.prompt(mode, text)])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            steer, throttle = policy.logits(inputs)
        steer = steer[0].float().cpu().numpy()
        throttle = throttle[0].float().cpu().numpy()
        if steer_bias is not None:
            steer = steer + steer_bias
        if throttle_bias is not None:
            throttle = throttle + throttle_bias
        if config.steer_bins == 3:
            return steer.argmax(-1), throttle.argmax(-1)
        softmax = lambda x: np.exp(x - x.max(-1, keepdims=True)) / np.exp(x - x.max(-1, keepdims=True)).sum(-1, keepdims=True)
        return softmax(steer), softmax(throttle)

    plan.steer_values = steer_bin_values(config.steer_bins) if config.steer_bins != 3 else None
    return plan


def make_blend_planner(policy: QwenDrivingPolicy, early: Path, switch_m: float, mode: str = "fast"):
    """Two checkpoints on one backbone: ``early`` drives until the car has
    reached ``switch_m`` metres along the racing line, ``policy`` after that.

    Both share the frozen Qwen weights and differ only in their LoRA and head
    weights (~14M values), which are swapped in place at the switch and back
    at the start of the next lap. Each uses its own prompt (line hint or not).
    """
    from safetensors.torch import load_file

    from .track_features import default_track

    params = {name: p for name, p in policy.named_parameters() if p.requires_grad}
    late_state = {name: p.detach().clone() for name, p in params.items()}
    early_config = QwenPolicyConfig.from_checkpoint(early)
    if (early_config.history, early_config.frame_stride, early_config.steer_bins) != (
        policy.config.history, policy.config.frame_stride, policy.config.steer_bins
    ):
        raise ValueError("blended checkpoints must share frames and steering levels")
    early_state = load_file(str(Path(early) / "trainable.safetensors"))
    early_state.pop("steer_weight"), early_state.pop("throttle_weight")
    if set(early_state) - set(params):
        raise ValueError("early checkpoint has parameters the policy lacks")
    early_state = {name: tensor.to(params[name].device, params[name].dtype) for name, tensor in early_state.items()}
    track = default_track()
    configs = {"early": early_config, "late": policy.config}
    state = {"active": None, "furthest": 0.0, "last_time": -1}

    def activate(which: str) -> None:
        if state["active"] == which:
            return
        source = early_state if which == "early" else late_state
        with torch.no_grad():
            for name, tensor in source.items():
                params[name].copy_(tensor)
        state["active"] = which

    @torch.no_grad()
    def plan(*args):
        *frames, telemetry = args
        now = int(telemetry["race_time_ms"])
        if now < state["last_time"]:
            state["furthest"] = 0.0  # a new lap
        state["last_time"] = now
        index, _, _ = track.locate(telemetry)
        state["furthest"] = max(state["furthest"], float(track.line.distance[index]))
        which = "late" if state["furthest"] >= switch_m else "early"
        activate(which)
        config = configs[which]
        text = telemetry_text(telemetry, turn_hint=config.turn_hint, track=track if config.line_hint else None)
        inputs = policy.encode([tuple(frames)], [policy.prompt(mode, text)])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            steer, throttle = policy.logits(inputs)
        softmax = lambda x: np.exp(x - x.max(-1, keepdims=True)) / np.exp(x - x.max(-1, keepdims=True)).sum(-1, keepdims=True)
        return softmax(steer[0].float().cpu().numpy()), softmax(throttle[0].float().cpu().numpy())

    plan.steer_values = steer_bin_values(policy.config.steer_bins)
    activate("early")
    return plan
