"""Render a recorded episode (frames.rgb + steps.jsonl) as an annotated GIF.

Bottom bar: race time, speed, the held steer/pedal, and WALL on wall contact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def render(episode: Path, output: Path, scale: int = 3, every: int = 1) -> Path:
    manifest = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
    width, height = manifest["capture"]["width"], manifest["capture"]["height"]
    size = width * height * 3
    raw = (episode / "frames.rgb").read_bytes()
    steps = [json.loads(line) for line in (episode / "steps.jsonl").open(encoding="utf-8")]
    try:
        font = ImageFont.truetype("arial.ttf", 6 * scale)
    except OSError:
        font = ImageFont.load_default()
    frames = []
    for i, step in enumerate(steps):
        if i % every:
            continue
        image = Image.frombytes("RGB", (width, height), raw[i * size : (i + 1) * size])
        image = image.resize((width * scale, height * scale), Image.BILINEAR)
        draw = ImageDraw.Draw(image)
        controls, telemetry = step["controls"], step["telemetry"]
        steer = "LEFT" if controls["left"] else "RIGHT" if controls["right"] else "straight"
        pedal = "GAS" if controls["accelerate"] else "BRAKE" if controls["brake"] else "coast"
        bar = height * scale - 20 * scale
        draw.rectangle((0, bar, width * scale, height * scale), fill=(0, 0, 0))
        draw.text((3 * scale, bar + 2 * scale), f"t={step['race_time_ms'] / 1000:5.2f}s  {telemetry['display_speed_kmh']:3d} km/h", fill=(255, 255, 255), font=font)
        colour = (255, 220, 0) if steer != "straight" else (200, 200, 200)
        draw.text((3 * scale, bar + 11 * scale), f"{steer:8s} {pedal}" + ("   WALL" if telemetry["lateral_contact"] else ""), fill=colour, font=font)
        if "expert_driving" in step:  # DAgger laps: who is driving this frame
            expert = step["expert_driving"]
            label, fill = ("EXPERT", (230, 40, 40)) if expert else ("MODEL", (40, 200, 60))
            draw.rectangle((width * scale - 42 * scale, 2 * scale, width * scale - 2 * scale, 12 * scale), fill=fill)
            draw.text((width * scale - 40 * scale, 3 * scale), label, fill=(255, 255, 255), font=font)
        frames.append(image.convert("P", palette=Image.ADAPTIVE, colors=128))
    period = manifest["capture"].get("period_ms", 50)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=period * every, loop=0, optimize=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--every", type=int, default=1, help="keep every Nth frame (2 = 10 fps)")
    args = parser.parse_args()
    path = render(args.episode, args.output, args.scale, args.every)
    print(f"{path} ({path.stat().st_size / 2**20:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
