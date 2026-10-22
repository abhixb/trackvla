"""Draw the system architecture as a paper-style figure.

  python scripts/make_architecture_diagram.py      -> docs/architecture.svg (+ .png via Edge)

Module names only; real game frames from a reference expert lap and the
racing line itself carry the rest.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trackmania_vla.track_features import JUMPS_M, default_racing_line  # noqa: E402

OUT = PROJECT_ROOT / "docs" / "architecture.svg"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
W, H = 1600, 900
FONT = "Helvetica, Arial, sans-serif"

NAVY, NAVY_LIGHT, TOKEN = "#23476b", "#dfe9f3", "#3d6a96"
BLUE, BLUE_FILL = "#4a90c2", "#d6ebf8"
ORANGE, ORANGE_FILL, ORANGE_TEXT = "#e07b22", "#fff4e8", "#c55a11"
GREEN, GREEN_FILL, GREEN_TEXT = "#4c9a5b", "#eef7ef", "#2f6b3a"
GRAY, GRAY_FILL = "#9a9a9a", "#e6e6e6"
INK, MUTED = "#1d2733", "#6b7480"
STEER = ["#c0392b", "#e6a39b", "#c4c4c4", "#9cc3e6", "#2e75b6"]

parts: list[str] = []


def add(s: str) -> None:
    parts.append(s)


def box(x, y, w, h, fill, stroke, rx=14, sw=2.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=18, weight="bold", fill=INK, anchor="middle", italic=False):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    style = ' font-style="italic"' if italic else ""
    add(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}"{style}>{s}</text>')


def arrow(points, color=NAVY, sw=2.4):
    d = " ".join(f"{'M' if i == 0 else 'L'}{px},{py}" for i, (px, py) in enumerate(points))
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" marker-end="url(#arr)"/>')


def image(x, y, w, h, array, tag=None):
    buffer = io.BytesIO()
    Image.fromarray(array).resize((w * 2, h * 2), Image.LANCZOS).save(buffer, "PNG")
    data = base64.b64encode(buffer.getvalue()).decode()
    add(f'<clipPath id="c{x}_{y}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8"/></clipPath>')
    add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" clip-path="url(#c{x}_{y})" href="data:image/png;base64,{data}"/>')
    box(x, y, w, h, "none", GRAY, rx=8, sw=1.5)
    if tag:
        main, _, sub = tag.partition("_")
        tw = 14 + 11 * len(main) + 8 * len(sub)
        box(x, y, tw, 26, INK, "none", rx=7, sw=0)
        add(f'<text x="{x + 7}" y="{y + 19}" font-family="Georgia, serif" font-size="17" font-style="italic" '
            f'font-weight="bold" fill="white">{main}<tspan font-size="12" dy="4">{sub}</tspan></text>')


def pill(x, y, w, h, pattern, stroke, label, color):
    box(x, y, w, h, f"url(#{pattern})", stroke, rx=h / 2)
    tw = 9.5 * len(label) + 20
    box(x + (w - tw) / 2, y + 6, tw, h - 12, "white", "none", rx=(h - 12) / 2, sw=0)
    text(x + w / 2, y + h / 2 + 6, label, size=16, fill=color)


def chips(x, y, colors, size=26, gap=5):
    for i, c in enumerate(colors):
        box(x + i * (size + gap), y, size, size, c, "none", rx=5, sw=0)


def stage(y, h, title):
    box(20, y, W - 40, h, "#fbfbfb", "#8c8c8c", rx=18, sw=2, dash="9 7")
    text(44, y + 42, title, size=32, anchor="start")


def token_bar(x, y, w, h, label, sub):
    box(x, y, w, h, NAVY, NAVY, rx=16)
    n = int((w - 30) // 46)
    for row_y in (y + 9, y + h - 23):
        for k in range(n):
            box(x + 18 + k * 46, row_y, 34, 14, TOKEN, "#7fa6cc", rx=7, sw=1.2)
    box(x + 40, y + 32, w - 80, h - 64, NAVY_LIGHT, NAVY_LIGHT, rx=10)
    text(x + w / 2, y + h / 2 + 8, label, size=26)
    text(x + w - 60, y + h / 2 + 7, sub, size=15, fill=MUTED, anchor="end", weight="normal")


def track_map(x, y, size, line):
    xs, zs = line.points[:, 0], line.points[:, 2]
    s = size / max(np.ptp(xs), np.ptp(zs))
    px, pz = x + (xs - xs.min()) * s, y + (zs - zs.min()) * s
    path = " ".join(f"{'M' if i == 0 else 'L'}{a:.1f},{b:.1f}" for i, (a, b) in enumerate(zip(px[::4], pz[::4])))
    add(f'<path d="{path}" fill="none" stroke="#c9c9c9" stroke-width="11" stroke-linecap="round" stroke-linejoin="round"/>')
    add(f'<path d="{path}" fill="none" stroke="{NAVY}" stroke-width="2.6" stroke-linejoin="round"/>')
    for n, jump in enumerate(JUMPS_M, 1):
        i = int(np.searchsorted(line.distance, jump))
        add(f'<circle cx="{px[i]:.1f}" cy="{pz[i]:.1f}" r="7" fill="{ORANGE}"/>')
        text(px[i], pz[i] - 13, f"J{n}", size=14, fill=ORANGE_TEXT)


def main() -> int:
    episode = sorted((PROJECT_ROOT / "datasets" / "reference_laps" / "episodes").iterdir())[0]
    frames = np.memmap(episode / "frames.rgb", dtype=np.uint8, mode="r").reshape(-1, 120, 160, 3)
    frame = lambda i: np.asarray(frames[i])  # noqa: E731
    line = default_racing_line()

    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    add('<defs>'
        '<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{NAVY}"/></marker>'
        '<pattern id="hatchG" patternUnits="userSpaceOnUse" width="9" height="9" patternTransform="rotate(45)">'
        f'<rect width="9" height="9" fill="{GREEN_FILL}"/><line x1="0" y1="0" x2="0" y2="9" stroke="{GREEN}" stroke-width="3.5"/></pattern>'
        '<pattern id="hatchO" patternUnits="userSpaceOnUse" width="9" height="9" patternTransform="rotate(45)">'
        f'<rect width="9" height="9" fill="{ORANGE_FILL}"/><line x1="0" y1="0" x2="0" y2="9" stroke="{ORANGE}" stroke-width="3.5"/></pattern>'
        '</defs>')
    add(f'<rect width="{W}" height="{H}" fill="white"/>')

    # ---------------- Stage 1: data ----------------
    stage(20, 300, "Stage 1")
    image(60, 75, 120, 90, frame(40), "o_1")
    image(60, 185, 120, 90, frame(300), "o_T")
    arrow([(180, 175), (230, 175)])
    box(230, 115, 190, 120, GRAY_FILL, GRAY)
    text(325, 168, "Replay", size=22)
    text(325, 196, "Playback", size=22)
    arrow([(420, 175), (470, 175)])
    chips(475, 132, STEER)
    chips(506, 190, ["#d9d9d9", "#5aa469", "#c0392b"])
    text(552, 250, "Labels", size=16, fill=MUTED, weight="normal")
    arrow([(640, 175), (690, 175)])
    box(690, 85, 240, 180, "white", NAVY, sw=2.5)
    text(810, 120, "Training Mix", size=22)
    for k, (name, share, color) in enumerate((("Expert", .62, "#2e75b6"), ("DAgger", .25, GREEN), ("Watched", .13, ORANGE))):
        width = 200 * share / .62
        box(710, 140 + k * 40, width, 30, color, "none", rx=6, sw=0)
        text(718 + width if share < .3 else 720, 161 + k * 40, name, size=15,
             fill=INK if share < .3 else "white", anchor="start")
    box(975, 95, 170, 70, GREEN_FILL, GREEN)
    text(1060, 138, "DAgger", size=20, fill=GREEN_TEXT)
    box(975, 185, 170, 70, ORANGE_FILL, ORANGE)
    text(1060, 228, "Watched Laps", size=20, fill=ORANGE_TEXT)
    arrow([(975, 130), (930, 130)])
    arrow([(975, 220), (930, 220)])
    box(1190, 85, 360, 180, "white", GRAY)
    track_map(1215, 108, 150, line)
    text(1460, 170, "Expert", size=22)
    text(1460, 198, "Envelope", size=22)

    # ---------------- Stage 2: policy + driving ----------------
    stage(340, 540, "Stage 2")
    # inputs
    for k, index in enumerate((114, 116, 118, 120)):
        image(60 + k * 92, 770, 84, 63, frame(index), ("o_t-3", "o_t-2", "o_t-1", "o_t")[k])
    box(470, 772, 330, 60, GREEN_FILL, GREEN, rx=30)
    text(635, 809, "telemetry + racing line", size=18, fill=GREEN_TEXT, italic=True)
    box(60, 680, 360, 54, BLUE_FILL, BLUE)
    text(225, 715, "ViT", size=22)
    text(395, 718, "\u2744", size=26, fill=BLUE)
    arrow([(240, 770), (240, 734)])
    arrow([(240, 680), (240, 640)])
    arrow([(635, 772), (635, 640)])
    token_bar(45, 520, 1010, 120, "Qwen2.5-VL-3B", "LoRA")

    # heads
    arrow([(1055, 580), (1085, 580), (1085, 470), (1110, 470)])
    arrow([(1085, 580), (1085, 700), (1110, 700)])
    box(1110, 395, 260, 150, GREEN_FILL, GREEN, sw=2.5)
    text(1240, 428, "Action Head", size=22, fill=GREEN_TEXT)
    pill(1130, 445, 220, 36, "hatchG", GREEN, "Action Query", GREEN_TEXT)
    box(1130, 492, 220, 36, "white", "#9ccaa5", rx=7, sw=1.5)
    text(1240, 516, "Cross-Attn", size=16, weight="normal")
    box(1110, 625, 260, 150, ORANGE_FILL, ORANGE, sw=2.5)
    text(1240, 658, "World Model", size=22, fill=ORANGE_TEXT)
    pill(1130, 675, 220, 36, "hatchO", ORANGE, "World Query", ORANGE_TEXT)
    box(1130, 722, 220, 36, "white", "#f0b27a", rx=7, sw=1.5)
    text(1240, 746, "Cross-Attn", size=16, weight="normal")

    # action outputs -> keys
    arrow([(1370, 470), (1400, 470)])
    probabilities = [[.05, .15, .6, .15, .05], [.02, .1, .55, .28, .05], [.01, .05, .4, .4, .14], [0, .03, .27, .45, .25]]
    for step, probs in enumerate(probabilities):
        for b, p in enumerate(probs):
            box(1405 + step * 30, 418 + b * 21, 4 + 20 * p / .6, 16, STEER[b], "none", rx=3, sw=0)
    text(1465, 400, "Actions", size=18, fill=GREEN_TEXT)
    # world outputs
    arrow([(1370, 712), (1400, 712)])
    text(1410, 690, "offset", size=16, fill=ORANGE_TEXT, anchor="start", weight="normal")
    text(1410, 717, "speed", size=16, fill=ORANGE_TEXT, anchor="start", weight="normal")
    text(1410, 744, "crash?", size=16, fill=ORANGE_TEXT, anchor="start", weight="normal")
    # closed loop: ensemble + sigma-delta -> keys -> game
    add(f'<circle cx="1465" cy="562" r="22" fill="white" stroke="{NAVY}" stroke-width="2.4"/>')
    text(1465, 570, "\u03a3\u0394", size=18)
    arrow([(1465, 522), (1465, 540)])
    for k, key in enumerate(("\u2190", "\u2192", "\u2191")):
        box(1408 + k * 40, 600, 34, 34, "white", INK, rx=6, sw=1.8)
        text(1425 + k * 40, 624, key, size=19)
    arrow([(1465, 584), (1465, 600)])
    add(f'<path d="M1520,617 L1550,617 L1550,858 L230,858 L230,846" fill="none" stroke="{NAVY}" stroke-width="2.4" '
        f'stroke-dasharray="7 6" marker-end="url(#arr)"/>')
    text(890, 851, "TMNF  ·  every 50 ms", size=15, fill=MUTED, weight="normal")

    add("</svg>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT}")
    if EDGE.exists():
        png = OUT.with_suffix(".png")
        subprocess.run([str(EDGE), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                        f"--window-size={W},{H}", f"--screenshot={png}", OUT.as_uri()], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"wrote {png}" if png.exists() else "PNG render failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
