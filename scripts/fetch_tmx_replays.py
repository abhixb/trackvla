"""Download A01-Race replays from TMNF Exchange and keep the keyboard ones.

Writes ``<out>/<replay_id>.Replay.Gbx`` plus ``<out>/index.json`` recording,
for every downloaded replay, whether it is a usable keyboard replay and why
not otherwise. Re-running resumes: existing files are not downloaded again.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trackmania_vla.replay_inputs import ReplayRejected, load_keyboard_replay

API = "https://tmnf.exchange/api/replays"
DOWNLOAD = "https://tmnf.exchange/recordgbx/{replay_id}"
USER_AGENT = "trackmania-vla-dataset (research; contact via project owner)"


def _get(url: str, timeout: float = 30.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def list_replays(track_id: int) -> list[dict]:
    replays: list[dict] = []
    after: int | None = None
    while True:
        query = {
            "trackId": track_id,
            "fields": "ReplayId,ReplayTime,User.Name",
            "count": 1000,
        }
        if after is not None:
            query["after"] = after
        page = json.loads(_get(f"{API}?{urllib.parse.urlencode(query)}"))
        results = page.get("Results", [])
        replays.extend(results)
        if not page.get("More") or not results:
            return replays
        after = results[-1]["ReplayId"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-id", type=int, default=2233, help="TMX id of the official A01-Race")
    parser.add_argument("--keyboard-count", type=int, default=300, help="stop after this many usable replays")
    parser.add_argument("--max-time-ms", type=int, default=60_000)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "datasets" / "replays" / "tmx_a01")
    parser.add_argument("--delay-s", type=float, default=0.5, help="politeness delay between downloads")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "index.json"
    index: dict[str, dict] = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}

    listing = [r for r in list_replays(args.track_id) if 0 < r["ReplayTime"] <= args.max_time_ms]
    # Spread selections across the whole time range instead of only the top.
    listing.sort(key=lambda r: r["ReplayTime"])
    stride = max(1, len(listing) // max(1, args.keyboard_count * 2))
    ordered = listing[::stride] + [r for i, r in enumerate(listing) if i % stride]
    print(f"TMX lists {len(listing)} replays for track {args.track_id}.", flush=True)

    usable = sum(entry["usable"] for entry in index.values())
    players = {entry["player"] for entry in index.values() if entry["usable"]}
    for replay in ordered:
        if usable >= args.keyboard_count:
            break
        replay_id = str(replay["ReplayId"])
        player = replay.get("User", {}).get("Name", "")
        if replay_id in index or player in players:
            continue  # one replay per player keeps the data diverse
        path = args.out / f"{replay_id}.Replay.Gbx"
        if not path.exists():
            data = _get(DOWNLOAD.format(replay_id=replay_id))
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(data)
            os.replace(temporary, path)
            time.sleep(args.delay_s)
        entry = {"player": player, "tmx_time_ms": replay["ReplayTime"], "usable": False, "reason": None}
        try:
            parsed = load_keyboard_replay(path, replay_id=replay_id, player=player)
            if parsed.race_time_ms != replay["ReplayTime"]:
                raise ReplayRejected("ghost time differs from TMX listing")
            entry["usable"] = True
            usable += 1
            players.add(player)
        except ReplayRejected as exc:
            entry["reason"] = str(exc)
        except Exception as exc:  # corrupt or unsupported Gbx
            entry["reason"] = f"parse_error: {type(exc).__name__}: {exc}"
        index[replay_id] = entry
        temporary = index_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(index, indent=1), encoding="utf-8")
        os.replace(temporary, index_path)
        status = "keyboard" if entry["usable"] else f"skip ({entry['reason']})"
        print(f"[{usable}/{args.keyboard_count}] {replay_id} {player[:20]:20s} {replay['ReplayTime'] / 1000:6.2f}s {status}", flush=True)

    print(f"Usable keyboard replays: {usable} in {args.out}")
    return 0 if usable else 1


if __name__ == "__main__":
    raise SystemExit(main())
