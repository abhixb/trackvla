"""Unattended success-rate check: compare steering rules by solo finish rate.

Drive --laps solo laps per steering rule with the checkpoint and keep the
rule with the highest finish rate (the baseline rule's known result is passed
with --baseline; --baseline-extra-laps drives more baseline laps and merges).

Progress goes to logs/sr_plan.log and evaluations/sr_plan.json.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
RULES = "argmax:0,expected:0.2,rounded:0"


def log(message: str) -> None:
    line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with (PROJECT_ROOT / "logs" / "sr_plan.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def run(args: list[str], log_name: str) -> str:
    """Run a project script to completion; return its output."""
    path = PROJECT_ROOT / "logs" / log_name
    for attempt in range(1, 4):
        with path.open("w", encoding="utf-8") as stream:
            completed = subprocess.run([PYTHON, *args], cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT, env=os.environ)
        output = path.read_text(encoding="utf-8", errors="replace")
        if completed.returncode == 0:
            return output
        # Native crashes (e.g. an access violation while loading the model)
        # have been one-offs; retry before giving up.
        log(f"{args[0]} exited with {completed.returncode} (attempt {attempt}/3)")
        path.replace(path.with_suffix(f".failed{attempt}.log"))
    raise RuntimeError(f"{args[0]} failed 3 times; see logs/{log_name}")


def solo_laps(checkpoint: Path, rule: str, deadzone: float, laps: int, tag: str) -> dict:
    output = run([
        "scripts/drive_policy.py", str(checkpoint), "--settings", "raw", "--laps", str(laps),
        "--replan-ms", "50", "--pwm-ms", "50", "--steer-rule", rule, "--steer-deadzone", str(deadzone),
        "--output-root", str(PROJECT_ROOT / "datasets" / "closed_loop" / "sr_plan" / tag),
    ], f"sr_plan_{tag}.log")
    summary = re.findall(r"^summary: (.+\.json)$", output, re.MULTILINE)[-1]
    results = json.loads((PROJECT_ROOT / summary).read_text(encoding="utf-8"))["results"]
    finished = [r for r in results if r["finished"]]
    return {
        "checkpoint": str(checkpoint), "rule": rule, "deadzone": deadzone, "laps": len(results),
        "finished": len(finished), "rate": len(finished) / max(1, len(results)),
        "times": sorted(r["race_time_s"] for r in finished), "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--baseline", default="expected,0,5,13", help="rule,deadzone,finished,laps already measured")
    parser.add_argument("--laps", type=int, default=10)
    parser.add_argument("--skip-rules", action="store_true", help="use the baseline rule without step 1")
    parser.add_argument("--rules", default=RULES, help="step 1 rules to drive, rule:deadzone comma list")
    parser.add_argument("--baseline-checkpoint", type=Path, help="checkpoint the baseline result belongs to")
    parser.add_argument("--baseline-extra-laps", type=int, default=0, help="drive this many more baseline laps and merge")
    parser.add_argument("--status-name", default="sr_plan.json")
    args = parser.parse_args()

    status_path = PROJECT_ROOT / "evaluations" / args.status_name
    status = {"started_utc": datetime.now(timezone.utc).isoformat(), "rules": []}

    def save() -> None:
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    rule, deadzone, finished, laps = args.baseline.split(",")
    baseline_checkpoint = args.baseline_checkpoint or args.checkpoint
    best = {"checkpoint": str(baseline_checkpoint), "rule": rule, "deadzone": float(deadzone),
            "finished": int(finished), "laps": int(laps), "rate": int(finished) / int(laps), "times": []}
    if args.baseline_extra_laps:
        extra = solo_laps(baseline_checkpoint, rule, float(deadzone), args.baseline_extra_laps, f"baseline-extra-{rule}-dz{float(deadzone):g}")
        best["finished"] += extra["finished"]; best["laps"] += extra["laps"]
        best["rate"] = best["finished"] / best["laps"]; best["times"] = extra["times"]
        log(f"BASELINE extra {extra['finished']}/{extra['laps']}; combined {best['finished']}/{best['laps']}")
    status["baseline"] = dict(best)
    log(f"START {args.checkpoint}; baseline {rule} dz{deadzone}: {finished}/{laps}")

    if not args.skip_rules:
        for rule, deadzone in ((r, float(d)) for r, d in (item.split(":") for item in args.rules.split(","))):
            result = solo_laps(args.checkpoint, rule, deadzone, args.laps, f"rules-{rule}-dz{deadzone:g}")
            status["rules"].append(result); save()
            log(f"STEP1 rule {rule} dz{deadzone:g}: {result['finished']}/{result['laps']} finished {result['times']}")
            if result["rate"] > best["rate"]:
                best = result
        log(f"STEP1 chosen rule {best['rule']} dz{best['deadzone']:g} ({best['finished']}/{best['laps']})")
    status["best"] = best; save()

    log(f"DONE best {best['checkpoint']} rule {best['rule']} dz{best['deadzone']:g}: {best['finished']}/{best['laps']}")
    status["finished_utc"] = datetime.now(timezone.utc).isoformat(); save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
