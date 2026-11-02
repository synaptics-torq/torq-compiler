#!/usr/bin/env python3

"""
Prints a Markdown summary of the repeated board runs (--torq-runtime-runs > 1): for each test,
the warm-up run, then the minimum, median and standard deviation over the other runs, read from
the *_runs.json files in a runtime profiling output directory. With --step-timings, also the time
spent per step (see --torq-step-timings).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

METRICS = [("total_duration", "NPU"), ("e2e_inference_time", "e2e")]


def runs_table(profiling_dir):
    lines = []
    files = sorted(Path(profiling_dir).glob("*_runs.json"))
    if not files:
        return ["No *_runs.json file found."]
    header = "| Test | Runs |" + "".join(f" {label} run 1 | {label} min | {label} median | {label} σ |" for _, label in METRICS)
    lines += [header, "|---|--:|" + "--:|" * 4 * len(METRICS)]
    for path in files:
        data = json.load(open(path))
        row = f"| {path.name[:-len('_runs.json')]} | {len(data['runs'])} |"
        for key, _ in METRICS:
            values = [data["warmup"].get(key), data.get("min", {}).get(key), data["median"].get(key)]
            row += "".join(f" {v / 1e6:.1f} |" if v is not None else " - |" for v in values)
            stdev, median = data["stdev"].get(key), data["median"].get(key)
            row += f" {stdev / 1e6:.1f} ({100 * stdev / median:.0f}%) |" if stdev is not None and median else " - |"
        lines.append(row)
    return ["Times in ms. Run 1 is the warm-up run; min, median and σ are over the other runs.", ""] + lines


def step_table(step_timings):
    results = json.load(open(step_timings))
    totals = defaultdict(float)
    for test in results.values():
        for name, seconds in test["steps"]:
            # group the per-run steps
            totals[" ".join(w if not w.isdigit() else "N" for w in name.split())] += seconds
    lines = ["Time per step, summed over all tests and xdist workers (s):", "",
             "| Step | Time (s) |", "|---|--:|"]
    for when in ("setup", "call", "teardown"):
        lines.append(f"| test {when} | {sum(t.get(when, 0) for t in results.values()):.1f} |")
    lines += [f"| {name} | {seconds:.1f} |" for name, seconds in sorted(totals.items(), key=lambda kv: -kv[1])]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("profiling_dir", help="Directory passed to --torq-runtime-profiling-output-dir")
    parser.add_argument("--step-timings", help="JSON file written with --torq-step-timings")
    args = parser.parse_args()

    lines = ["## Board runs", ""] + runs_table(args.profiling_dir)
    if args.step_timings and Path(args.step_timings).exists():
        lines += ["", "## Where the time goes", ""] + step_table(args.step_timings)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
