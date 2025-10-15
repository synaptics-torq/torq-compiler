#!/usr/bin/env python3

import torq.performance
import argparse
from collections import defaultdict


def compute_durations(log: torq.performance.PerformanceLog):
    durations = defaultdict(lambda: defaultdict(list))

    for scenario in log.scenarios:
        for event in scenario.events:
            durations[scenario.name_as_tuple()][event.name].append(event.end_time_ms - event.start_time_ms)

    return durations


def main():
    parser  = argparse.ArgumentParser(description="Compares test performance logs from two CSV files")
    parser.add_argument("--tolerance", type=float, default=0.2, help="Relative tolerance for reporting differences")
    parser.add_argument("--markdown", action="store_true", default=False, help="Output results in markdown format")
    parser.add_argument("--all-changes", action="store_true", default=False, help="Report all changes, not just slow downs")
    parser.add_argument("reference", help="Path to the performance log CSV file")
    parser.add_argument("new", help="Path to the performance log CSV file")

    args = parser.parse_args()

    reference_performance = torq.performance.load_performance(args.reference)
    new_performance = torq.performance.load_performance(args.new)

    all_reference_durations = compute_durations(reference_performance)
    all_new_durations = compute_durations(new_performance)

    differences = defaultdict(lambda: defaultdict(list))

    for scenario_name, event_durations in all_reference_durations.items():
        new_event_durations = all_new_durations.get(scenario_name, {})

        for event_name, ref_durations in event_durations.items():
            new_durations = new_event_durations.get(event_name, [])

            if not new_durations:
                print(f"{scenario_name} Event: {event_name} - missing in new performance log")
                continue
            
            if len(new_durations) != len(ref_durations):
                print(f"{scenario_name} Event: {event_name} - different number of measurements (reference: {len(ref_durations)}, new: {len(new_durations)})")
                continue

            for i in range(len(ref_durations)):
                if new_durations[i] <= 0:
                    print(f"{scenario_name} Event: {event_name} - invalid new duration {new_durations[i]} ms")
                    continue

                if ref_durations[i] <= 0:
                    print(f"{scenario_name} Event: {event_name} - invalid reference duration {ref_durations[i]} ms")
                    continue

                changed = False
                if args.all_changes:
                    if abs(new_durations[i] - ref_durations[i]) / ref_durations[i] > args.tolerance:
                        changed = True
                else:
                    if new_durations[i] > ref_durations[i] * (1.0 + args.tolerance):
                        changed = True

                if changed:
                    differences[scenario_name][event_name].append((i, ref_durations[i], new_durations[i]))
    
    if not differences:
        print("No significant differences found")
        exit(0)

    if args.markdown:
        print("# Timing differences\n```\n")
    else:
        print("Timing differences:\n")

    for scenario_name, event_diffs in differences.items():        
        scenario_display = "::".join(scenario_name)
        print(f"Scenario: {scenario_display}")
        for event_name, diffs in event_diffs.items():
            print(f"  Event: {event_name}")
            for idx, ref_duration, new_duration in diffs:
                change = ((new_duration - ref_duration) / ref_duration) * 100.0
                print(f"    Measurement {idx}: {ref_duration:.2f} ms -> {new_duration:.2f} ms ({change:+.2f}%)")
            print("")

    if args.markdown:
        print("```\n")

    exit(1)


if __name__ == "__main__":
    main()