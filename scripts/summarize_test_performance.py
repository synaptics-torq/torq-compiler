#!/usr/bin/env python3

import torq.performance
import argparse
from collections import defaultdict


def main():
    parser  = argparse.ArgumentParser(description="Summarize test performance logs from a CSV file")
    parser.add_argument("input", help="Path to the performance log CSV file")
    parser.add_argument("--top", type=int, default=10, help="Number of top entries to show")

    args = parser.parse_args()

    performance = torq.performance.load_performance(args.input)

    def split_namespace(scenario_name):
        parts = scenario_name.split("::")
        if len(parts) == 1:
            return (tuple(), scenario_name)
        return (tuple(parts[:-1]), parts[-1])

    # Group events by full namespace path and event name
    events_by_namespace = defaultdict(lambda: defaultdict(list))

    for scenario in performance.scenarios:
        for event in scenario.events:
            scenario_namespaces, scenario_name = split_namespace(scenario.name)            
            events_by_namespace[scenario_namespaces][event.name].append(
                (scenario_name, event.end_time_ms - event.start_time_ms)
            )

    def format_namespace(ns_tuple):
        return "::".join(ns_tuple) if ns_tuple else "<global>"

    for ns_tuple, events in events_by_namespace.items():
        ns_display = f"Namespace: {format_namespace(ns_tuple)}"
        print(ns_display)
        for event_name, entries in events.items():
            entries.sort(key=lambda x: x[1], reverse=True)
            print(f"  Event: {event_name}")
            for scenario_name, duration in entries[:args.top]:
                print(f"    {duration:.2f} ms - {scenario_name}")
            print("")


if __name__ == "__main__":
    main()