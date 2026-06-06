# Copyright 2025-2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Executor discovery state management.

Contains the ``ExecutorDiscoveryState`` class and the global singleton
``_discovery_state`` used to accumulate results during a pytest discovery
session.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from torq.gen_config.core import (
    DEFAULT_TOLERANCE,
    EXECUTOR_ORDER,
    TIMING_PRECISION,
    _discovery_log,
)


class ExecutorDiscoveryState:
    """Global state for executor discovery results."""

    def __init__(self) -> None:
        self.results: Dict[str, Dict[str, Dict]] = {}  # layer_id -> executor -> result
        self.locations: Dict[str, str] = {}  # layer_id -> mlir_location (op_type for layer tests)
        self.node_indices: Dict[str, int] = {}  # layer_id -> node_index in source graph
        self.mlir_files: Dict[str, Path] = {}  # layer_id -> mlir file path
        # layer_id -> line:column from full model MLIR
        self.full_mlir_locations: Dict[str, str] = {}
        # Full model comparison metrics (populated when running full model test)
        self.full_model_metrics: Optional[Dict[str, Any]] = None
        # layer_id -> recommended_executor (loaded from JSON to preserve user edits)
        self.recommended_executors: Dict[str, str] = {}

    def record_result(
        self,
        layer_id: str,
        executor: str,
        status: str,
        tolerance_used: Dict[str, float],
        max_diff: Optional[Dict] = None,
        failure_report: Optional[Dict] = None,
        timing: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record test result for a layer/executor."""
        if layer_id not in self.results:
            self.results[layer_id] = {}

        result: Dict[str, Any] = {"status": status, "tolerance_used": tolerance_used}
        if max_diff:
            result["max_diff"] = max_diff
        if failure_report:
            result["failure_report"] = failure_report
        if timing:
            result["timing"] = timing

        self.results[layer_id][executor] = result

    def record_metadata(
        self,
        layer_id: str,
        node_index: Optional[int] = None,
        mlir_location: Optional[str] = None,
        full_mlir_location: Optional[str] = None,
        mlir_file: Optional[Path] = None,
    ) -> None:
        """Record metadata for a layer."""
        if node_index is not None:
            self.node_indices[layer_id] = node_index
        if mlir_location:
            self.locations[layer_id] = mlir_location
        if full_mlir_location:
            self.full_mlir_locations[layer_id] = full_mlir_location
        if mlir_file:
            self.mlir_files[layer_id] = mlir_file

    def load_from_json(self, json_data: Dict[str, Any]) -> None:
        """Load existing results from JSON data (for skip mode consistency)."""
        ops = json_data.get("ops", {})
        loaded_count = 0
        for layer_id, op_data in ops.items():
            executors = op_data.get("executors", {})
            for executor, result in executors.items():
                # Only load if not already recorded
                if layer_id not in self.results:
                    self.results[layer_id] = {}
                if executor not in self.results[layer_id]:
                    self.results[layer_id][executor] = result
                    loaded_count += 1

            # Load metadata
            node_index = op_data.get("_node_index")
            if node_index is not None and layer_id not in self.node_indices:
                self.node_indices[layer_id] = node_index
            mlir_loc = op_data.get("mlir_location")
            if mlir_loc and layer_id not in self.locations:
                self.locations[layer_id] = mlir_loc
            recommended = op_data.get("recommended_executor")
            if recommended is not None:
                self.recommended_executors[layer_id] = recommended

        if loaded_count > 0:
            _discovery_log(f"Loaded {loaded_count} cached results from existing JSON")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        status_counts = {"success": 0, "difference": 0, "error": 0}
        executor_counts = {executor: 0 for executor in EXECUTOR_ORDER}

        # Timing statistics
        timing_available = False
        executor_timing = {executor: [] for executor in EXECUTOR_ORDER}

        for executors in self.results.values():
            for executor, result in executors.items():
                status = result.get("status", "error")
                status_counts[status] = status_counts.get(status, 0) + 1
                if status == "success" and executor in executor_counts:
                    executor_counts[executor] += 1

                # Collect timing data
                timing = result.get("timing")
                if timing and "runtime_ms" in timing:
                    timing_available = True
                    executor_timing[executor].append(timing["runtime_ms"])

        summary = {
            "total_layers": len(self.results),
            "status_counts": status_counts,
            "executor_counts": executor_counts,
        }

        # Add timing summary if available
        if timing_available:
            timing_summary = {}
            for executor in EXECUTOR_ORDER:
                times = executor_timing[executor]
                if times:
                    timing_summary[executor] = {
                        "avg_ms": round(sum(times) / len(times), TIMING_PRECISION),
                        "min_ms": round(min(times), TIMING_PRECISION),
                        "max_ms": round(max(times), TIMING_PRECISION),
                        "samples": len(times),
                    }
            if timing_summary:
                summary["timing_summary"] = timing_summary

        return summary


# Global state
_discovery_state = ExecutorDiscoveryState()
