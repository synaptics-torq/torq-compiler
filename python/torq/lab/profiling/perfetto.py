"""
Perfetto Trace Generator for Profile Analysis

Converts compile-time profiling CSV data into Perfetto trace format.
Includes hardware utilization analysis (DMA, SLICE, IDLE time) with visual overview tracks.

Usage:
    python -m torq.lab.profiling.perfetto profile1.csv --pb output.pb
"""

try:
    from . import perfetto_pb2 as perfetto
except ImportError:
    import perfetto_pb2 as perfetto

import argparse
import bisect
from collections import OrderedDict
import os
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from .debug_info import DebugInfo, DispatchDebugInfo, HostCopyWorkUnitDebugInfo, HostProgramWorkUnitDebugInfo, HalWorkUnitDebugInfo, NssProgramWorkUnitDebugInfo, NssCfgWorkUnitDebugInfo, \
                                DmaInWorkUnitDebugInfo, DmaOutWorkUnitDebugInfo, CdmaWorkUnitDebugInfo, SliceProgramWorkUnitDebugInfo, CssProgramWorkUnitDebugInfo

import torq.lab.profiling.debug_info as torq_debug_info


# =============================================================================
# CONFIGURATION
# =============================================================================

# Toggle to show shortened MLIR locations (filename only) vs full paths
SHOW_SHORT_LOCATIONS = True

def format_time_duration(ns):
    """
    Format nanosecond duration into human-readable string with appropriate unit.
    
    Args:
        ns: Duration in nanoseconds (integer)
        
    Returns:
        Formatted string like "1.234s", "456.789ms", "123.456µs", or "123ns"
    """
    if ns >= 1_000_000_000:  # >= 1 second
        return f"{ns / 1_000_000_000:.3f}s"
    elif ns >= 1_000_000:  # >= 1 millisecond
        return f"{ns / 1_000_000:.3f}ms"
    elif ns >= 1_000:  # >= 1 microsecond
        return f"{ns / 1_000:.3f}µs"
    else:
        return f"{ns}ns"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PerfettoEvent:
    """Represents a single trace event with timing information."""
        
    tuuid: str
    function: str
    start_time: int
    end_time: int
    category: Optional[str] = None
    annotations: Optional[dict] = None
    # Flow arrows: the viewer links this slice to the next slice with the same flow id
    flow_ids: list = field(default_factory=list)
    terminating_flow_ids: list = field(default_factory=list)


class PerfettoTraceWriter:
    """Manages Perfetto trace creation with processes and threads."""
    
    def __init__(self, filename, timestamp_origin_ns=0):
        self.file = open(filename, "wb")
        self.trace = perfetto.Trace()
        self.global_uuid = 0
        self.timestamp_origin_ns = timestamp_origin_ns

    def close(self):
        if not self.file.closed:
            self.flush()
            self.file.close()

    def __del__(self):
        self.close()
    
    def flush(self):
        """Write accumulated trace packets to file and clear buffer."""
        if not self.file.closed:
            self.file.write(self.trace.SerializeToString())
            self.trace.Clear()

    def get_next_uuid(self):
        """Generate unique ID for processes and threads."""
        self.global_uuid += 1
        return self.global_uuid

    def add_process_descriptor(self, process_name,
                               child_ordering=perfetto.TrackDescriptor.ChildTracksOrdering.CHRONOLOGICAL):
        """Register a new process in the trace."""
        
        puuid = self.get_next_uuid()        
        
        packet = self.trace.packet.add()
        track_des = perfetto.TrackDescriptor()        
        track_des.uuid = puuid
        track_des.name = process_name
        track_des.child_ordering = child_ordering
        packet.track_descriptor.CopyFrom(track_des)

        return puuid

    def add_thread_descriptor(self, puuid, thread_name, order_rank=None, child_ordering=None):
        """
        Register a new thread within a process, or a child track within another track.
        order_rank is used when the parent has EXPLICIT ordering. child_ordering is set when
        this track has child tracks.
        """

        tuuid = self.get_next_uuid()
        
        packet = self.trace.packet.add()
        track_des = perfetto.TrackDescriptor()
        track_des.name = thread_name        
        track_des.uuid = tuuid
        track_des.parent_uuid = puuid
        if order_rank is not None:
            track_des.sibling_order_rank = order_rank
        if child_ordering is not None:
            track_des.child_ordering = child_ordering
        packet.track_descriptor.CopyFrom(track_des)

        return tuuid

    def add_counter_descriptor(self, puuid, counter_name, order_rank=None):
        """Register a counter track (values in bytes) within a process."""

        tuuid = self.get_next_uuid()

        packet = self.trace.packet.add()
        track_des = perfetto.TrackDescriptor()
        track_des.name = counter_name
        track_des.uuid = tuuid
        track_des.parent_uuid = puuid
        track_des.counter.unit = perfetto.CounterDescriptor.Unit.UNIT_SIZE_BYTES
        if order_rank is not None:
            track_des.sibling_order_rank = order_rank
        packet.track_descriptor.CopyFrom(track_des)

        return tuuid

    def add_counter_value(self, tuuid, time, value):
        """Set the value of a counter track from the given time on."""

        packet = self.trace.packet.add()
        packet.timestamp = int(time - self.timestamp_origin_ns)
        packet.track_event.type = perfetto.TrackEvent.Type.TYPE_COUNTER
        packet.track_event.track_uuid = tuuid
        packet.track_event.counter_value = int(value)
        packet.trusted_packet_sequence_id = 1

        if len(self.trace.packet) > 1024:
            self.flush()

    def add_event(self, event):
        """Add a trace event (slice) to the timeline."""

        self.set_track_event(event.start_time, perfetto.TrackEvent.Type.TYPE_SLICE_BEGIN, event)
        self.set_track_event(event.end_time, perfetto.TrackEvent.Type.TYPE_SLICE_END, event)

        # Periodically flush to avoid memory buildup
        if len(self.trace.packet) > 1024:
            self.flush()
    
    def set_track_event(self, time, event_type, event):
        """Create a Perfetto track event packet."""

        packet = self.trace.packet.add()
        track_event = perfetto.TrackEvent()
        
        track_event.type = event_type
        track_event.track_uuid = event.tuuid
        track_event.name = event.function
        
        # Color hint
        if event.category is not None:
            track_event.categories.append(event.category)

        # Attach detail-panel key/values and flow arrows to the slice's begin event
        if event_type == perfetto.TrackEvent.Type.TYPE_SLICE_BEGIN:
            for key, value in (event.annotations or {}).items():
                annotation = track_event.debug_annotations.add()
                annotation.name = str(key)
                # Keep integers typed so they can be filtered and summed with SQL in the viewer
                if isinstance(value, int) and not isinstance(value, bool):
                    annotation.int_value = value
                else:
                    annotation.string_value = str(value)

            track_event.flow_ids.extend(event.flow_ids)
            track_event.terminating_flow_ids.extend(event.terminating_flow_ids)

        packet.timestamp = int(time - self.timestamp_origin_ns)
        packet.track_event.CopyFrom(track_event)
        packet.trusted_packet_sequence_id = 1


# =============================================================================
# INTERVAL MATH UTILITIES
# =============================================================================

def merge_intervals(intervals):
    """
    Merge overlapping time intervals.
    
    Args:
        intervals: List of (start, end) tuples
    
    Returns:
        List of non-overlapping merged intervals
    
    Example:
        [(1, 5), (3, 8), (10, 12)] -> [(1, 8), (10, 12)]
    """
    if not intervals:
        return []
    
    # Sort by start time
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        # Check if current overlaps with last merged interval
        if current[0] <= last[1]:
            # Extend the last interval
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            # Add as new interval
            merged.append(current)
    
    return merged


def subtract_intervals(base, subtract):
    """
    Remove 'subtract' intervals from 'base' intervals.
    
    Used to find DMA-only time (DMA active while SLICE inactive).
    
    Args:
        base: List of (start, end) intervals to subtract from
        subtract: List of (start, end) intervals to remove
    
    Returns:
        List of intervals representing base minus subtract
    """
    if not base:
        return []
    if not subtract:
        return list(base)
    
    subtract = merge_intervals(subtract)
    result = []
    
    for base_start, base_end in base:
        current_start = base_start
        
        for sub_start, sub_end in subtract:
            # Skip if subtract interval doesn't overlap
            if sub_end <= current_start or sub_start >= base_end:
                continue
            
            # Add portion before subtract interval
            if sub_start > current_start:
                result.append((current_start, min(sub_start, base_end)))
            
            # Move current start past subtract interval
            current_start = max(current_start, sub_end)
            
            if current_start >= base_end:
                break
        
        # Add remaining portion if any
        if current_start < base_end:
            result.append((current_start, base_end))
    
    return result


def intersect_intervals(intervals_a, intervals_b):
    """
    Find overlapping portions of two interval lists.
    
    Used to find DMA+SLICE overlap time.
    
    Args:
        intervals_a: First list of (start, end) intervals
        intervals_b: Second list of (start, end) intervals
    
    Returns:
        List of intervals where both lists overlap
    """
    if not intervals_a or not intervals_b:
        return []
    
    intervals_a = merge_intervals(intervals_a)
    intervals_b = merge_intervals(intervals_b)
    
    intersections = []
    i = j = 0
    
    while i < len(intervals_a) and j < len(intervals_b):
        start_a, end_a = intervals_a[i]
        start_b, end_b = intervals_b[j]
        
        # Find overlap
        overlap_start = max(start_a, start_b)
        overlap_end = min(end_a, end_b)
        
        if overlap_start < overlap_end:
            intersections.append((overlap_start, overlap_end))
        
        # Advance the interval that ends first
        if end_a < end_b:
            i += 1
        else:
            j += 1
    
    return intersections


def get_non_hal_bounds(workunits):
    """Return the outer time bounds of all non-HAL workunits."""
    non_hal_intervals = []

    for workunit in workunits:
        if isinstance(workunit, HalWorkUnitDebugInfo):
            continue

        start_time_ns = workunit.start_time_ns
        end_time_ns = workunit.end_time_ns
        if start_time_ns is None or end_time_ns is None:
            continue
        if end_time_ns < start_time_ns:
            continue

        non_hal_intervals.append((start_time_ns, end_time_ns))

    if not non_hal_intervals:
        return None, None

    return min(start for start, _ in non_hal_intervals), max(end for _, end in non_hal_intervals)


# =============================================================================
# STRING FORMATTING UTILITIES
# =============================================================================


def add_numeric_prefix(log_dict, start_index=1, width=2):
    """
    Add numeric prefixes to names for lexicographic ordering.
    
    Args:
        log_dict: Dictionary of {name: path}
        start_index: Starting number for prefix
        width: Zero-padding width
    
    Returns:
        Dictionary with prefixed names
    
    Example:
        {'profile.csv': '/path'} -> {'01 profile.csv': '/path'}
    """
    ordered = {}
    for i, (name, path) in enumerate(log_dict.items(), start=start_index):
        prefixed_name = f"{i:0{width}d} {name}"
        ordered[prefixed_name] = path
    return ordered



# =============================================================================
# METRICS CALCULATION
# =============================================================================


@dataclass
class MetricDescription:
    perfetto_name: tuple
    perfetto_thread_name: str
    perfetto_label: str
    dashboard_name: str
    dashboard_description: str
    dashboard_unit: str

_METRICS = [
    MetricDescription(('OVERALL', 'time'), '15 OVERALL', 'OVERALL', 'total_duration', 'Total execution duration of the workload including HAL operations', 'ns'),
    MetricDescription(('TOTAL_NPU_OPERATIONS', 'time'), None, None, 'total_npu_operations', 'Total execution duration of NPU operations', 'ns'),
    MetricDescription(('DMA', 'time'), '02 OVERVIEW DMA', 'DMA total', 'dma_time', 'Combined time for DMA and CDMA operations (union)', 'ns'),
    MetricDescription(('DMA', 'percent'), None, None, 'dma_percent', 'Percentage of dma_time over total duration', '%'),
    MetricDescription(('DMA_ONLY', 'time'), '08 OVERVIEW DMA ONLY', 'DMA/CDMA ONLY (no compute)', 'dma_only_time', 'Time spent exclusively on DMA/CDMA without compute overlap', 'ns'),
    MetricDescription(('DMA_ONLY', 'percent'), None, None, 'dma_only_percent', 'Percentage of dma_only_time over total duration', '%'),
    MetricDescription(('DMA_COMBINED', 'time'), '00 OVERVIEW DMA COMBINED', 'DMA+CDMA union', 'dma_total_time', 'Total DMA operation time (doesnt include CDMA)', 'ns'),
    MetricDescription(('DMA_COMBINED', 'percent'), None, None, 'dma_total_percent', 'Percentage of dma_total_time over total duration', '%'),
    MetricDescription(('CDMA', 'time'), '03 OVERVIEW CDMA', 'CDMA total', 'cdma_time', 'Total CDMA (Constant DMA) operation time', 'ns'),
    MetricDescription(('CDMA', 'percent'), None, None, 'cdma_percent', 'Percentage of cdma_time over total duration', '%'),
    MetricDescription(('DMA_IN', 'time'), None, None, 'dma_in_time', 'Time spent on DMA input transfers', 'ns'),
    MetricDescription(('DMA_IN', 'percent'), None, None, 'dma_in_percent', 'Percentage of dma_in_time over total duration', '%'),
    MetricDescription(('DMA_OUT', 'time'), None, None, 'dma_out_time', 'Time spent on DMA output transfers', 'ns'),
    MetricDescription(('DMA_OUT', 'percent'), None, None, 'dma_out_percent', 'Percentage of dma_out_time over total duration', '%'),
    MetricDescription(('COMPUTE_COMBINED', 'time'), '01 OVERVIEW COMPUTE COMBINED', 'SLICE+CSS union', 'compute_time', 'Combined compute time for SLICE and CSS operations (union)', 'ns'),
    MetricDescription(('COMPUTE_COMBINED', 'percent'), None, None, 'compute_percent', 'Percentage of compute_time over total duration', '%'),
    MetricDescription(('COMPUTE_ONLY', 'time'), '09 OVERVIEW COMPUTE ONLY', 'COMPUTE ONLY (no DMA/CDMA)', 'compute_only_time', 'Time spent exclusively on compute without DMA overlap', 'ns'),
    MetricDescription(('COMPUTE_ONLY', 'percent'), None, None, 'compute_only_percent', 'Percentage of compute_only_time over total duration', '%'),
    MetricDescription(('SLICE', 'time'), '04 OVERVIEW SLICE', 'SLICE 0 + 1 union', 'slice_time', 'Total time spent on SLICE 0 and 1 compute operations (union)', 'ns'),
    MetricDescription(('SLICE', 'percent'), None, None, 'slice_percent', 'Percentage of slice_time over total duration', '%'),
    MetricDescription(('SLICE_0', 'time'), '05 OVERVIEW SLICE 0', 'SLICE 0', 'slice_0_time', 'Time spent on SLICE 0 compute operations (exclusive)', 'ns'),
    MetricDescription(('SLICE_0', 'percent'), None, None, 'slice_0_percent', 'Percentage of slice_0_time over total duration', '%'),
    MetricDescription(('SLICE_1', 'time'), '06 OVERVIEW SLICE 1', 'SLICE 1', 'slice_1_time', 'Time spent on SLICE 1 compute operations (exclusive)', 'ns'),
    MetricDescription(('SLICE_1', 'percent'), None, None, 'slice_1_percent', 'Percentage of slice_1_time over total duration', '%'),
    MetricDescription(('CSS', 'time'), '07 OVERVIEW CSS', 'CSS', 'css_time', 'Total time spent on CSS (Compute Subsystem) operations', 'ns'),
    MetricDescription(('CSS', 'percent'), None, None, 'css_percent', 'Percentage of css_time over total duration', '%'),
    MetricDescription(('DMA_COMPUTE_OVERLAP', 'time'), '10 OVERVIEW DMA COMPUTE OVERLAP', 'DMA/CDMA<->COMPUTE overlap', 'overlap_time', 'Time where DMA/CDMA and compute operations overlap', 'ns'),
    MetricDescription(('DMA_COMPUTE_OVERLAP', 'percent'), None, None, 'overlap_percent', 'Percentage of overlap_time over total duration', '%'),
    MetricDescription(('IDLE', 'time'), '14 OVERVIEW IDLE', 'IDLE', 'idle_time', 'Time when hardware is idle (no DMA or compute, it could be host operation as well)', 'ns'),
    MetricDescription(('IDLE', 'percent'), None, None, 'idle_percent', 'Percentage of idle_time over total duration', '%'),
    MetricDescription(('HAL', 'time'), '11 OVERVIEW HAL', 'HAL total', 'hal_time', 'Total time spent on HAL operations', 'ns'),
    MetricDescription(('HAL', 'percent'), None, None, 'hal_percent', 'Percentage of hal_time over total duration', '%'),
    MetricDescription(('HOST', 'time'), '12 OVERVIEW HOST', 'HOST total', 'host_time', 'Total time spent on HOST operations', 'ns'),
    MetricDescription(('HOST', 'percent'), None, None, 'host_percent', 'Percentage of host_time over total duration', '%'),
    MetricDescription(('HOST_COPY', 'time'), '13 OVERVIEW HOST COPY', 'HOST COPY total', 'host_copy_time', 'Total time spent on HOST COPY operations', 'ns'),
    MetricDescription(('HOST_COPY', 'percent'), None, None, 'host_copy_percent', 'Percentage of host_copy_time over total duration', '%'),
]


def _interval_total(intervals):
    """Sum the lengths of merged (non-overlapping) intervals."""
    return sum(end - start for start, end in intervals)


def _split_dma_compute(dma_combined, compute_combined):
    """Split merged DMA and compute intervals into DMA-only, compute-only and overlap intervals."""
    return {
        'DMA_ONLY': subtract_intervals(dma_combined, compute_combined),
        'COMPUTE_ONLY': subtract_intervals(compute_combined, dma_combined),
        'DMA_COMPUTE_OVERLAP': intersect_intervals(dma_combined, compute_combined),
    }


def _overall_intervals(overall_start, overall_end):
    """Return the dispatch interval as a list, or an empty list if a bound is missing."""
    if overall_start is None or overall_end is None:
        return []
    return [(overall_start, overall_end)]


def _build_metrics(intervals, overall_start, overall_end, busy_intervals):
    """
    Build the metrics dict ({name: {'time', 'percent'}}) from merged intervals.

    IDLE time is the overall time minus the busy time, and OVERALL has no percent.
    """
    overall_time = (overall_end - overall_start) if (overall_start is not None and overall_end is not None) else 0
    busy_time = _interval_total(busy_intervals)
    idle_time = overall_time - busy_time if overall_time >= busy_time else 0

    def calc_percent(time):
        return (time / overall_time * 100) if overall_time else 0

    metrics = {}
    for name, merged in intervals.items():
        if name in ('IDLE', 'OVERALL'):
            continue
        total = _interval_total(merged)
        metrics[name] = {'time': total, 'percent': calc_percent(total)}

    metrics['IDLE'] = {'time': idle_time, 'percent': calc_percent(idle_time)}
    metrics['OVERALL'] = {'time': overall_time}
    return metrics


def compute_compile_metrics(dispatch: DispatchDebugInfo):
    """
    Compute hardware utilization metrics from the compile-time work unit timings.

    Returns:
        Dictionary with 'metrics', 'intervals' (merged intervals per metric),
        'overall_start' and 'overall_end'
    """

    overall_end = dispatch.end_time_ns
    overall_start = dispatch.start_time_ns

    dma_in_intervals = []
    dma_out_intervals = []
    cdma_dtcm_intervals = []
    cdma_itcm_intervals = []
    slice_intervals = []
    css_intervals = []

    for workunit in dispatch.workunits:

        start_time_ns = workunit.start_time_ns
        end_time_ns = workunit.end_time_ns

        if start_time_ns is None or end_time_ns is None:
            continue

        if isinstance(workunit, DmaInWorkUnitDebugInfo):
            dma_in_intervals.append((start_time_ns, end_time_ns))

        elif isinstance(workunit, DmaOutWorkUnitDebugInfo):
            dma_out_intervals.append((start_time_ns, end_time_ns))

        elif isinstance(workunit, SliceProgramWorkUnitDebugInfo):
            slice_intervals.append((start_time_ns, end_time_ns))
        
        elif isinstance(workunit, CdmaWorkUnitDebugInfo):
            if workunit.is_to_dtcm() or workunit.is_from_dtcm():
                cdma_dtcm_intervals.append((start_time_ns, end_time_ns))
            elif workunit.is_to_itcm() or workunit.is_from_itcm():
                cdma_itcm_intervals.append((start_time_ns, end_time_ns))
                
        elif isinstance(workunit, CssProgramWorkUnitDebugInfo):
            css_intervals.append((start_time_ns, end_time_ns))            

    cdma_intervals = cdma_dtcm_intervals + cdma_itcm_intervals

    # "DMA" statistics use the union of DMA and CDMA, "COMPUTE" statistics the union of SLICE and CSS
    dma_combined = merge_intervals(dma_in_intervals + dma_out_intervals + cdma_intervals)
    compute_combined = merge_intervals(slice_intervals + css_intervals)
    busy = merge_intervals(dma_combined + compute_combined)
    overall = _overall_intervals(overall_start, overall_end)

    intervals = {
        'DMA': merge_intervals(dma_in_intervals + dma_out_intervals),
        'DMA_IN': merge_intervals(dma_in_intervals),
        'DMA_OUT': merge_intervals(dma_out_intervals),
        'SLICE': merge_intervals(slice_intervals),
        'CDMA_DTCM': merge_intervals(cdma_dtcm_intervals),
        'CDMA_ITCM': merge_intervals(cdma_itcm_intervals),
        'CDMA': merge_intervals(cdma_intervals),
        'CSS': merge_intervals(css_intervals),
        'DMA_COMBINED': dma_combined,
        'COMPUTE_COMBINED': compute_combined,
        **_split_dma_compute(dma_combined, compute_combined),
        'IDLE': subtract_intervals(overall, busy),
        'OVERALL': overall,
    }

    return {
        'metrics': _build_metrics(intervals, overall_start, overall_end, busy),
        'intervals': intervals,
        'overall_start': overall_start,
        'overall_end': overall_end
    }


def _collect_runtime_intervals(workunits):
    """Collect the raw (start, end) intervals of each runtime executor from the work units."""
    collected = {name: [] for name in (
        'DMA', 'DMA_IN', 'DMA_OUT', 'CDMA', 'SLICE', 'SLICE_0', 'SLICE_1', 'CSS', 'HAL', 'HOST_COPY', 'HOST')}

    for workunit in workunits:
        
        start_time_ns = workunit.start_time_ns
        end_time_ns = workunit.end_time_ns

        if start_time_ns is None or end_time_ns is None:
            continue

        interval = (start_time_ns, end_time_ns)
        
        if isinstance(workunit, NssProgramWorkUnitDebugInfo):
            # NSS program timing covers every unit it uses
            if workunit.dma_in_used or workunit.dma_out_used:
                collected['DMA'].append(interval)
            if workunit.dma_in_used:
                collected['DMA_IN'].append(interval)
            if workunit.dma_out_used:
                collected['DMA_OUT'].append(interval)
            if workunit.cdma_used:
                collected['CDMA'].append(interval)
            if workunit.slice_used[0] or workunit.slice_used[1]:
                collected['SLICE'].append(interval)
            if workunit.slice_used[0]:
                collected['SLICE_0'].append(interval)
            if workunit.slice_used[1]:
                collected['SLICE_1'].append(interval)
            if workunit.css_used:
                collected['CSS'].append(interval)
        elif isinstance(workunit, HostCopyWorkUnitDebugInfo):        
            collected['HOST_COPY'].append(interval)
        elif isinstance(workunit, HostProgramWorkUnitDebugInfo):
            collected['HOST'].append(interval)
        elif isinstance(workunit, HalWorkUnitDebugInfo):
            collected['HAL'].append(interval)

    return collected


def compute_runtime_metrics(dispatch: DispatchDebugInfo):
    """
    Compute utilization metrics from the measured runtime work unit timings.

    Returns:
        Dictionary with 'metrics', 'intervals' (merged intervals per metric),
        'overall_start' and 'overall_end'
    """

    overall_start = dispatch.start_time_ns
    overall_end = dispatch.end_time_ns
    non_hal_start, non_hal_end = get_non_hal_bounds(dispatch.workunits)

    collected = _collect_runtime_intervals(dispatch.workunits)
    intervals = {name: merge_intervals(raw) for name, raw in collected.items()}

    dma_combined = merge_intervals(collected['DMA'] + collected['CDMA'])
    compute_combined = merge_intervals(collected['SLICE_0'] + collected['SLICE_1'] + collected['CSS'])

    # BUSY covers DMA, compute, HAL, host copy, and host activity.
    busy = merge_intervals(dma_combined + compute_combined + intervals['HAL'] + intervals['HOST_COPY'] + intervals['HOST'])
    overall = _overall_intervals(overall_start, overall_end)

    intervals.update({
        'DMA_COMBINED': dma_combined,
        'COMPUTE_COMBINED': compute_combined,
        **_split_dma_compute(dma_combined, compute_combined),
        'IDLE': subtract_intervals(overall, busy),
        'OVERALL': overall,
    })

    metrics = _build_metrics(intervals, overall_start, overall_end, busy)

    base_npu_ops = (
        non_hal_end - non_hal_start
        if (non_hal_start is not None and non_hal_end is not None)
        else 0
    )
    npu_activity = merge_intervals(dma_combined + compute_combined)
    host_activity = merge_intervals(intervals['HOST_COPY'] + intervals['HOST'])
    exclusive_host_intervals = subtract_intervals(host_activity, npu_activity + intervals['HAL'])
    metrics['TOTAL_NPU_OPERATIONS'] = {'time': max(0, base_npu_ops - _interval_total(exclusive_host_intervals))}
    
    return {
        'metrics': metrics,
        'intervals': intervals,
        'overall_start': overall_start,
        'overall_end': overall_end
    }


def print_metrics_summary(profile_name, metrics):
    """Print formatted metrics summary to console."""
    print(f"\n=== Overview ({profile_name}) ===")
    order = [
        'OVERALL', 'TOTAL_NPU_OPERATIONS', 'DMA_COMPUTE_OVERLAP', 'SLICE', 'SLICE_0', 'SLICE_1', 
        'COMPUTE_COMBINED', 'CSS', 'COMPUTE_ONLY', 'DMA_COMBINED', 
        'DMA_ONLY', 'DMA', 'CDMA', 'IDLE'
    ]
    for key in order:
        if key in metrics:
            metric = metrics[key]
            label = 'TOTAL NPU OPERATIONS' if key == 'TOTAL_NPU_OPERATIONS' else key
            if 'percent' in metric:
                print(f"{label} time: {metric['time']} ({metric['percent']:.2f}%)")
            else:
                print(f"{label} time: {metric['time']}")


# =============================================================================
# PERFETTO TRACE RENDERING
# =============================================================================


def format_overview_summary(label, metric):
    """
    Format an overview total such as "DMA total: 1.234µs (5.00%)".

    report.py reads these strings back from the .pb file, so keep the format.
    """
    summary = f"{label}: {format_time_duration(metric['time'])}"
    if 'percent' in metric:
        summary += f" ({metric['percent']:.2f}%)"
    return summary


@dataclass
class PropertyTrack:
    property_name: str
    thread_name: str
    index: Optional[int] = None
    uuid: Optional[str] = None



def _get_compile_time_range(dispatch, start_op, end_op):
    """Get compile time range from start/end operations. Returns (start_ns, end_ns, duration_ns) or None."""
    start_info = dispatch.get_compile_time_info(start_op)
    end_info = dispatch.get_compile_time_info(end_op)
    if start_info is None or end_info is None:
        return None
    start_ns = start_info.start_time_ns
    end_ns = end_info.end_time_ns
    if start_ns is None or end_ns is None:
        return None
    duration_ns = end_ns - start_ns
    if duration_ns <= 0:
        return None
    return start_ns, end_ns, duration_ns


def _get_compile_runtime_scale(compile_duration_ns, runtime_start_ns, runtime_end_ns):
    """Fit the compile overlay to runtime; this is separate from clock normalization."""
    if runtime_start_ns is None or runtime_end_ns is None:
        raise Exception("Missing runtime start or end timestamp")

    runtime_duration_ns = runtime_end_ns - runtime_start_ns
    if runtime_duration_ns <= 0 or compile_duration_ns <= 0:
        raise Exception("Zero or negative runtime or compile time duration")

    return runtime_duration_ns / compile_duration_ns


# Compile-time tracks for each work unit type. Other work unit types have no compile-time slice.
_COMPILE_TRACK_NAMES = OrderedDict([
    (torq_debug_info.NssProgramWorkUnitDebugInfo, ["NSS Programs (Compile)"]),
    (torq_debug_info.NssCfgWorkUnitDebugInfo, ["NSS CFG Tasks (Compile)"]),
    (torq_debug_info.CdmaWorkUnitDebugInfo, ["CDMA (Compile)"]),
    (torq_debug_info.DmaInWorkUnitDebugInfo, ["NDMA In (Compile)"]),
    (torq_debug_info.DmaOutWorkUnitDebugInfo, ["NDMA Out (Compile)"]),
    (torq_debug_info.SliceProgramWorkUnitDebugInfo, ["Slice 0 (Compile)", "Slice 1 (Compile)"]),
    (torq_debug_info.CssProgramWorkUnitDebugInfo, ["CSS (Compile)"]),
])


@dataclass
class CompilePlacement:
    """Where the compile-time estimate of a work unit is drawn, in runtime time."""
    workunit: object
    track_index: int  # index in the compile tracks of the work unit type (the executor instance)
    start: int
    end: int
    async_duration_ns: Optional[int]


def _index_nss_programs(dispatch):
    """
    Index the NSS programs that have compile times. Returns [(program, compile start, compile end,
    scale)], {related work unit: (program, compile start, scale)} and {program: scale}.
    """
    nss_program_lookup = []
    parent_nss_map = {}
    nss_scale_map = {}

    for wu in dispatch.workunits:
        if not isinstance(wu, NssProgramWorkUnitDebugInfo):
            continue
        time_range = _get_compile_time_range(dispatch, wu.start_operation, wu.end_operation)
        if time_range is not None and wu.start_time_ns is not None:
            scale = _get_compile_runtime_scale(time_range[2], wu.start_time_ns, wu.end_time_ns)
            nss_program_lookup.append((wu, time_range[0], time_range[1], scale))
            nss_scale_map[wu] = scale
            for related_wu in wu.related_workunits:
                parent_nss_map[related_wu] = (wu, time_range[0], scale)

    return nss_program_lookup, parent_nss_map, nss_scale_map


def _scaled_async_duration(dispatch, workunit, scale):
    """Compile-time async duration of a work unit, scaled to runtime, or None."""
    compile_info = dispatch.get_compile_time_info(workunit.start_operation)
    async_dur = compile_info.async_duration_ns if compile_info else None
    return round(async_dur * scale) if async_dur is not None else None


def _nss_program_placement(dispatch, workunit, nss_scale_map):
    """Place an NSS program at its runtime start, with its scaled compile duration."""
    time_range = _get_compile_time_range(dispatch, workunit.start_operation, workunit.end_operation)
    if time_range is None or workunit.start_time_ns is None:
        return None
    scale = nss_scale_map[workunit]
    end = workunit.start_time_ns + round(time_range[2] * scale)
    return CompilePlacement(workunit, 0, workunit.start_time_ns, end,
                            _scaled_async_duration(dispatch, workunit, scale))


def _managed_parent(dispatch, workunit, parent_nss_map):
    """
    Compile range and parent NSS program of an NSS-managed work unit, from its nss_task op.
    Returns (compile start, compile duration, parent, parent compile start, scale), or None.
    """
    has_parent_start = workunit.start_operation is not None and hasattr(workunit.start_operation, 'parent')
    has_parent_end = workunit.end_operation is not None and hasattr(workunit.end_operation, 'parent')
    if not has_parent_start or not has_parent_end:
        return None
    time_range = _get_compile_time_range(dispatch, workunit.start_operation.parent, workunit.end_operation.parent)
    parent_entry = parent_nss_map.get(workunit)
    if time_range is None or parent_entry is None:
        return None
    parent_nss, parent_nss_compile_start, scale = parent_entry
    return time_range[0], time_range[2], parent_nss, parent_nss_compile_start, scale


def _cfg_parent(dispatch, workunit, nss_program_lookup):
    """
    Compile range and parent NSS program of an NSS CFG task. The parent is the NSS program whose
    compile range contains the task. Returns the same tuple as _managed_parent, or None.
    """
    time_range = _get_compile_time_range(dispatch, workunit.start_operation, workunit.end_operation)
    if time_range is None:
        return None
    compile_start, compile_end, compile_duration = time_range
    for (nss_wu, nss_cs, nss_ce, nss_scale) in nss_program_lookup:
        if nss_cs <= compile_start and compile_end <= nss_ce and workunit.dispatch is nss_wu.dispatch:
            return compile_start, compile_duration, nss_wu, nss_cs, nss_scale
    return None


def _child_placement(dispatch, workunit, parent_info):
    """Place a work unit at the runtime start of its parent NSS program, plus its scaled compile offset."""
    compile_start, compile_duration, parent_nss, parent_nss_compile_start, scale = parent_info
    if parent_nss.start_time_ns is None:
        return None
    track_index = workunit.executor_instance_id
    if track_index >= len(_COMPILE_TRACK_NAMES[type(workunit)]):
        return None
    offset_within_nss = compile_start - parent_nss_compile_start
    start = parent_nss.start_time_ns + round(offset_within_nss * scale)
    end = parent_nss.start_time_ns + round((offset_within_nss + compile_duration) * scale)
    return CompilePlacement(workunit, track_index, start, end, _scaled_async_duration(dispatch, workunit, scale))


def _compile_placements(dispatch):
    """
    Return the CompilePlacement of each work unit whose compile-time estimate can be drawn:
    the estimates are scaled into the measured runtime interval of their NSS program.
    """
    if not dispatch.compile_operation_times_ns:
        return []

    nss_program_lookup, parent_nss_map, nss_scale_map = _index_nss_programs(dispatch)
    placements = []
    for workunit in dispatch.workunits:
        if type(workunit) not in _COMPILE_TRACK_NAMES:
            continue
        if isinstance(workunit, NssProgramWorkUnitDebugInfo):
            placement = _nss_program_placement(dispatch, workunit, nss_scale_map)
        else:
            if isinstance(workunit, torq_debug_info.NssManagedWorkUnitDebugInfo):
                parent_info = _managed_parent(dispatch, workunit, parent_nss_map)
            else:
                parent_info = _cfg_parent(dispatch, workunit, nss_program_lookup)
            placement = _child_placement(dispatch, workunit, parent_info) if parent_info is not None else None
        if placement is not None:
            placements.append(placement)
    return placements


def _emit_compile_event(trace_writer, tuuid, placement, flow_ids):
    """Emit a compile-time event, splitting into active + WAITING if async duration is shorter."""
    workunit = placement.workunit
    async_duration_ns = placement.async_duration_ns
    if async_duration_ns is not None and async_duration_ns < placement.end - placement.start:
        active_end = placement.start + async_duration_ns
    else:
        active_end = placement.end

    trace_writer.add_event(PerfettoEvent(tuuid, workunit.trace_name, placement.start, active_end,
                                         annotations=workunit.annotations,
                                         terminating_flow_ids=flow_ids.get(workunit, [])))
    if active_end < placement.end:
        trace_writer.add_event(PerfettoEvent(tuuid, "WAITING", active_end, placement.end))


def _log_compile_time_traces(trace_writer, dispatch, placements, flow_ids):
    """
    Log compile-time estimates (see _compile_placements) on their own tracks.

    flow_ids maps a work unit to the flow that links its runtime slice to its compile slice.
    """
    if not dispatch.compile_operation_times_ns:
        return

    # --- NPU (Compile): detailed per-workunit view ---
    compile_npu_process = trace_writer.add_process_descriptor("NPU Task Breakdown")

    compile_workunits_tracks = {}
    for workunit_type, track_names in _COMPILE_TRACK_NAMES.items():
        for track_name in track_names:
            tuuid = trace_writer.add_thread_descriptor(compile_npu_process, track_name)
            compile_workunits_tracks.setdefault(workunit_type, []).append(tuuid)

    for placement in placements:
        tuuid = compile_workunits_tracks[type(placement.workunit)][placement.track_index]
        _emit_compile_event(trace_writer, tuuid, placement, flow_ids)


@dataclass
class WorkUnitFlows:
    """Flow ids that start at (outgoing) or end at (incoming) the runtime slice of one work unit."""
    outgoing: list = field(default_factory=list)
    incoming: list = field(default_factory=list)


def _data_dependency_links(workunits):
    """
    Find (producer, consumer) pairs of work units that pass data through a buffer:
    - a DMA In or CDMA links to each later Slice program that uses the buffer it wrote
    - a Slice program links to each later DMA Out or CDMA that reads a buffer the slice used last

    Slice programs give no read/write direction for their buffers, so Slice -> Slice links are not made.
    Buffers are keyed with their dispatch, because each dispatch numbers its global buffers from 0
    and a combined view has the work units of more than one dispatch.
    """
    last_transfer = {}  # (dispatch, buffer) -> data transfer that wrote it last
    last_slice = {}     # (dispatch, buffer) -> Slice program that used it last
    links = []

    timed = [wu for wu in workunits if wu.start_time_ns is not None]
    for workunit in sorted(timed, key=lambda wu: wu.start_time_ns):
        for buffer in workunit.buffers_used:
            key = (workunit.dispatch, buffer)
            if key in last_transfer:
                links.append((last_transfer[key], workunit))
            last_slice[key] = workunit

        for buffer in workunit.buffers_read:
            key = (workunit.dispatch, buffer)
            if key in last_slice:
                links.append((last_slice[key], workunit))

        for buffer in workunit.buffers_written:
            key = (workunit.dispatch, buffer)
            last_transfer[key] = workunit
            last_slice.pop(key, None)

    # A pair can share more than one buffer, keep one arrow per pair
    return list(dict.fromkeys(links))


def _assign_flow_ids(trace_writer, dispatch, compile_placements):
    """
    Choose the flow arrows of the runtime slices:
    - data dependencies between DMA/CDMA transfers and Slice programs (see _data_dependency_links)
    - each work unit with a compile-time slice (see _compile_placements) links to that slice

    Returns the runtime flows per work unit and the compile flow id per work unit.
    """
    runtime_flows = {}
    compile_flow_ids = {}

    for producer, consumer in _data_dependency_links(dispatch.workunits):
        flow_id = trace_writer.get_next_uuid()
        runtime_flows.setdefault(producer, WorkUnitFlows()).outgoing.append(flow_id)
        runtime_flows.setdefault(consumer, WorkUnitFlows()).incoming.append(flow_id)

    for placement in compile_placements:
        flow_id = trace_writer.get_next_uuid()
        runtime_flows.setdefault(placement.workunit, WorkUnitFlows()).outgoing.append(flow_id)
        compile_flow_ids[placement.workunit] = [flow_id]

    return runtime_flows, compile_flow_ids


def _emit_runtime_workunit(trace_writer, tuuid, workunit, flows):
    """Emit the slice of one runtime work unit, plus a WAITING slice if it is ready before it releases the executor."""
    ready_time_ns = workunit.ready_time_ns
    active_end = ready_time_ns if ready_time_ns is not None else workunit.end_time_ns

    trace_writer.add_event(PerfettoEvent(tuuid, workunit.trace_name, workunit.start_time_ns, active_end,
                                         annotations=workunit.annotations,
                                         flow_ids=flows.outgoing, terminating_flow_ids=flows.incoming))
    if ready_time_ns is not None:
        trace_writer.add_event(PerfettoEvent(tuuid, "WAITING", ready_time_ns, workunit.end_time_ns))


# Kind of data that a DMA/CDMA source label (see torq_debug_info.buffer_label) puts in an LRAM buffer
_LRAM_KINDS = ("const", "code", "activations")


def _lram_buffer_kind(source_label):
    """Group the label of a transfer source into one of _LRAM_KINDS."""
    if source_label == "const":
        return "const"
    if source_label.startswith("code ") or source_label == "nss code":
        return "code"
    return "activations"


def _lram_buffer_lifetimes(dispatch, lram_ranges):
    """
    Return {buffer: (first start, last end)} of the LRAM buffers, from the first to the last
    work unit that touches the buffer, and {buffer: kind} from the transfer that fills it.
    A buffer that no transfer fills is written by a slice, so it holds activations.
    """
    lifetimes = {}
    kinds = {}

    for workunit in dispatch.workunits:
        if workunit.start_time_ns is None or workunit.end_time_ns is None:
            continue

        buffers = workunit.buffers_read + workunit.buffers_written + workunit.buffers_used
        for buffer in buffers:
            if buffer not in lram_ranges:
                continue
            start, end = lifetimes.get(buffer, (workunit.start_time_ns, workunit.end_time_ns))
            lifetimes[buffer] = (min(start, workunit.start_time_ns), max(end, workunit.end_time_ns))

        for buffer in workunit.buffers_written:
            if buffer in lram_ranges and buffer not in kinds:
                kinds[buffer] = _lram_buffer_kind(torq_debug_info.buffer_label(workunit.buffers_read[0]))

    return lifetimes, {buffer: kinds.get(buffer, "activations") for buffer in lifetimes}


def _lram_usage_steps(lifetimes, lram_ranges):
    """
    Return [(time, used bytes)] at each time where the set of live buffers changes.
    Used bytes is the size of the union of the address ranges of the live buffers.
    """
    changes = {}
    for buffer, (start, end) in lifetimes.items():
        changes.setdefault(start, []).append((buffer, True))
        changes.setdefault(end, []).append((buffer, False))

    live = set()
    steps = []
    for time in sorted(changes):
        for buffer, is_start in changes[time]:
            if is_start:
                live.add(buffer)
            else:
                live.discard(buffer)
        # A buffer that starts and ends at the same time is not counted
        used = _interval_total(merge_intervals([lram_ranges[buffer] for buffer in live]))
        steps.append((time, used))
    return steps


def _lram_counter_steps(dispatch):
    """
    Return {counter name: [(time, used bytes)]} for one MLIR dispatch: "LRAM used" for all
    buffers, and "LRAM <kind>" for each kind of data (see _LRAM_KINDS).
    """
    lram_ranges = dispatch.lram_buffer_ranges()
    lifetimes, kinds = _lram_buffer_lifetimes(dispatch, lram_ranges)

    steps = {"LRAM used": _lram_usage_steps(lifetimes, lram_ranges)}
    for kind in _LRAM_KINDS:
        kind_lifetimes = {buffer: span for buffer, span in lifetimes.items() if kinds[buffer] == kind}
        steps[f"LRAM {kind}"] = _lram_usage_steps(kind_lifetimes, lram_ranges)
    return steps


def log_lram_usage(trace_writer, dispatch):
    """
    Add counter tracks with the bytes of LRAM in use over time. A combined view runs its
    dispatches one after the other, so the steps of all dispatches are joined on one track.
    """
    counters = OrderedDict()
    for part in dispatch.lram_dispatches():
        for counter_name, steps in _lram_counter_steps(part).items():
            counters.setdefault(counter_name, []).extend(steps)

    if not any(counters.values()):
        return

    process_uuid = trace_writer.add_process_descriptor(
        "LRAM Usage", perfetto.TrackDescriptor.ChildTracksOrdering.EXPLICIT)

    for rank, (counter_name, steps) in enumerate(counters.items()):
        peak = max((used for _, used in steps), default=0)
        tuuid = trace_writer.add_counter_descriptor(
            process_uuid, f"{counter_name} (peak {torq_debug_info.format_bytes(peak)})", order_rank=rank)
        for time, used in sorted(steps):
            trace_writer.add_counter_value(tuuid, time, used)


# Causes of a slice stall. When units overlap during a stall, the time goes to the unit
# with the lowest number.
_STALL_PRIORITY = {"DMA In": 0, "CDMA": 1, "DMA Out": 2, "css": 3, "host copy": 4, "slice": 5, "nss": 6}

_TRANSFER_DIRECTIONS = [
    (DmaInWorkUnitDebugInfo, "DMA In"),
    (CdmaWorkUnitDebugInfo, "CDMA"),
    (DmaOutWorkUnitDebugInfo, "DMA Out"),
]


@dataclass
class StallActivity:
    """A unit that can run while a slice stalls: its active interval and how a stall shows it."""
    start: int
    end: int
    name: str      # child slice name, e.g. "DMA In const 768 B"
    kind: str      # cause, e.g. "DMA In const"
    priority: int


def _active_end_ns(workunit):
    """End of the active part of a work unit: when its result is ready."""
    return workunit.ready_time_ns if workunit.ready_time_ns is not None else workunit.end_time_ns


def _transfer_kind(direction, trace_name):
    """Cause of a transfer without its size, e.g. "DMA In const" or "DMA In code"."""
    label = re.sub(r" [0-9.]+ (B|KiB|MiB)$", "", trace_name)
    if label.startswith("code "):
        label = "code"
    return f"{direction} {label}"


def _stall_activity(workunit):
    """Return the StallActivity of a work unit, or None if the work unit is not a stall cause."""
    start, end = workunit.start_time_ns, _active_end_ns(workunit)
    if start is None or end is None or end <= start:
        return None

    for workunit_type, direction in _TRANSFER_DIRECTIONS:
        if isinstance(workunit, workunit_type):
            name = workunit.trace_name
            return StallActivity(start, end, f"{direction} {name}", _transfer_kind(direction, name),
                                 _STALL_PRIORITY[direction])

    if isinstance(workunit, SliceProgramWorkUnitDebugInfo):
        slice_name = f"slice {workunit.executor_instance_id}"
        return StallActivity(start, end, f"{slice_name} {workunit.trace_name}", slice_name, _STALL_PRIORITY["slice"])

    if isinstance(workunit, CssProgramWorkUnitDebugInfo):
        return StallActivity(start, end, "css", "css", _STALL_PRIORITY["css"])

    if isinstance(workunit, HostCopyWorkUnitDebugInfo):
        return StallActivity(start, end, "host copy", "host copy", _STALL_PRIORITY["host copy"])

    if isinstance(workunit, NssCfgWorkUnitDebugInfo):
        return StallActivity(start, end, "nss", "nss", _STALL_PRIORITY["nss"])

    return None


class _ActivityIndex:
    """Stall activities sorted by start time, to find the ones that overlap an interval."""

    def __init__(self, activities):
        self.activities = sorted(activities, key=lambda activity: activity.start)
        self.starts = [activity.start for activity in self.activities]
        self.max_duration = max((activity.end - activity.start for activity in self.activities), default=0)

    def overlapping(self, start, end):
        first = bisect.bisect_left(self.starts, start - self.max_duration)
        last = bisect.bisect_left(self.starts, end)
        return [activity for activity in self.activities[first:last] if activity.end > start]


def _slice_gaps(dispatch, slice_id):
    """
    Return the stalls of one slice as [(start, end, next slice program or None)]: the time from
    the dispatch start to the first program, between programs, and after the last program.
    """
    programs = [wu for wu in dispatch.workunits
                if isinstance(wu, SliceProgramWorkUnitDebugInfo) and wu.executor_instance_id == slice_id
                and wu.start_time_ns is not None and _active_end_ns(wu) is not None]
    if not programs:
        return []
    programs.sort(key=lambda wu: wu.start_time_ns)

    gaps = []
    previous_end = dispatch.start_time_ns if dispatch.start_time_ns is not None else programs[0].start_time_ns
    for program in programs:
        if program.start_time_ns > previous_end:
            gaps.append((previous_end, program.start_time_ns, program))
        previous_end = max(previous_end, _active_end_ns(program))

    if dispatch.end_time_ns is not None and dispatch.end_time_ns > previous_end:
        gaps.append((previous_end, dispatch.end_time_ns, None))
    return gaps


def _stall_segments(gap_start, gap_end, activities):
    """
    Split a stall into segments [(start, end, name, kind)]. Each segment goes to the running
    activity with the highest priority, or to "idle" if nothing runs. Neighbor segments with
    the same name are joined.
    """
    points = {gap_start, gap_end}
    for activity in activities:
        points.update((max(activity.start, gap_start), min(activity.end, gap_end)))
    points = sorted(points)

    segments = []
    for seg_start, seg_end in zip(points, points[1:]):
        running = [a for a in activities if a.start < seg_end and a.end > seg_start]
        if running:
            top = min(running, key=lambda activity: activity.priority)
            name, kind = top.name, top.kind
        else:
            name, kind = "idle", "idle"

        if segments and segments[-1][2] == name:
            segments[-1] = (segments[-1][0], seg_end, name, kind)
        else:
            segments.append((seg_start, seg_end, name, kind))
    return segments


def _emit_stall(trace_writer, tuuid, start, end, next_program, segments):
    """
    Emit one stall slice, named after the cause with the most time, and one child slice per
    segment. The events are written in time order so that the children nest in the stall.
    Returns the name of the stall slice.
    """
    time_per_kind = OrderedDict()
    for seg_start, seg_end, _, kind in segments:
        time_per_kind[kind] = time_per_kind.get(kind, 0) + seg_end - seg_start
    main_kind = max(time_per_kind, key=time_per_kind.get)

    annotations = {"stall ns": end - start}
    if next_program is not None:
        annotations["waiting slice"] = next_program.trace_name
        annotations["location"] = next_program.annotations["location"]
        annotations["operator"] = _operator_label(next_program)
    annotations.update({f"{kind} ns": time for kind, time in time_per_kind.items()})

    stall = PerfettoEvent(tuuid, f"stall: {main_kind}", start, end, annotations=annotations)
    trace_writer.set_track_event(start, perfetto.TrackEvent.Type.TYPE_SLICE_BEGIN, stall)
    for seg_start, seg_end, name, _ in segments:
        trace_writer.add_event(PerfettoEvent(tuuid, name, seg_start, seg_end))
    trace_writer.set_track_event(end, perfetto.TrackEvent.Type.TYPE_SLICE_END, stall)
    return stall.function


def log_slice_stalls(trace_writer, dispatch):
    """
    Add one track per slice with its stalls (see _slice_gaps). Each stall shows what runs
    during it (see _stall_segments), so you can see which transfer a slice waits for.

    Returns the stalls as [(start, end, next slice program or None, stall name)].
    """
    slice_ids = sorted({wu.executor_instance_id for wu in dispatch.workunits
                        if isinstance(wu, SliceProgramWorkUnitDebugInfo)})
    if not slice_ids:
        return []

    activities = [activity for activity in map(_stall_activity, dispatch.workunits) if activity is not None]
    index = _ActivityIndex(activities)

    process_uuid = trace_writer.add_process_descriptor(
        "Slice Stalls", perfetto.TrackDescriptor.ChildTracksOrdering.EXPLICIT)

    stalls = []
    for rank, slice_id in enumerate(slice_ids):
        gaps = _slice_gaps(dispatch, slice_id)
        total = sum(end - start for start, end, _ in gaps)
        tuuid = trace_writer.add_thread_descriptor(
            process_uuid, f"Slice {slice_id} stalls (total {format_time_duration(total)})", order_rank=rank)
        for start, end, next_program in gaps:
            segments = _stall_segments(start, end, index.overlapping(start, end))
            name = _emit_stall(trace_writer, tuuid, start, end, next_program, segments)
            stalls.append((start, end, next_program, name))
    return stalls


def _operator_label(workunit):
    """
    Input operators of a work unit, e.g. "tosa.conv2d@L317". If the input file cannot be read,
    the short location of the work unit, e.g. "@L317C12".
    """
    debug_info = workunit.dispatch.debug_info
    try:
        operators = debug_info.get_original_operators(workunit.location)
    except (OSError, IndexError):
        operators = []
    return " ".join(operators) if operators else debug_info.pretty_print_location(workunit.location)


def _join_overlapping(intervals):
    """Join (start, end, name) intervals that overlap in time. A joined interval keeps the first name."""
    joined = []
    for start, end, name in sorted(intervals):
        if joined and start < joined[-1][1]:
            joined[-1] = (joined[-1][0], max(end, joined[-1][1]), joined[-1][2])
        else:
            joined.append((start, end, name))
    return joined


def log_operator_stalls(trace_writer, stalls):
    """
    Add one track per input operator with the stalls before its slice programs, ranked by the
    stall time. Stalls of two slices that overlap in time are joined, and the time in the track
    name is the joined time, so it is the length of the slices on the track.
    """
    stalls_per_operator = {}
    for start, end, next_program, name in stalls:
        if next_program is not None:
            stalls_per_operator.setdefault(_operator_label(next_program), []).append((start, end, name))
    if not stalls_per_operator:
        return

    joined = {label: _join_overlapping(intervals) for label, intervals in stalls_per_operator.items()}
    totals = {label: sum(end - start for start, end, _ in intervals) for label, intervals in joined.items()}
    ranked = sorted(totals, key=lambda label: (-totals[label], label))
    width = max(2, len(str(len(ranked))))

    process_uuid = trace_writer.add_process_descriptor(
        "Operator Stalls", perfetto.TrackDescriptor.ChildTracksOrdering.EXPLICIT)
    for rank, label in enumerate(ranked):
        tuuid = trace_writer.add_thread_descriptor(
            process_uuid, f"{rank + 1:0{width}d} {label} - {format_time_duration(totals[label])}", order_rank=rank)
        for start, end, name in joined[label]:
            trace_writer.add_event(PerfettoEvent(tuuid, name, start, end))


# Operator type of the locations whose operator is not known
_OTHER_LOCATIONS = "other locations"

# Per-location metrics, in the order of the summary slice annotations
_LOCATION_METRICS = ("SLICE", "DMA", "DMA_IN", "DMA_OUT", "CDMA", "CSS", "DMA_COMBINED", "COMPUTE_COMBINED",
                     "DMA_ONLY", "COMPUTE_ONLY", "DMA_COMPUTE_OVERLAP")


def _location_metrics(workunits):
    """
    Return the total time of a location (the union of the intervals of its timed work units,
    see _timed_location_workunits) and its metrics ({name: ns}).
    """
    intervals = []
    per_type = {"DMA_IN": [], "DMA_OUT": [], "CDMA": [], "SLICE": [], "CSS": []}
    type_names = [(DmaInWorkUnitDebugInfo, "DMA_IN"), (DmaOutWorkUnitDebugInfo, "DMA_OUT"),
                  (CdmaWorkUnitDebugInfo, "CDMA"), (SliceProgramWorkUnitDebugInfo, "SLICE"),
                  (CssProgramWorkUnitDebugInfo, "CSS")]

    for workunit in workunits:
        start, end = workunit.start_time_ns, workunit.end_time_ns
        intervals.append((start, end))
        for workunit_type, name in type_names:
            if isinstance(workunit, workunit_type):
                per_type[name].append((start, end))

    dma_combined = merge_intervals(per_type["DMA_IN"] + per_type["DMA_OUT"] + per_type["CDMA"])
    compute_combined = merge_intervals(per_type["SLICE"] + per_type["CSS"])
    merged = {name: merge_intervals(raw) for name, raw in per_type.items()}
    merged.update({
        "DMA": merge_intervals(per_type["DMA_IN"] + per_type["DMA_OUT"]),
        "DMA_COMBINED": dma_combined,
        "COMPUTE_COMBINED": compute_combined,
        **_split_dma_compute(dma_combined, compute_combined),
    })

    metrics = {name: _interval_total(merged[name]) for name in _LOCATION_METRICS}
    return _interval_total(merge_intervals(intervals)), metrics


def _read_source_lines(original_locations):
    """Read the input source lines of the locations: {file: {line number: text}}."""
    requests = {}
    for original_loc_info in original_locations:
        raw_file = original_loc_info.location.file.strip('"') if original_loc_info.location.file else ""
        if not raw_file or not os.path.isfile(raw_file):
            continue
        try:
            requests.setdefault(raw_file, set()).add(int(original_loc_info.location.line))
        except ValueError:
            continue

    source_lines = {}
    for raw_file, requested in requests.items():
        found = {}
        try:
            with open(raw_file, "r") as src_f:
                for line_no, line in enumerate(src_f, start=1):
                    if line_no in requested:
                        found[line_no] = line
                        if len(found) == len(requested):
                            break
        except OSError:
            found = {}
        source_lines[raw_file] = found
    return source_lines


def _location_source_line(original_loc_info, source_lines):
    """Return (line number, text) of the input source line of a location, or None."""
    raw_file = original_loc_info.location.file.strip('"') if original_loc_info.location.file else ""
    try:
        line_no = int(original_loc_info.location.line)
    except ValueError:
        return None
    line = source_lines.get(raw_file, {}).get(line_no)
    return (line_no, line) if line is not None else None


@dataclass
class LocationGroup:
    """One input location in the location tracks: its label, operator type, times and work units."""
    info: object
    label: str          # e.g. "tosa.conv2d@L294", or "file:line:col" if the operator is not known
    operator_type: str  # e.g. "tosa.conv2d"
    total_ns: int
    metrics: dict
    workunits: list     # the work units with valid times, without NSS programs


def _timed_location_workunits(original_loc_info):
    """Work units of a location with valid times. NSS programs are left out, because they contain the others."""
    return [wu for wu in original_loc_info.workunits
            if not isinstance(wu, NssProgramWorkUnitDebugInfo)
            and wu.start_time_ns is not None and wu.end_time_ns is not None
            and wu.end_time_ns >= wu.start_time_ns]


def _location_groups(dispatch):
    """
    Return the LocationGroup of each input location that has timed work units. In a runtime
    profile, only the host actions have times, so most locations are left out. A location on a
    func.func line is left out, because it is the whole function and not one operator. The
    operator name is read with the location's own dispatch, because a combined view has no debug info.
    """
    source_lines = _read_source_lines(dispatch.original_locations)
    groups = []
    for original_loc_info in dispatch.original_locations:
        workunits = _timed_location_workunits(original_loc_info)
        if not workunits:
            continue
        source_line = _location_source_line(original_loc_info, source_lines)
        if source_line is not None and "func.func" in source_line[1]:
            continue
        op_name = None
        if source_line is not None:
            op_name = original_loc_info.dispatch.debug_info._extract_operator(*source_line)
        location = original_loc_info.location
        label = op_name if op_name is not None else f"{location.file}:{location.line}:{location.col}"
        operator_type = op_name.split("@L", 1)[0] if op_name is not None else _OTHER_LOCATIONS
        total_ns, metrics = _location_metrics(workunits)
        groups.append(LocationGroup(original_loc_info, label, operator_type, total_ns, metrics, workunits))
    return groups


def _location_workunit_track_name(workunit):
    """Name of the work unit track in a location group, e.g. "Dma In" or "Slice Program 1"."""
    if isinstance(workunit, SliceProgramWorkUnitDebugInfo):
        return f"{workunit.pretty_name()} {workunit.executor_instance_id}"
    return workunit.pretty_name()


def _emit_location_group(trace_writer, parent_uuid, rank, group):
    """
    Emit the tracks of one location: a summary track with one slice per time span in which the
    location is busy (the merged work unit intervals), with the location metrics as annotations,
    and one track per work unit type with the work units of the location.
    """
    explicit = perfetto.TrackDescriptor.ChildTracksOrdering.EXPLICIT
    location_uuid = trace_writer.add_thread_descriptor(
        parent_uuid, f"{group.label} - {format_time_duration(group.total_ns)}", order_rank=rank,
        child_ordering=explicit)

    summary_uuid = trace_writer.add_thread_descriptor(location_uuid, "summary", order_rank=0)
    annotations = {f"{name} ns": value for name, value in group.metrics.items() if value > 0}
    annotations["total ns"] = group.total_ns
    for start, end in merge_intervals([(wu.start_time_ns, wu.end_time_ns) for wu in group.workunits]):
        trace_writer.add_event(PerfettoEvent(summary_uuid, group.label, start, end, annotations=annotations))

    tracks = {}
    for workunit in group.workunits:
        track_name = _location_workunit_track_name(workunit)
        if track_name not in tracks:
            tracks[track_name] = trace_writer.add_thread_descriptor(location_uuid, track_name,
                                                                    order_rank=len(tracks) + 1)
        trace_writer.add_event(PerfettoEvent(tracks[track_name], workunit.trace_name, workunit.start_time_ns,
                                             workunit.end_time_ns, annotations=workunit.annotations))


def _log_operator_type_totals(trace_writer, dispatch, groups):
    """Add one track per operator type with a bar of its total time, sorted by the total."""
    operator_total_times = {}
    for group in groups:
        if group.operator_type != _OTHER_LOCATIONS and group.total_ns > 0:
            operator_total_times[group.operator_type] = operator_total_times.get(group.operator_type, 0) + group.total_ns
    if not operator_total_times:
        return

    process_uuid = trace_writer.add_process_descriptor("Operator Type Totals")
    base_start = dispatch.start_time_ns if dispatch.start_time_ns is not None else 0
    sorted_operator_totals = sorted(operator_total_times.items(), key=lambda item: (-item[1], item[0]))
    width = max(2, len(str(len(sorted_operator_totals))))

    for index, (operator_type, total_time_ns) in enumerate(sorted_operator_totals, start=1):
        tuuid = trace_writer.add_thread_descriptor(process_uuid, f"{index:0{width}d} {operator_type}")
        trace_writer.add_event(PerfettoEvent(tuuid, f"{operator_type}: {format_time_duration(total_time_ns)}",
                                             base_start, base_start + total_time_ns, "overview"))


def log_location_tracks(trace_writer, dispatch):
    """
    Add the "Compiler Input Locations" tracks, grouped as operator type -> location -> work unit
    type. Operator types and locations are sorted by their total time, and the locations without
    a known operator come last. Then add the operator type totals.
    """
    groups = _location_groups(dispatch)
    if not groups:
        return
    explicit = perfetto.TrackDescriptor.ChildTracksOrdering.EXPLICIT

    groups_per_type = {}
    for group in groups:
        groups_per_type.setdefault(group.operator_type, []).append(group)
    type_totals = {operator_type: sum(group.total_ns for group in type_groups)
                   for operator_type, type_groups in groups_per_type.items()}

    process_uuid = trace_writer.add_process_descriptor("Compiler Input Locations", explicit)
    ranked_types = sorted(type_totals, key=lambda name: (name == _OTHER_LOCATIONS, -type_totals[name], name))
    for type_rank, operator_type in enumerate(ranked_types):
        type_uuid = trace_writer.add_thread_descriptor(
            process_uuid, f"{operator_type} - {format_time_duration(type_totals[operator_type])}",
            order_rank=type_rank, child_ordering=explicit)
        type_groups = sorted(groups_per_type[operator_type], key=lambda group: (-group.total_ns, group.label))
        for rank, group in enumerate(type_groups):
            _emit_location_group(trace_writer, type_uuid, rank, group)

    _log_operator_type_totals(trace_writer, dispatch, groups)


def log_runtime_profile_data(trace_writer, dispatch: DispatchDebugInfo):
    """
    Log pre-parsed runtime profile data to Perfetto trace.
    
    Args:
        view_name: Name of the process/view
        trace_writer: PerfettoTraceWriter instance
        all_rows: List of AnnotatedWorkloadProfilingData instances for all the workloads in the profile
        overall_start: Start timestamp of the entire profile
        overall_end: End timestamp of the entire profile
    """

    # track names for each type of workload
    workunits_track_names = OrderedDict([
        (torq_debug_info.HalWorkUnitDebugInfo, ["HAL"]),
        (torq_debug_info.NssProgramWorkUnitDebugInfo, ["NSS Programs"]),
        (torq_debug_info.NssCfgWorkUnitDebugInfo, ["NSS CFG Tasks"]),
        (torq_debug_info.HostProgramWorkUnitDebugInfo, ["Host Programs"]),
        (torq_debug_info.HostCopyWorkUnitDebugInfo, ["Host Copy"]),
        (torq_debug_info.CdmaWorkUnitDebugInfo, ["CDMA"]),
        (torq_debug_info.DmaInWorkUnitDebugInfo, ["NDMA In"]),
        (torq_debug_info.DmaOutWorkUnitDebugInfo, ["NDMA Out"]),
        (torq_debug_info.SliceProgramWorkUnitDebugInfo, ["Slice 0", "Slice 1"]),
        (torq_debug_info.CssProgramWorkUnitDebugInfo, ["CSS"]),
    ])

    # stores track process and thread for each workunit type track
    workunits_tracks = {}

    host_process = trace_writer.add_process_descriptor("Host")
    npu_process = trace_writer.add_process_descriptor("NPU Task Breakdown")

    # create the tracks for all workunits
    for workunit_type, track_names in workunits_track_names.items():

        process = host_process

        if issubclass(workunit_type, torq_debug_info.NssManagedWorkUnitDebugInfo) or \
           workunit_type == torq_debug_info.NssCfgWorkUnitDebugInfo:
            process = npu_process

        for track_name in track_names:
            tuuid = trace_writer.add_thread_descriptor(process, track_name)
            workunits_tracks.setdefault(workunit_type, []).append(tuuid)


    # create tracks of all busy states
    busy_track_process = trace_writer.add_process_descriptor("NPU Task Measurements")


    busy_tracks = [
        PropertyTrack("dma_in_used", "DMA In Used"),
        PropertyTrack("dma_out_used", "DMA Out Used"),
        PropertyTrack("cdma_used", "CDMA Used"),
        PropertyTrack("slice_used", "Slice 0 Used", 0),
        PropertyTrack("slice_used", "Slice 1 Used", 1),
        PropertyTrack("css_used", "CSS Used")
    ]    
    
    for busy_track in busy_tracks:
        busy_track.uuid = trace_writer.add_thread_descriptor(busy_track_process, busy_track.thread_name)

    compile_placements = _compile_placements(dispatch)
    runtime_flows, compile_flow_ids = _assign_flow_ids(trace_writer, dispatch, compile_placements)
    
    # create an event for each workunit on the appropriate track with detailed information
    for workunit in dispatch.workunits:

        track_uuids = workunits_tracks.get(type(workunit))

        # Unsupported workunit type - skip        
        if track_uuids is None:
            continue

        try:
            tuuid = track_uuids[workunit.executor_instance_id]
        except IndexError:
            # Unsupported executor instance - skip
            continue

        # workunit with invalid timing - skip
        if workunit.total_time_ns is None:
            continue

        _emit_runtime_workunit(trace_writer, tuuid, workunit, runtime_flows.get(workunit, WorkUnitFlows()))

        # if this is a NSS program workunit, also log events on the busy tracks for each hardware component used
        # during this workunit's execution
        if isinstance(workunit, NssProgramWorkUnitDebugInfo):            
            
            for busy_track in busy_tracks:                

                # get whether a given NPU component is in use during this workunit
                usage_attr = getattr(workunit, busy_track.property_name)

                if busy_track.index is not None:
                    usage_attr = usage_attr[busy_track.index]

                # if it busy add an event to the corresponding busy track for the entire duration of the workunit
                if usage_attr:                    
                    busy_event = PerfettoEvent(busy_track.uuid, workunit.trace_name,
                                               workunit.start_time_ns, workunit.end_time_ns)
                    trace_writer.add_event(busy_event)

    _log_compile_time_traces(trace_writer, dispatch, compile_placements, compile_flow_ids)

    log_lram_usage(trace_writer, dispatch)

    stalls = log_slice_stalls(trace_writer, dispatch)

    log_operator_stalls(trace_writer, stalls)

    log_location_tracks(trace_writer, dispatch)


def render_overview_tracks(view_name, overview_data, trace_writer):
    """
    Render one overview track per metric (DMA, SLICE, IDLE, ...).

    The track name holds the metric total. The slices on the track are the real
    intervals of the metric, so you can see where in the timeline the hardware
    is idle, only moves data or only computes.
    """
    process_uuid = trace_writer.add_process_descriptor(
        f"{view_name} Overview", perfetto.TrackDescriptor.ChildTracksOrdering.EXPLICIT)

    metrics = overview_data['metrics']
    intervals = overview_data['intervals']

    overview_metrics = sorted((m for m in _METRICS if m.perfetto_thread_name is not None),
                              key=lambda m: m.perfetto_thread_name)

    for rank, m in enumerate(overview_metrics):
        metric_name = m.perfetto_name[0]
        metric = metrics.get(metric_name, {m.perfetto_name[1]: 0})
        thread_name = f"{m.perfetto_thread_name} | {format_overview_summary(m.perfetto_label, metric)}"
        tuuid = trace_writer.add_thread_descriptor(process_uuid, thread_name, order_rank=rank)

        for start, end in intervals.get(metric_name, []):
            trace_writer.add_event(PerfettoEvent(tuuid, m.perfetto_label, start, end, "overview"))


def get_dashboard_metrics():
    return [{'name': desc.dashboard_name, 'description': desc.dashboard_description, 'unit': desc.dashboard_unit} for desc in _METRICS]


def get_measurements(overview_data):
    results = {}
    for desc in _METRICS:
        metric_value = overview_data['metrics'].get(desc.perfetto_name[0], {}).get(desc.perfetto_name[1])
        if metric_value is not None:
            results[desc.dashboard_name] = metric_value

    return results


# =============================================================================
# MAIN CONVERSION LOGIC
# =============================================================================

def convert_to_perfetto(debug_dir, pb_path, frequency_mhz, view_name="Compile Profile"):
    """
    Convert a single compile profile CSV to Perfetto trace format.
    
    Args:
        debug_dir: Path to the debug directory for the model
        pb_path: Output path for Perfetto .pb file
        overview_data: Optional dict with pre-calculated metrics (contains 'metrics', 'overall_start', 'overall_end')
        original_mlir_file: Optional path to original MLIR file for operator extraction
        view_name: Name for the process in Perfetto (default: "Compile Profile")
        frequency_mhz: Chip frequency in MHz.
    
    Returns:
        Numeric measurements at the target clock; originals and normalized values match.
    """

    pb_path = Path(pb_path)
    pb_path.mkdir(parents=True, exist_ok=True)
    
    debug_info = DebugInfo(debug_dir)
    if not debug_info.dispatch_names:
        return {}

    for dispatch_name in debug_info.dispatch_names:

        trace_writer = PerfettoTraceWriter(pb_path / (dispatch_name + ".pb"))
    
        dispatch = debug_info.get_dispatch(dispatch_name)

        # compute the timestamps for action and tasks based on the cycle counts and clock frequency
        dispatch.infer_runtime_profile_from_cycles(frequency_mhz)

        try:
            log_runtime_profile_data(trace_writer, dispatch)

            overview_data = compute_compile_metrics(dispatch)
            
            render_overview_tracks(view_name, overview_data, trace_writer)

        finally:        
            trace_writer.close()
    
    return get_measurements(overview_data)


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Convert timeline CSV logs to Perfetto trace with overview analysis"
    )
    parser.add_argument("debug_dir", nargs='+', 
                       help="Path(s) to debug directory for the model")
    parser.add_argument("--pb", type=str, required=True,
                       help="Output path for Perfetto .pb file")
    parser.add_argument("--frequency-mhz", type=int, default=1000,
                       help="Chip frequency for compile estimates in MHz (default: 1 GHz)") 
    args = parser.parse_args()
    
    # Build profile dictionaries with filenames as keys
    compile_profile_logs = {}
    for path in args.debug_dir:
        filename = os.path.basename(path)
        compile_profile_logs[filename] = path
    
    # Calculate metrics for all profiles
    overview_map = {}
    
    #for name, path in compile_profile_logs.items():
    #    overview_data = calculate_compile_profile_metrics(path)
    #    overview_map[name] = overview_data
    #    print_metrics_summary(name, overview_data['metrics'])
    
    # Add numeric prefixes for process ordering in Perfetto UI only if multiple files
    total_files = len(compile_profile_logs)
    
    if total_files > 1:
        compile_prefixed = add_numeric_prefix(compile_profile_logs, start_index=1)
        
        # Update overview_map keys to match prefixed names
        new_overview_map = {}
        for i, (name, _) in enumerate(compile_profile_logs.items(), start=1):
            prefixed_name = f"{i:02d} {name}"
            new_overview_map[prefixed_name] = overview_map[name]
    else:
        compile_prefixed = compile_profile_logs
        new_overview_map = overview_map
    
    # Generate Perfetto trace
    convert_to_perfetto(
        debug_dir=args.debug_dir[0],
        pb_path=args.pb,
        frequency_mhz=args.frequency_mhz,
    )
    
    print(f"\n✓ Perfetto trace generated: {args.pb}")


if __name__ == "__main__":
    main()
