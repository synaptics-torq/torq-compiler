"""
Perfetto Trace Generator for Profile Analysis

Converts compile-time profiling CSV data into Perfetto trace format.
Includes hardware utilization analysis (DMA, SLICE, IDLE time) with visual overview tracks.

Usage:
    python perfetto_logger.py profile1.csv --pb output.pb
"""

try:
    from . import perfetto_api as perfetto
except ImportError:
    import perfetto_api as perfetto

import argparse
from collections import OrderedDict
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from torq.debug_info import DebugInfo, DispatchDebugInfo, HostCopyWorkUnitDebugInfo, HostProgramWorkUnitDebugInfo, HalWorkUnitDebugInfo, NssProgramWorkUnitDebugInfo, NssCfgWorkUnitDebugInfo, \
                                DmaInWorkUnitDebugInfo, DmaOutWorkUnitDebugInfo, CdmaWorkUnitDebugInfo, SliceProgramWorkUnitDebugInfo, CssProgramWorkUnitDebugInfo

import torq.debug_info


# =============================================================================
# CONFIGURATION
# =============================================================================

# Toggle to show shortened MLIR locations (filename only) vs full paths
SHOW_SHORT_LOCATIONS = True

# Clock frequency in MHz for cycle-to-time conversion
CLOCK_FREQ_MHZ = 800

def cycles_to_ns(cycles):
    """Convert cycles to nanoseconds assuming 800MHz clock."""
    return int(cycles * 1000 / CLOCK_FREQ_MHZ)

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
    correlation_id: Optional[str] = None


class PerfettoTraceWriter:
    """Manages Perfetto trace creation with processes and threads."""
    
    def __init__(self, filename):
        self.file = open(filename, "wb")
        self.trace = perfetto.Trace()
        self.global_uuid = 0

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

    def add_process_descriptor(self, process_name):
        """Register a new process in the trace."""
        
        puuid = self.get_next_uuid()        
        
        packet = self.trace.packet.add()
        track_des = perfetto.TrackDescriptor()        
        track_des.uuid = puuid
        track_des.name = process_name
        track_des.child_ordering = perfetto.TrackDescriptor.ChildTracksOrdering.CHRONOLOGICAL
        packet.track_descriptor.CopyFrom(track_des)

        return puuid

    def add_thread_descriptor(self, puuid, thread_name):
        """Register a new thread within a process."""

        puuid = puuid
        tuuid = self.get_next_uuid()
        
        packet = self.trace.packet.add()
        track_des = perfetto.TrackDescriptor()
        track_des.name = thread_name        
        track_des.uuid = tuuid
        track_des.parent_uuid = puuid
        packet.track_descriptor.CopyFrom(track_des)

        return tuuid

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

        # we need to upgrade perfetto
        #if event.correlation_id is not None:
        #    track_event.correlation_id = event.correlation_id

        packet.timestamp = time
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
    MetricDescription(('OVERALL', 'time'), '15 OVERALL', 'OVERALL', 'total_duration', 'Total execution duration of the workload', 'ns'),
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


def compute_compile_metrics(dispatch: DispatchDebugInfo):
    """
    Compute hardware utilization metrics from parsed compile profile intervals.
    
    Args:
        dma_intervals: List of (start, end) tuples for DMA
        cdma_dtcm_intervals: List of (start, end) tuples for CDMA DTCM
        cdma_itcm_intervals: List of (start, end) tuples for CDMA ITCM
        slice_intervals: List of (start, end) tuples for Slice
        css_intervals: List of (start, end) tuples for CSS
        overall_start: Start timestamp
        overall_end: End timestamp
        
    Returns:
        Dictionary containing metrics, overall_start, overall_end
    """

    overall_end = dispatch.end_time_ns
    overall_start = dispatch.start_time_ns

    dma_intervals = []
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

        if isinstance(workunit, DmaOutWorkUnitDebugInfo) or \
              isinstance(workunit, DmaInWorkUnitDebugInfo):
            dma_intervals.append((start_time_ns, end_time_ns))

            if isinstance(workunit, DmaInWorkUnitDebugInfo):
                dma_in_intervals.append((start_time_ns, end_time_ns))
            else:
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

    # Merge overlapping intervals
    merged_dma = merge_intervals(dma_intervals)
    merged_dma_in = merge_intervals(dma_in_intervals)
    merged_dma_out = merge_intervals(dma_out_intervals)
    merged_slice = merge_intervals(slice_intervals)
    merged_cdma_dtcm = merge_intervals(cdma_dtcm_intervals)
    merged_cdma_itcm = merge_intervals(cdma_itcm_intervals)
    merged_cdma_all = merge_intervals(cdma_dtcm_intervals + cdma_itcm_intervals)
    merged_css = merge_intervals(css_intervals)
    
    # Combine DMA and CDMA for "DMA" statistics
    # Use merge_intervals to calculate the Union (handling any overlaps between DMA and CDMA)
    merged_dma_combined = merge_intervals(dma_intervals + cdma_dtcm_intervals + cdma_itcm_intervals)
    
    # Combine SLICE and CSS for "COMPUTE" statistics
    # Use merge_intervals to calculate the Union (handling any overlaps between Slice and CSS)
    merged_compute_combined = merge_intervals(slice_intervals + css_intervals)
    
    # Calculate DMA-only time (DMA/CDMA active, SLICE/CSS inactive)
    # This subtracts any time where Compute is active from the DMA Combined time
    dma_only_intervals = subtract_intervals(merged_dma_combined, merged_compute_combined)
    total_dma_only = sum(end - start for start, end in dma_only_intervals)
    
    # Calculate COMPUTE-only time (SLICE/CSS active, DMA/CDMA inactive)
    compute_only_intervals = subtract_intervals(merged_compute_combined, merged_dma_combined)
    total_compute_only = sum(end - start for start, end in compute_only_intervals)
    
    # Calculate DMA+COMPUTE overlap time
    # This finds the time where BOTH DMA/CDMA and Compute are active
    overlap_intervals = intersect_intervals(merged_dma_combined, merged_compute_combined)
    total_overlap = sum(end - start for start, end in overlap_intervals)
    
    # Calculate totals
    total_dma = sum(end - start for start, end in merged_dma)
    total_dma_in = sum(end - start for start, end in merged_dma_in)
    total_dma_out = sum(end - start for start, end in merged_dma_out)
    total_slice = sum(end - start for start, end in merged_slice)
    total_cdma_dtcm = sum(end - start for start, end in merged_cdma_dtcm)
    total_cdma_itcm = sum(end - start for start, end in merged_cdma_itcm)
    total_cdma_all = sum(end - start for start, end in merged_cdma_all)
    total_css = sum(end - start for start, end in merged_css)
    
    total_dma_combined = sum(end - start for start, end in merged_dma_combined)
    total_compute_combined = sum(end - start for start, end in merged_compute_combined)
    
    overall_time = (overall_end - overall_start) if (overall_start is not None and overall_end is not None) else 0
    
    # Calculate BUSY time (union of DMA/CDMA and SLICE/CSS)
    busy_intervals = merge_intervals(merged_dma_combined + merged_compute_combined)
    busy_time = sum(end - start for start, end in busy_intervals)
    
    # Calculate IDLE time
    idle_time = overall_time - busy_time if overall_time >= busy_time else 0
    
    # Calculate percentages
    def calc_percent(time):
        return (time / overall_time * 100) if overall_time else 0
    
    metrics = {
        'DMA': {'time': total_dma, 'percent': calc_percent(total_dma)},
        'DMA_IN': {'time': total_dma_in, 'percent': calc_percent(total_dma_in)},
        'DMA_OUT': {'time': total_dma_out, 'percent': calc_percent(total_dma_out)},
        'SLICE': {'time': total_slice, 'percent': calc_percent(total_slice)},
        'CDMA_DTCM': {'time': total_cdma_dtcm, 'percent': calc_percent(total_cdma_dtcm)},
        'CDMA_ITCM': {'time': total_cdma_itcm, 'percent': calc_percent(total_cdma_itcm)},
        'CDMA': {'time': total_cdma_all, 'percent': calc_percent(total_cdma_all)},
        'CSS': {'time': total_css, 'percent': calc_percent(total_css)},
        'DMA_COMBINED': {'time': total_dma_combined, 'percent': calc_percent(total_dma_combined)},
        'COMPUTE_COMBINED': {'time': total_compute_combined, 'percent': calc_percent(total_compute_combined)},
        'IDLE': {'time': idle_time, 'percent': calc_percent(idle_time)},
        'DMA_ONLY': {'time': total_dma_only, 'percent': calc_percent(total_dma_only)},
        'COMPUTE_ONLY': {'time': total_compute_only, 'percent': calc_percent(total_compute_only)},
        'DMA_COMPUTE_OVERLAP': {'time': total_overlap, 'percent': calc_percent(total_overlap)},
        'OVERALL': {'time': overall_time}
    }
    
    return {
        'metrics': metrics,
        'overall_start': overall_start,
        'overall_end': overall_end
    }



def compute_runtime_metrics(dispatch: DispatchDebugInfo):
    """
    Compute utilization metrics from parsed runtime profile rows.
    
    Args:
        all_rows: List of event tuples
        overall_start: Start timestamp
        overall_end: End timestamp
        
    Returns:
        Dictionary containing metrics, overall_start, overall_end
    """
    
    overall_start = dispatch.start_time_ns
    overall_end = dispatch.end_time_ns

    dma_intervals = []
    dma_in_intervals = []
    dma_out_intervals = []
    slice_intervals = []
    slice_0_intervals = []
    slice_1_intervals = []
    cdma_intervals = []
    css_intervals = []
    hal_intervals = []
    host_copy_intervals = []
    host_intervals = []
    
    for workunit in dispatch.workunits:
        
        start_time_ns = workunit.start_time_ns
        end_time_ns = workunit.end_time_ns

        if start_time_ns is None or end_time_ns is None:
            continue
        
        if isinstance(workunit, NssProgramWorkUnitDebugInfo):

            if start_time_ns is None or end_time_ns is None:
                raise ValueError(f"WorkUnit {workunit} is missing start or end time.")

            if workunit.dma_in_used or workunit.dma_out_used:
                dma_intervals.append((start_time_ns, end_time_ns))

                if workunit.dma_in_used:
                    dma_in_intervals.append((start_time_ns, end_time_ns))

                if workunit.dma_out_used:
                    dma_out_intervals.append((start_time_ns, end_time_ns))

            if workunit.cdma_used:
                cdma_intervals.append((start_time_ns, end_time_ns))
            
            if workunit.slice_used[0] or workunit.slice_used[1]:
                slice_intervals.append((start_time_ns, end_time_ns))
                
            if workunit.slice_used[0]:
                slice_0_intervals.append((start_time_ns, end_time_ns))
                
            if workunit.slice_used[1]:
                slice_1_intervals.append((start_time_ns, end_time_ns))

            if workunit.css_used:
                css_intervals.append((start_time_ns, end_time_ns))
    
        elif isinstance(workunit, HostCopyWorkUnitDebugInfo):        

            if start_time_ns is None or end_time_ns is None:
                raise ValueError(f"WorkUnit {workunit} is missing start or end time.")

            host_copy_intervals.append((start_time_ns, end_time_ns))

        elif isinstance(workunit, HostProgramWorkUnitDebugInfo):

            if start_time_ns is None or end_time_ns is None:
                raise ValueError(f"WorkUnit {workunit} is missing start or end time.")

            host_intervals.append((start_time_ns, end_time_ns))

        elif isinstance(workunit, HalWorkUnitDebugInfo):

            if start_time_ns is None or end_time_ns is None:
                raise ValueError(f"WorkUnit {workunit} is missing start or end time.")

            hal_intervals.append((start_time_ns, end_time_ns))
    
    # Merge overlapping intervals
    merged_dma = merge_intervals(dma_intervals)
    merged_dma_in = merge_intervals(dma_in_intervals)
    merged_dma_out = merge_intervals(dma_out_intervals)
    merged_cdma = merge_intervals(cdma_intervals)
    merged_slice = merge_intervals(slice_intervals)
    merged_slice_0 = merge_intervals(slice_0_intervals)
    merged_slice_1 = merge_intervals(slice_1_intervals)
    merged_css = merge_intervals(css_intervals)
    merged_hal = merge_intervals(hal_intervals)
    merged_host_copy = merge_intervals(host_copy_intervals)
    merged_host = merge_intervals(host_intervals)
    
    merged_compute_combined = merge_intervals(slice_0_intervals + slice_1_intervals + css_intervals)
    
    # Combined DMA = DMA + CDMA
    merged_dma_combined = merge_intervals(dma_intervals + cdma_intervals)

    # Calculate exclusive and overlap times
    dma_only_intervals = subtract_intervals(merged_dma_combined, merged_compute_combined)
    compute_only_intervals = subtract_intervals(merged_compute_combined, merged_dma_combined)
    overlap_intervals = intersect_intervals(merged_dma_combined, merged_compute_combined)
    
    # Calculate totals
    total_dma = sum(end - start for start, end in merged_dma)
    total_dma_in = sum(end - start for start, end in merged_dma_in)
    total_dma_out = sum(end - start for start, end in merged_dma_out)
    total_cdma = sum(end - start for start, end in merged_cdma)
    total_slice = sum(end - start for start, end in merged_slice)
    total_slice_0 = sum(end - start for start, end in merged_slice_0)
    total_slice_1 = sum(end - start for start, end in merged_slice_1)
    total_css = sum(end - start for start, end in merged_css)
    total_hal = sum(end - start for start, end in merged_hal)
    total_compute_combined = sum(end - start for start, end in merged_compute_combined)
    total_dma_only = sum(end - start for start, end in dma_only_intervals)
    total_compute_only = sum(end - start for start, end in compute_only_intervals)
    total_overlap = sum(end - start for start, end in overlap_intervals)
    total_dma_combined = sum(end - start for start, end in merged_dma_combined)
    total_host_copy = sum(end - start for start, end in merged_host_copy)
    total_host = sum(end - start for start, end in merged_host)
    
    overall_time = (overall_end - overall_start) if (overall_start is not None and overall_end is not None) else 0
    
    # Calculate BUSY time (union of DMA+CDMA, SLICE+CSS, HAL, HOST, and HOST_COPY)
    busy_intervals = merge_intervals(merged_dma_combined + merged_compute_combined + merged_hal + host_copy_intervals + host_intervals)
    busy_time = sum(end - start for start, end in busy_intervals)
    
    # Calculate IDLE time
    idle_time = overall_time - busy_time if overall_time >= busy_time else 0
    
    def calc_percent(time):
        return (time / overall_time * 100) if overall_time else 0
    
    metrics = {
        'DMA': {'time': total_dma, 'percent': calc_percent(total_dma)},
        'DMA_IN': {'time': total_dma_in, 'percent': calc_percent(total_dma_in)},
        'DMA_OUT': {'time': total_dma_out, 'percent': calc_percent(total_dma_out)},
        'SLICE': {'time': total_slice, 'percent': calc_percent(total_slice)},
        'SLICE_0': {'time': total_slice_0, 'percent': calc_percent(total_slice_0)},
        'SLICE_1': {'time': total_slice_1, 'percent': calc_percent(total_slice_1)},
        'CDMA': {'time': total_cdma, 'percent': calc_percent(total_cdma)}, 
        'CSS': {'time': total_css, 'percent': calc_percent(total_css)},
        'DMA_COMBINED': {'time': total_dma_combined, 'percent': calc_percent(total_dma_combined)},
        'COMPUTE_COMBINED': {'time': total_compute_combined, 'percent': calc_percent(total_compute_combined)},
        'IDLE': {'time': idle_time, 'percent': calc_percent(idle_time)},
        'DMA_ONLY': {'time': total_dma_only, 'percent': calc_percent(total_dma_only)},
        'COMPUTE_ONLY': {'time': total_compute_only, 'percent': calc_percent(total_compute_only)},
        'DMA_COMPUTE_OVERLAP': {'time': total_overlap, 'percent': calc_percent(total_overlap)},
        'OVERALL': {'time': overall_time},
        'HAL': {'time': total_hal, 'percent': calc_percent(total_hal)},
        'HOST_COPY': {'time': total_host_copy, 'percent': calc_percent(total_host_copy)},
        'HOST': {'time': total_host, 'percent': calc_percent(total_host)}
    }
    
    return {
        'metrics': metrics,
        'overall_start': overall_start,
        'overall_end': overall_end
    }


def print_metrics_summary(profile_name, metrics):
    """Print formatted metrics summary to console."""
    print(f"\n=== Overview ({profile_name}) ===")
    order = [
        'OVERALL', 'DMA_COMPUTE_OVERLAP', 'SLICE', 'SLICE_0', 'SLICE_1', 
        'COMPUTE_COMBINED', 'CSS', 'COMPUTE_ONLY', 'DMA_COMBINED', 
        'DMA_ONLY', 'DMA', 'CDMA', 'IDLE'
    ]
    for key in order:
        if key in metrics:
            metric = metrics[key]
            if 'percent' in metric:
                print(f"{key} time: {metric['time']} ({metric['percent']:.2f}%)")
            else:
                print(f"{key} time: {metric['time']}")


# =============================================================================
# PERFETTO TRACE RENDERING
# =============================================================================


def create_overview_event(tuuid, label, metric, base_start):
    """
    Create a Perfetto event for an overview track.
    
    Returns:
        PerfettoEvent instance or None if duration is invalid
    """
    duration = metric['time']
    if duration < 0:
        return None
    
    start = base_start
    end = base_start + duration
    
    # Build name with formatted duration and percent (if available)
    name = f"{label}: {format_time_duration(duration)}"
    if 'percent' in metric:
        name += f" ({metric['percent']:.2f}%)"

    return PerfettoEvent(tuuid, name, start, end, "overview")


@dataclass
class PropertyTrack:
    property_name: str
    thread_name: str
    index: Optional[int] = None
    uuid: Optional[str] = None


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
        (torq.debug_info.HalWorkUnitDebugInfo, ["HAL"]),
        (torq.debug_info.NssProgramWorkUnitDebugInfo, ["NSS Programs"]),
        (torq.debug_info.NssCfgWorkUnitDebugInfo, ["NSS CFG Tasks"]),
        (torq.debug_info.HostProgramWorkUnitDebugInfo, ["Host Programs"]),
        (torq.debug_info.HostCopyWorkUnitDebugInfo, ["Host Copy"]),
        (torq.debug_info.CdmaWorkUnitDebugInfo, ["CDMA"]),
        (torq.debug_info.DmaInWorkUnitDebugInfo, ["NDMA In"]),
        (torq.debug_info.DmaOutWorkUnitDebugInfo, ["NDMA Out"]),
        (torq.debug_info.SliceProgramWorkUnitDebugInfo, ["Slice 0", "Slice 1"]),
        (torq.debug_info.CssProgramWorkUnitDebugInfo, ["CSS"]),
    ])

    # stores track process and thread for each workunit type track
    workunits_tracks = {}

    host_process = trace_writer.add_process_descriptor("Host")
    npu_process = trace_writer.add_process_descriptor("NPU")

    # create the tracks for all workunits
    for workunit_type, track_names in workunits_track_names.items():

        process = host_process

        if issubclass(workunit_type, torq.debug_info.NssManagedWorkUnitDebugInfo) or \
           workunit_type == torq.debug_info.NssCfgWorkUnitDebugInfo:
            process = npu_process

        for track_name in track_names:
            tuuid = trace_writer.add_thread_descriptor(process, track_name)
            workunits_tracks.setdefault(workunit_type, []).append(tuuid)


    # create tracks of all busy states
    busy_track_process = trace_writer.add_process_descriptor("NPU Usage")


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

        start_time_ns = workunit.start_time_ns
        end_time_ns = workunit.end_time_ns
        duration_ns = workunit.total_time_ns

        # workunit with invalid timing - skip
        if duration_ns is None:
            continue

        # create label for event                
        details = workunit.pretty_print

        ready_time_ns = workunit.ready_time_ns

        if ready_time_ns is not None:
            event = PerfettoEvent(tuuid, details, start_time_ns, ready_time_ns)
            trace_writer.add_event(event)
            event = PerfettoEvent(tuuid, "WAITING", ready_time_ns, end_time_ns)
            trace_writer.add_event(event)
        else:
            event = PerfettoEvent(tuuid, details, start_time_ns, end_time_ns)
            trace_writer.add_event(event)

        # if this is a NSS program workunit, also log events on the busy tracks for each hardware component used
        # during this workunit's execution
        if isinstance(workunit, NssProgramWorkUnitDebugInfo):            
            
            for busy_track in busy_tracks:                

                # get whether a given NPU component is in use during this workunit
                usage_attr = getattr(workunit, busy_track.property_name)

                if busy_track.index:
                    usage_attr = usage_attr[busy_track.index]

                # if it busy add an event to the corresponding busy track for the entire duration of the workunit
                if usage_attr:                    
                    busy_event = PerfettoEvent(busy_track.uuid, details, start_time_ns, end_time_ns)
                    trace_writer.add_event(busy_event)

    original_lines_process = trace_writer.add_process_descriptor("Compiler Input Locations")

    for original_loc_info in dispatch.original_locations:

        original_line_desc = f"{original_loc_info.location.file}:{original_loc_info.location.line}:{original_loc_info.location.col}"        

        # we create a new track for each workunit type so that we can have overlapping
        # slices. Perfetto will group all these tracks that have the same name.
        workunits_tracks = {}

        for workunit in original_loc_info.workunits:

            details = workunit.pretty_print
            start_time_ns = workunit.start_time_ns
            end_time_ns = workunit.end_time_ns

            if start_time_ns is None or end_time_ns is None:
                continue

            workunit_track = workunits_tracks.get(type(workunit))

            if workunit_track is None:
                workunit_track = trace_writer.add_thread_descriptor(original_lines_process, original_line_desc)
                workunits_tracks[type(workunit)] = workunit_track
            
            event = PerfettoEvent(workunit_track, details, start_time_ns, end_time_ns, correlation_id=type(workunit).__name__)            
            trace_writer.add_event(event)
        


def render_overview_tracks(view_name, overall_start, metrics, trace_writer):
    """
    Render summary overview tracks showing total DMA, SLICE, IDLE time.
    
    Creates a separate process with overview tracks in fixed order.
    """
    process_uuid= trace_writer.add_process_descriptor(f"{view_name} Overview")

    base_start = overall_start if overall_start is not None else 0
        
    overview_tracks = [(m.perfetto_thread_name, m.perfetto_label, metrics.get(m.perfetto_name[0], {m.perfetto_name[1]: 0})) 
                        for m in _METRICS if m.perfetto_thread_name is not None]
    
    overview_tracks = sorted(overview_tracks, key=lambda x: x[0])

    for thread_name, label, metric in overview_tracks:
        tuuid = trace_writer.add_thread_descriptor(process_uuid, thread_name)
        perfetto_event = create_overview_event(tuuid, label, metric, base_start)
        if perfetto_event:
            trace_writer.add_event(perfetto_event)


def get_dashboard_metrics():
    return [{'name': desc.dashboard_name, 'description': desc.dashboard_description, 'unit': desc.dashboard_unit} for desc in _METRICS]


def get_measurements(overview_data):
    results = []
    for desc in _METRICS:
        metric_value = overview_data['metrics'].get(desc.perfetto_name[0], {}).get(desc.perfetto_name[1])
        if metric_value is not None:
            results.append({'metric': desc.dashboard_name, 'value': metric_value})

    return results


# =============================================================================
# MAIN CONVERSION LOGIC
# =============================================================================

def convert_to_perfetto(debug_dir, pb_path, view_name="Compile Profile"):
    """
    Convert a single compile profile CSV to Perfetto trace format.
    
    Args:
        debug_dir: Path to the debug directory for the model
        pb_path: Output path for Perfetto .pb file
        overview_data: Optional dict with pre-calculated metrics (contains 'metrics', 'overall_start', 'overall_end')
        original_mlir_file: Optional path to original MLIR file for operator extraction
        view_name: Name for the process in Perfetto (default: "Compile Profile")
    
    Returns:
        Path to generated Perfetto trace file
    """

    pb_path = Path(pb_path)
    pb_path.mkdir(parents=True, exist_ok=True)
    
    debug_info = DebugInfo(debug_dir)

    for dispatch_name in debug_info.dispatch_names:

        trace_writer = PerfettoTraceWriter(pb_path / (dispatch_name + ".pb"))
    
        dispatch_name = debug_info.dispatch_names[0]
        dispatch = debug_info.get_dispatch(dispatch_name)

        # print(f"Processing dispatch: {dispatch_name} with {len(dispatch.workunits)} workunits")

        # compute the timestamps for action and tasks based on the cycle counts and clock frequency
        dispatch.infer_runtime_profile_from_cycles()

        try:
            log_runtime_profile_data(trace_writer, dispatch)

            overview_data = compute_compile_metrics(dispatch)
            
            render_overview_tracks(view_name, overview_data['overall_start'], 
                                overview_data['metrics'], trace_writer)

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
        pb_path=args.pb#,
        #overview_map=new_overview_map
    )
    
    print(f"\n✓ Perfetto trace generated: {args.pb}")


if __name__ == "__main__":
    main()
