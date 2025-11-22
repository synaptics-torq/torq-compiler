"""
Perfetto Trace Generator for Profile Analysis

Converts compile-time and runtime profiling CSV data into Perfetto trace format.
Includes hardware utilization analysis (DMA, SLICE, IDLE time) with visual overview tracks.

Usage:
    python perfetto_logger.py --compile_profile profile1.csv --runtime_profile runtime.csv --pb output.pb
"""

import perfetto_api as perfetto
import csv
import argparse
import os
import re
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

# Toggle to show shortened MLIR locations (filename only) vs full paths
SHOW_SHORT_LOCATIONS = True


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class PerfettoEvent:
    """Represents a single trace event with timing information."""
    
    def __init__(self, process, thread, function, start_time, end_time):
        self.process = process
        self.thread = thread
        self.function = function
        self.start_time = start_time
        self.end_time = end_time


class PerfettoTraceWriter:
    """Manages Perfetto trace creation with processes and threads."""
    
    def __init__(self, filename):
        self.file = open(filename, "wb")
        self.trace = perfetto.Trace()
        self.process_to_uuid = {}
        self.process_to_thread = {}
        self.global_uuid = 0

    def __del__(self):
        self.flush()
    
    def flush(self):
        """Write accumulated trace packets to file and clear buffer."""
        self.file.write(self.trace.SerializeToString())
        self.trace.Clear()

    def get_next_uuid(self):
        """Generate unique ID for processes and threads."""
        self.global_uuid += 1
        return self.global_uuid

    def add_process_descriptor(self, process_name):
        """Register a new process in the trace."""
        if process_name not in self.process_to_uuid:
            puuid = self.get_next_uuid()
            self.process_to_uuid[process_name] = puuid
            
            packet = self.trace.packet.add()
            track_des = perfetto.TrackDescriptor()
            pd = perfetto.ProcessDescriptor()
            pd.pid = puuid
            pd.process_name = process_name
            track_des.process.CopyFrom(pd)
            track_des.uuid = puuid
            track_des.child_ordering = perfetto.TrackDescriptor.ChildTracksOrdering.LEXICOGRAPHIC
            packet.track_descriptor.CopyFrom(track_des)

    def add_thread_descriptor(self, process_name, thread_name):
        """Register a new thread within a process."""
        if process_name not in self.process_to_thread:
            self.process_to_thread[process_name] = {}
            self.add_process_descriptor(process_name)

        if thread_name not in self.process_to_thread[process_name]:
            thread_to_uuid = self.process_to_thread[process_name]
            puuid = self.process_to_uuid[process_name]
            tuuid = self.get_next_uuid()

            thread_to_uuid[thread_name] = tuuid
            packet = self.trace.packet.add()
            track_des = perfetto.TrackDescriptor()
            thd = perfetto.ThreadDescriptor()
            thd.pid = puuid
            thd.tid = tuuid
            thd.thread_name = thread_name
            track_des.thread.CopyFrom(thd)
            track_des.uuid = tuuid
            packet.track_descriptor.CopyFrom(track_des)

    def add_event(self, event):
        """Add a trace event (slice) to the timeline."""
        self.add_thread_descriptor(event.process, event.thread)
        self.set_track_event(event.start_time, perfetto.TrackEvent.Type.TYPE_SLICE_BEGIN, event)
        self.set_track_event(event.end_time, perfetto.TrackEvent.Type.TYPE_SLICE_END, event)

        # Periodically flush to avoid memory buildup
        if len(self.trace.packet) > 1024:
            self.flush()
    
    def set_track_event(self, time, event_type, event):
        """Create a Perfetto track event packet."""
        thread_to_uuid = self.process_to_thread[event.process]
        tuuid = thread_to_uuid[event.thread]

        packet = self.trace.packet.add()
        track_event = perfetto.TrackEvent()
        
        track_event.type = event_type
        track_event.track_uuid = tuuid
        track_event.name = event.function
        
        # Color hint for overview tracks
        if 'OVERVIEW' in event.thread:
            track_event.categories.append('overview')

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

def shorten_mlir_location(location):
    """
    Shorten MLIR file location to just filename and line/col info.
    
    Example:
        loc(callsite("/path/to/file.mlir":2:3)) -> file.mlir:2:3
        loc(callsite("/path/to/file.mlir":18:11 at "/path/to/file.mlir":2:3)) -> file.mlir:18:11
    """
    if not location or not SHOW_SHORT_LOCATIONS:
        return location
    
    # Remove loc(...) and callsite(...) wrappers
    normalized = location.replace('\\', '/').strip()
    
    # Handle loc(callsite("path":line:col at "path":line:col)) - extract the FIRST path (before "at")
    match = re.search(r'callsite\("?([^"]+)"?:(\d+):(\d+)', normalized)
    if match:
        path = match.group(1)
        line = match.group(2)
        col = match.group(3)
        filename = os.path.basename(path)
        return f"{filename}:{line}:{col}"
    
    # Handle simpler formats: loc("path":line:col)
    match = re.search(r'"?([^"]+\.mlir)"?:(\d+):(\d+)', normalized)
    if match:
        path = match.group(1)
        line = match.group(2)
        col = match.group(3)
        filename = os.path.basename(path)
        return f"{filename}:{line}:{col}"
    
    # Fallback: try basic cleanup
    normalized = re.sub(r'^loc\(callsite\("?([^"]+)"?\)\)$', r'\1', normalized)
    normalized = re.sub(r'^loc\("?([^"]+)"?\)$', r'\1', normalized)
    normalized = normalized.replace('"', '').rstrip(')')
    
    trailing = ''
    temp = normalized
    
    # Extract trailing :line:col groups
    while True:
        match = re.match(r'^(.*):(\d+)$', temp)
        if match:
            trailing = f":{match.group(2)}{trailing}"
            temp = match.group(1)
        else:
            break
    
    base = os.path.basename(temp) if temp else temp
    return base + trailing if base else location


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
# CSV PARSING
# =============================================================================

def parse_compile_profile_csv(csv_path):
    """
    Parse compile profile CSV and extract event intervals.
    
    CSV Format: event, start, end, src_line_id, etype
    
    Returns:
        Tuple of (dma_intervals, slice_intervals, overall_start, overall_end, all_rows)
    """
    dma_intervals = []
    slice_intervals = []
    all_rows = []
    overall_start = None
    overall_end = None
    
    with open(csv_path, newline='') as fp:
        reader = csv.reader(fp)
        for row in reader:
            if len(row) != 5:
                continue
            
            event, start, end, src_line_id, etype = row
            
            try:
                start_time = int(start)
                end_time = int(end)
            except ValueError:
                continue
            
            # Track overall time span
            if overall_start is None or start_time < overall_start:
                overall_start = start_time
            if overall_end is None or end_time > overall_end:
                overall_end = end_time
            
            # Categorize by event type
            if event.startswith('DI') or event.startswith('DO'):
                dma_intervals.append((start_time, end_time))
            elif event.startswith('S'):
                slice_intervals.append((start_time, end_time))
            
            all_rows.append((event, start_time, end_time, src_line_id, etype))
    
    return dma_intervals, slice_intervals, overall_start, overall_end, all_rows


def parse_runtime_profile_csv(csv_path):
    """
    Parse runtime profile CSV and extract dispatch intervals.
    
    CSV Format: dispatch_id;time_since_open;time_since_start;mlir_loc
    
    Returns:
        Tuple of (dispatch_intervals, overall_start, overall_end, all_rows)
    """
    dispatch_intervals = []
    all_rows = []
    overall_start = None
    overall_end = None
    
    with open(csv_path, newline='') as fp:
        reader = csv.reader(fp, delimiter=';')
        for row in reader:
            if len(row) == 0 or 'id' in row:
                continue
            if len(row) != 4:
                continue
            
            dispatch_id, time_since_open, time_since_start, mlir_loc = row
            
            try:
                start_time = int(time_since_open)
                duration = int(time_since_start)
                end_time = start_time + duration
            except ValueError:
                continue
            
            if overall_start is None or start_time < overall_start:
                overall_start = start_time
            if overall_end is None or end_time > overall_end:
                overall_end = end_time
            
            dispatch_intervals.append((start_time, end_time))
            all_rows.append((dispatch_id, start_time, end_time, mlir_loc))
    
    return dispatch_intervals, overall_start, overall_end, all_rows


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_compile_profile_metrics(csv_path):
    """
    Calculate hardware utilization metrics for compile profile.
    
    Returns:
        Dictionary containing:
        - metrics: Dict with DMA, SLICE, IDLE, DMA_ONLY, DMA_SLICE_OVERLAP, OVERALL
        - overall_start: Start timestamp
        - overall_end: End timestamp
    """
    dma_intervals, slice_intervals, overall_start, overall_end, _ = parse_compile_profile_csv(csv_path)
    
    # Merge overlapping intervals
    merged_dma = merge_intervals(dma_intervals)
    merged_slice = merge_intervals(slice_intervals)
    
    # Calculate DMA-only time (DMA active, SLICE inactive)
    dma_only_intervals = subtract_intervals(merged_dma, merged_slice)
    total_dma_only = sum(end - start for start, end in dma_only_intervals)
    
    # Calculate DMA+SLICE overlap time
    overlap_intervals = intersect_intervals(merged_dma, merged_slice)
    total_overlap = sum(end - start for start, end in overlap_intervals)
    
    # Calculate totals
    total_dma = sum(end - start for start, end in merged_dma)
    total_slice = sum(end - start for start, end in merged_slice)
    overall_time = (overall_end - overall_start) if (overall_start is not None and overall_end is not None) else 0
    
    # Calculate BUSY time (union of DMA and SLICE)
    busy_intervals = merge_intervals(merged_dma + merged_slice)
    busy_time = sum(end - start for start, end in busy_intervals)
    
    # Calculate IDLE time
    idle_time = overall_time - busy_time if overall_time >= busy_time else 0
    
    # Calculate percentages
    def calc_percent(time):
        return (time / overall_time * 100) if overall_time else 0
    
    metrics = {
        'DMA': {'time': total_dma, 'percent': calc_percent(total_dma)},
        'SLICE': {'time': total_slice, 'percent': calc_percent(total_slice)},
        'IDLE': {'time': idle_time, 'percent': calc_percent(idle_time)},
        'DMA_ONLY': {'time': total_dma_only, 'percent': calc_percent(total_dma_only)},
        'DMA_SLICE_OVERLAP': {'time': total_overlap, 'percent': calc_percent(total_overlap)},
        'OVERALL': {'time': overall_time}
    }
    
    return {
        'metrics': metrics,
        'overall_start': overall_start,
        'overall_end': overall_end
    }


def calculate_runtime_profile_metrics(csv_path):
    """
    Calculate utilization metrics for runtime profile.
    
    Runtime profiles only track dispatch execution (BUSY vs IDLE).
    
    Returns:
        Dictionary containing metrics, overall_start, overall_end
    """
    dispatch_intervals, overall_start, overall_end, _ = parse_runtime_profile_csv(csv_path)
    
    # Merge overlapping dispatches
    merged = merge_intervals(dispatch_intervals)
    busy_time = sum(end - start for start, end in merged)
    
    overall_time = (overall_end - overall_start) if (overall_start is not None and overall_end is not None) else 0
    idle_time = overall_time - busy_time if overall_time >= busy_time else 0
    
    def calc_percent(time):
        return (time / overall_time * 100) if overall_time else 0
    
    metrics = {
        'DMA': {'time': 0, 'percent': 0},
        'SLICE': {'time': 0, 'percent': 0},
        'IDLE': {'time': idle_time, 'percent': calc_percent(idle_time)},
        'DMA_ONLY': {'time': 0, 'percent': 0},
        'DMA_SLICE_OVERLAP': {'time': 0, 'percent': 0},
        'OVERALL': {'time': overall_time}
    }
    
    return {
        'metrics': metrics,
        'overall_start': overall_start,
        'overall_end': overall_end
    }


def print_metrics_summary(profile_name, metrics):
    """Print formatted metrics summary to console."""
    print(f"\n=== Overview ({profile_name}) ===")
    for key in ['DMA', 'SLICE', 'DMA_ONLY', 'DMA_SLICE_OVERLAP', 'IDLE', 'OVERALL']:
        metric = metrics[key]
        if 'percent' in metric:
            print(f"{key} time: {metric['time']} ({metric['percent']:.2f}%)")
        else:
            print(f"{key} time: {metric['time']}")


# =============================================================================
# PERFETTO TRACE RENDERING
# =============================================================================

def create_compile_profile_event(view_name, event, start_time, end_time, 
                                 src_line_id, etype, overall_time):
    """
    Create a Perfetto event for a compile profile entry.
    
    Returns:
        PerfettoEvent instance
    """
    # Categorize event
    event_category = "UNKNOWN"
    if event.startswith("DI"):
        event_category = "DMA IN"
    elif event.startswith("DO"):
        event_category = "DMA OUT"
    elif event.startswith("S"):
        event_category = "SLICE"
        try:
            slice_num = int(event.replace("S", ""))
            slice_num = slice_num % 2  # Hardware has 2 slices
            event_category = f"{event_category} {slice_num}"
        except Exception:
            event_category = f"{event_category} UNKNOWN"
    
    # Build event details with duration and percentage
    duration = end_time - start_time
    percent = (duration / overall_time * 100) if overall_time else 0
    short_loc = shorten_mlir_location(src_line_id)
    details = f"{etype}, {event}, MLIR - {short_loc}, dur={duration} ({percent:.2f}%)"
    
    return PerfettoEvent(view_name, event_category, details, start_time, end_time)


def create_runtime_profile_event(view_name, dispatch_id, start_time, end_time, 
                                 mlir_loc, overall_time):
    """
    Create a Perfetto event for a runtime profile entry.
    
    Returns:
        PerfettoEvent instance
    """
    duration = end_time - start_time
    percent = (duration / overall_time * 100) if overall_time else 0
    short_loc = shorten_mlir_location(mlir_loc)
    details = f"{short_loc}, dur={duration} ({percent:.2f}%)"
    
    return PerfettoEvent(view_name, dispatch_id, details, start_time, end_time)


def create_overview_event(process_name, thread_name, label, metric, base_start):
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
    
    # Build name with duration and percent (if available)
    name = f"{label}: {duration}"
    if 'percent' in metric:
        name += f" ({metric['percent']:.2f}%)"
    
    return PerfettoEvent(process_name, thread_name, name, start, end)


def render_compile_profile_events(view_name, trace_writer, csv_path):
    """Render compile profile events to Perfetto trace with duration/percent info."""
    _, _, overall_start, overall_end, all_rows = parse_compile_profile_csv(csv_path)
    overall_time = (overall_end - overall_start) if (overall_start is not None and overall_end is not None) else 0
    
    for event, start_time, end_time, src_line_id, etype in all_rows:

        perfetto_event = create_compile_profile_event(
            view_name, event, start_time, end_time, src_line_id, etype, overall_time
        )
        trace_writer.add_event(perfetto_event)


def render_runtime_profile_events(view_name, trace_writer, csv_path):
    """Render runtime profile events to Perfetto trace with duration/percent info."""
    _, overall_start, overall_end, all_rows = parse_runtime_profile_csv(csv_path)
    overall_time = (overall_end - overall_start) if (overall_start is not None and overall_end is not None) else 0
    
    for dispatch_id, start_time, end_time, mlir_loc in all_rows:
        perfetto_event = create_runtime_profile_event(
            view_name, dispatch_id, start_time, end_time, mlir_loc, overall_time
        )
        trace_writer.add_event(perfetto_event)


def render_overview_tracks(view_name, overall_start, metrics, trace_writer):
    """
    Render summary overview tracks showing total DMA, SLICE, IDLE time.
    
    Creates a separate process with overview tracks in fixed order.
    """
    process_name = f"{view_name} Overview"
    base_start = overall_start if overall_start is not None else 0
    
    # Define track order (using numeric prefix for lexicographic sorting)
    overview_tracks = [
        ("00 OVERVIEW DMA", "DMA total", metrics['DMA']),
        ("01 OVERVIEW SLICE", "SLICE total", metrics['SLICE']),
        ("02 OVERVIEW DMA ONLY", "DMA ONLY (exclusive)", metrics['DMA_ONLY']),
        ("03 OVERVIEW DMA+SLICE OVERLAP", "DMA & SLICE overlap", metrics['DMA_SLICE_OVERLAP']),
        ("04 OVERVIEW IDLE", "IDLE", metrics['IDLE']),
        ("05 OVERALL", "OVERALL", metrics['OVERALL']),
    ]
    
    for thread_name, label, metric in overview_tracks:
        perfetto_event = create_overview_event(process_name, thread_name, label, metric, base_start)
        if perfetto_event:
            trace_writer.add_event(perfetto_event)


# =============================================================================
# MAIN CONVERSION LOGIC
# =============================================================================

def convert_to_perfetto(compile_profile_logs, runtime_profile_logs, pb_path, overview_map=None):
    """
    Convert CSV profiles to Perfetto trace format.
    
    Args:
        compile_profile_logs: Dict of {name: csv_path} for compile profiles
        runtime_profile_logs: Dict of {name: csv_path} for runtime profiles
        pb_path: Output path for Perfetto .pb file
        overview_map: Optional dict with pre-calculated metrics per profile
    
    Returns:
        Path to generated Perfetto trace file
    """
    pb_path = Path(pb_path)
    pb_path.parent.mkdir(parents=True, exist_ok=True)
    
    trace_writer = PerfettoTraceWriter(pb_path)
    
    # Render compile profiles
    for name, csv_path in compile_profile_logs.items():
        render_compile_profile_events(name, trace_writer, csv_path)
        
        # Add overview tracks if metrics available
        if overview_map and name in overview_map:
            overview = overview_map[name]
            render_overview_tracks(name, overview['overall_start'], 
                                  overview['metrics'], trace_writer)
    
    # Render runtime profiles
    for name, csv_path in runtime_profile_logs.items():
        render_runtime_profile_events(name, trace_writer, csv_path)
        
        # Add overview tracks if metrics available
        if overview_map and name in overview_map:
            overview = overview_map[name]
            render_overview_tracks(name, overview['overall_start'], 
                                  overview['metrics'], trace_writer)
    
    return pb_path


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Convert timeline CSV logs to Perfetto trace with overview analysis"
    )
    parser.add_argument("--compile_profile", nargs='+', default=[], 
                       help="Path(s) to compile profile CSV file(s)")
    parser.add_argument("--runtime_profile", nargs='+', default=[], 
                       help="Path(s) to runtime profile CSV file(s)")
    parser.add_argument("--pb", type=str, required=True,
                       help="Output path for Perfetto .pb file")
    args = parser.parse_args()
    
    # Build profile dictionaries with filenames as keys
    compile_profile_logs = {}
    for path in args.compile_profile:
        filename = os.path.basename(path)
        compile_profile_logs[filename] = path
    
    runtime_profile_logs = {}
    for path in args.runtime_profile:
        filename = os.path.basename(path)
        runtime_profile_logs[filename] = path
    
    # Calculate metrics for all profiles
    overview_map = {}
    
    for name, path in compile_profile_logs.items():
        overview_data = calculate_compile_profile_metrics(path)
        overview_map[name] = overview_data
        print_metrics_summary(name, overview_data['metrics'])
    
    for name, path in runtime_profile_logs.items():
        overview_data = calculate_runtime_profile_metrics(path)
        overview_map[name] = overview_data
        print_metrics_summary(name, overview_data['metrics'])
    
    # Add numeric prefixes for process ordering in Perfetto UI
    compile_prefixed = add_numeric_prefix(compile_profile_logs, start_index=1)
    runtime_prefixed = add_numeric_prefix(runtime_profile_logs, 
                                          start_index=1 + len(compile_profile_logs))
    
    # Update overview_map keys to match prefixed names
    new_overview_map = {}
    for i, (name, _) in enumerate(compile_profile_logs.items(), start=1):
        prefixed_name = f"{i:02d} {name}"
        new_overview_map[prefixed_name] = overview_map[name]
    
    for i, (name, _) in enumerate(runtime_profile_logs.items(), 
                                  start=1 + len(compile_profile_logs)):
        prefixed_name = f"{i:02d} {name}"
        new_overview_map[prefixed_name] = overview_map[name]
    
    # Generate Perfetto trace
    output_path = convert_to_perfetto(
        compile_profile_logs=compile_prefixed,
        runtime_profile_logs=runtime_prefixed,
        pb_path=args.pb,
        overview_map=new_overview_map
    )
    
    print(f"\n✓ Perfetto trace generated: {output_path}")


if __name__ == "__main__":
    main()
