import contextlib
import time
from typing import Dict, List
from dataclasses import dataclass
from iree.compiler.ir import Context, Module, BlockArgument
import pandas as pd
import re
from torq.model_profiler import perfetto_logger

class Duration(contextlib.AbstractContextManager):
    def __init__(self, scenario: 'Scenario', test_name: str):
        self.scenario = scenario
        self.test_name = test_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter() * 1000
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = time.perf_counter() * 1000
        self.scenario.log(self.test_name, self.start_time, end_time)


@dataclass
class Event:
    name: str
    start_time_ms: float
    end_time_ms: float


class ScenarioLog:
    def __init__(self, name):
        if isinstance(name, list):
            self.name = "::".join(name)
        else:
            self.name = name
        self.events = []

    def log(self, event_name: str, start_time_ms: float, end_time_ms: float):
        self.events.append(Event(event_name, start_time_ms, end_time_ms))

    def event(self, name: str) -> Duration:
        return Duration(self, name)
    
    def name_as_tuple(self):
        return tuple(self.name.split("::"))
    
    def write(self, fp):
        for event in self.events:
            duration = event.end_time_ms - event.start_time_ms
            fp.write(f"{self.name},{event.start_time_ms},{event.end_time_ms},{duration},{event.name}\n")


class PerformanceLog:
    def __init__(self):
        self.scenarios: List[ScenarioLog] = []

    def add_scenario(self, name: str) -> ScenarioLog:
        scenario = ScenarioLog(name)
        self.scenarios.append(scenario)
        return scenario

    def write(self, filepath: str):
        with open(filepath, 'w') as f:
            for scenario in self.scenarios:
                scenario.write(f)


def load_performance(filepath: str) -> PerformanceLog:
    performance = PerformanceLog()

    scenarios: Dict[str, ScenarioLog] = {}

    with open(filepath, 'r') as f:
        for line in f:
            scenario_name, start_time_ms_str, end_time_ms_str, _, event_name = line.strip().split(',')
            start_time_ms = float(start_time_ms_str)
            end_time_ms = float(end_time_ms_str)
            
            scenario = scenarios.get(scenario_name)

            if scenario is None:
                scenario = performance.add_scenario(scenario_name)                
                scenarios[scenario_name] = scenario

            scenario.log(event_name, start_time_ms, end_time_ms)

    return performance


def parse_mlir_file(file_path):
    with open(file_path, 'r') as mlir_file:
        mlir_content = mlir_file.read()
        
    return Module.parse(mlir_content)    


def find_descendants_of_type(op, descendant_type, depth=-1):
    found_descendants = []

    for region in op.operation.regions:
        for block in region:
            for child in block.operations:
                if child.operation.name == descendant_type:
                    found_descendants.append(child)

                if depth < 0 or depth > 0:
                    found_descendants.extend(find_descendants_of_type(child, descendant_type, depth - 1))
                

    return found_descendants


def find_dispatches(module):
    dispatches = []
    for op in find_descendants_of_type(module, "hal.executable", depth=1):
        funcs = find_descendants_of_type(op, "func.func")

        if len(funcs) != 1:
            raise ValueError(f"Expected exactly one function in executable, found {len(funcs)}")
        
        dispatches.append(funcs[0])

    return dispatches


def extract_program_ops(dispatch_op):
    invocation_programs = {}

    for invocation in find_descendants_of_type(dispatch_op, "torq_hl.create_invocation", depth=0):
        invocation_programs[invocation.results[0]] = invocation.operation.operands[0].owner

    args = {}
    for start_op in find_descendants_of_type(dispatch_op, "torq_hl.start_program", depth=0):
        args[start_op.operation.operands[0]] = start_op.operation.operands[2:]

    all_task_ops = []

    for op in find_descendants_of_type(dispatch_op, "torq_hl.wait_program", depth=0):
        waited_program = invocation_programs[op.operation.operands[0]]

        # ignore host programs that are not currently traced        
        if str(op.operation.operands[0].type) != "!torq_hl.invocation<nss>":
            continue

        task_ops = []
        for task in find_descendants_of_type(waited_program, "torq_hw.nss_task", depth=0):
        
            unordered_task_ops = task.operation.regions[0].blocks[0]

            task_op_order = ["torq_hw.dma_in_cfg", "torq_hw.dma_in_start", "torq_hw.dma_in_wait",
                             "torq_hw.slice_start", "torq_hw.slice_wait",
                             "torq_hw.dma_out_cfg", "torq_hw.dma_out_start", "torq_hw.dma_out_wait",
                             "torq_hw.css_start", "torq_hw.css_wait",
                             "torq_hw.cdma_start", "torq_hw.cdma_wait"]
            
            task_op_order = {x: idx for idx, x in enumerate(task_op_order)}

            def sort_key(op):                
                return task_op_order[op.operation.name]
            
            ordered_task_ops = sorted(unordered_task_ops, key=sort_key)

            task_ops.extend(ordered_task_ops)

        all_task_ops.append([task_ops, args[op.operation.operands[0]]])

    return all_task_ops


def extract_ops_based_on_action_ids(dispatch_op):
    """
    Returns a list of (ops, args, job_id) for each torq-action-id, ordered by action-id.
    Each element is a tuple: (list of all ops in the program for that action-id, args for that action, job_id if it's nss/css op).
    This includes all ops, not just those in the program region.
    """
    # Collect all ops in the dispatch_op (flat, in order)
    all_ops = []
    for region in dispatch_op.operation.regions:
        for block in region:
            for op in block.operations:
                all_ops.append(op)

    # Build a mapping from invocation value to its create_invocation op
    invocation_to_create = {}
    for op in all_ops:
        if op.operation.name == "torq_hl.create_invocation":
            invocation_val = op.results[0]
            invocation_to_create[invocation_val] = op

    # Find all ops with a torq-action-id attribute
    action_ops = []
    for op in all_ops:
        job_id = None
        if "torq-action-id" in op.operation.attributes:
            action_id_attr = op.operation.attributes["torq-action-id"]
            # If this is a wait_program, try to find the job_id
            # note: host ops does not have job id
            if op.operation.name == "torq_hl.wait_program" or op.operation.name == "torq_hl.start_program":
                invocation_operand = op.operation.operands[0]
                create_invocation_op = invocation_to_create.get(invocation_operand)
                if create_invocation_op is not None:
                    attrs = create_invocation_op.operation.attributes
                    if "torq-job-id" in attrs:
                        job_id = attrs["torq-job-id"].value
            action_ops.append((action_id_attr.value, op, job_id))

    # Sort by action-id
    action_ops.sort(key=lambda x: x[0])

    return action_ops

def extract_line_numbers(location_str):
    # Extracts line numbers from location string, excluding call site locations after " at "
    # Handles both simple callsites and fused locations
    # Example: loc(callsite("...":4:10 at "...":2:3)) -> [4]
    # Example: loc(fused[callsite("...":147:12 at "...":2:3), callsite("...":145:12 at "...":2:3)]) -> [147, 145]
    
    # Split by callsite boundaries to process each separately
    line_numbers = []
    # Find all callsite(...) patterns
    for callsite_match in re.finditer(r'callsite\([^)]+\)', location_str):
        callsite_str = callsite_match.group(0)
        # Extract the first line:col before " at " (this is the actual location, not the call site)
        match = re.search(r'":?(\d+):(\d+)(?:\s+at\s+|")', callsite_str)
        if match:
            line_num = int(match.group(1))
            line_numbers.append(line_num)
    
    return line_numbers if line_numbers else [None]

def get_operator_from_line(line):
    line = line.strip()
    if not line:
        return ""
    
    # Remove result assignment (e.g., "%0 = ...")
    if "=" in line:
        lhs = line.split("=", 1)[0].strip()
        # Only split if the LHS looks like a result definition (starts with % or ()
        if lhs.startswith("%") or lhs.startswith("("):
            line = line.split("=", 1)[1].strip()
    
    # Handle torch.operator "..." pattern (covers ONNX and other operators)
    if line.startswith('torch.operator '):
        match = re.search(r'"([^"]+)"', line)
        if match:
            return match.group(1)

    # Handle TOSA operators (both quoted and unquoted forms)
    if line.startswith('"tosa.'):
        match = re.search(r'"([^"]+)"', line)
        if match:
            return match.group(1)
    elif line.startswith('tosa.'):
        # Return the first token (e.g., "tosa.conv2d")
        return line.split()[0]

    return ""

def write_host_annotated_profile(profiling_dict, actions_ops, nss_program_ops, output_file, original_mlir_lines=None, perfetto_file=None):
    """
    Generates an annotated performance profile report for host execution, mapping low-level operations
    to original MLIR source lines and tracking hardware resource usage.
    This function processes a sequence of executed actions, correlates them with profiling data,
    and produces two types of outputs: a tabular report (CSV or Excel) and an optional Perfetto trace.
    Args:
        profiling_dict (dict): A dictionary mapping action IDs to profiling data (timestamps, locations, total_time).
        actions_ops (list): A list of tuples (action_id, op, job_id) representing the sequence of executed operations.
        nss_program_ops (dict): A mapping of job IDs to their constituent NSS program operations and arguments.
                                This is used to expand `wait_program` calls into specific hardware instructions
                                (e.g., slice start/wait, DMA start/wait).
        output_file (str): The file path where the tabular report (CSV or .xlsx) will be written.
        original_mlir_lines (list[str], optional): A list of original source code lines used to resolve
                                                   operator names from line numbers found in location data.
        perfetto_file (str, optional): The file path where the Perfetto trace will be written. If None,
                                       no trace is generated.
    Internal Data Structures:
        rows (list[dict]): Accumulates data for the tabular output (CSV or Excel). Each dictionary represents
                           a single operation's metrics, including resolved names and resource usage flags.
        perfetto_rows (list[tuple]): Accumulates data specifically formatted for the Perfetto trace writer.
                                     Each tuple contains timestamped event data required for visual rendering.
    """
    rows = []
    perfetto_rows = []
    slice_active = [False, False]
    dma_in_active = False
    dma_out_active = False
    cdma_active = False
    css_active = False

    last_slice_program_name = [None, None]

    for (action_id, op, job_id) in actions_ops:
        
        is_host_wait_task = (
            op.operation.name == "torq_hl.wait_program"
            and len(op.operation.operands) > 0
            and str(op.operation.operands[0].type) == "!torq_hl.invocation<host>"
        )

        original_operator = ""
        # we dont log the host_wait_program seperately as host_start_program is synchronous.
        if original_mlir_lines and not is_host_wait_task:
            loc_str = ','.join(profiling_dict[action_id].get("location", []))
            line_nums = extract_line_numbers(loc_str)
            # Sort line numbers in ascending order
            line_nums = sorted([ln for ln in line_nums if ln is not None])
            operators = []
            for line_num in line_nums:
                if 1 <= line_num <= len(original_mlir_lines):
                    operator_name = get_operator_from_line(original_mlir_lines[line_num - 1])
                    if operator_name:  # Only add non-empty operators
                        # Append line number to operator name
                        operators.append(f"{operator_name}@L{line_num}")
            original_operator = " + ".join(operators) if operators else ""

        operation = op.operation.name
        invocation_name = None
        if operation == "torq_hl.wait_program" and job_id is not None:
            ops, args = nss_program_ops[job_id]
            slice_used = [slice_active[0], slice_active[1]]
            dma_out_used = dma_out_active
            dma_in_used = dma_in_active
            cdma_used = cdma_active
            css_used = css_active

            for op in ops:
                operation = op.operation.name

                if operation == "torq_hw.slice_start" or operation == "torq_hw.slice_wait":
                    slice_id = op.operation.attributes["id"].value

                if operation.startswith("torq_hw.slice_start"):
                    slice_active[slice_id] = True
                    slice_used[slice_id] = True
                elif operation.startswith("torq_hw.slice_wait"):
                    slice_active[slice_id] = False                
                if operation == "torq_hw.dma_in_start":
                    dma_in_active = True
                    dma_in_used = True
                elif operation == "torq_hw.dma_in_wait":
                    dma_in_active = False                
                if operation == "torq_hw.dma_out_start":
                    dma_out_active = True
                    dma_out_used = True
                elif operation == "torq_hw.dma_out_wait":
                    dma_out_active = False
                if operation == "torq_hw.cdma_start":
                    cdma_active = True
                    cdma_used = True
                elif operation == "torq_hw.cdma_wait":
                    cdma_active = False
                if operation == "torq_hw.css_start":
                    css_active = True
                    css_used = True
                elif operation == "torq_hw.css_wait":
                    css_active = False

            for op in ops:
                operation = op.operation.name

                slice_id = None
                invocation_name = None

                if operation == "torq_hw.slice_start" or operation == "torq_hw.slice_wait":
                    slice_id = op.operation.attributes["id"].value
                    
                    if operation == "torq_hw.slice_start":
                        invocation_arg = BlockArgument(op.operands[0])
                        invocation = args[invocation_arg.arg_number]                    
                        invocation_name = str(invocation.owner.attributes["name"].value)                                        
                        invocation_name = invocation_name[len("slice_program_torq_hl."):]
                        last_slice_program_name[slice_id] = invocation_name
                    else:
                        invocation_name = last_slice_program_name[slice_id]
                    

                else:
                    slice_id = None

                row = {
                    "action_id": action_id,
                    "job_id": job_id if job_id is not None else "",
                    "operation": operation,
                    "invocation_name": invocation_name,
                    "total_time": profiling_dict[action_id].get("total_time"),
                    "slice_id": slice_id,
                    "slice_used_0_in_program": slice_used[0],
                    "slice_used_1_in_program": slice_used[1],
                    "dma_in_used_in_program": dma_in_used,
                    "dma_out_used_in_program": dma_out_used,
                    "cdma_used_in_program": cdma_used,
                    "css_used_in_program": css_used,
                    "timestamp_start": profiling_dict[action_id].get("timestamp_start"),
                    "timestamp_end": profiling_dict[action_id].get("timestamp_end"),
                    "location": ','.join(profiling_dict[action_id].get("location", [])),
                    "original_operator": original_operator
                }

                rows.append(row)

                # Collect data for Perfetto
                if perfetto_file:
                    ts_start = profiling_dict[action_id].get("timestamp_start")
                    ts_end = profiling_dict[action_id].get("timestamp_end")
                    if ts_start is not None and ts_end is not None:
                        perfetto_rows.append((
                            action_id,
                            job_id if job_id is not None else "",
                            ts_start,
                            ts_end,
                            ','.join(profiling_dict[action_id].get("location", [])),
                            original_operator,
                            invocation_name,
                            operation,
                            slice_id,
                            slice_used[0],
                            slice_used[1],
                            dma_in_used,
                            dma_out_used,
                            cdma_used,
                            css_used,
                        ))

        elif is_host_wait_task: 
            # Host execution is currently synchronous, so we skip this since it is already logged with host_start_task.
            continue
        else:
            # Since host execution is synchronous, we skip the wait_program log as it does not make sense; 
            # the easiest way to indicate that the host op has finished is when we log the start.
            # TODO: Maybe find a way to determine this from profiling data instead of using this heuristic.
            if operation == "torq_hl.start_program" and str(op.operation.operands[0].type) == "!torq_hl.invocation<host>":
                operation += " (host)"

            row = {
                    "action_id": action_id,
                    "job_id": job_id if job_id is not None else "",
                    "operation": operation,
                    "invocation_name": None, # Not an invocation
                    "total_time": profiling_dict[action_id].get("total_time"),
                    "slice_id": None, # Not an slice job
                    "slice_used_0_in_program": None, # Not an slice job
                    "slice_used_1_in_program": None, # Not an slice job
                    "dma_in_used_in_program": None, # Not an slice job
                    "dma_out_used_in_program": None, # Not an slice job
                    "cdma_used_in_program": None, # Not an slice job
                    "css_used_in_program": None, # Not an slice job
                    "timestamp_start": profiling_dict[action_id].get("timestamp_start"),
                    "timestamp_end": profiling_dict[action_id].get("timestamp_end"),
                    "location": ','.join(profiling_dict[action_id].get("location", [])),
                    "original_operator": original_operator
                }
            rows.append(row)

            # Collect data for Perfetto
            if perfetto_file:
                ts_start = profiling_dict[action_id].get("timestamp_start")
                ts_end = profiling_dict[action_id].get("timestamp_end")
                if ts_start is not None and ts_end is not None:
                    perfetto_rows.append((
                        action_id,
                        job_id if job_id is not None else "",
                        ts_start,
                        ts_end,
                        ','.join(profiling_dict[action_id].get("location", [])),
                        original_operator,
                        None, # invocation_name
                        operation,
                        None, # slice_id
                        None, # slice_used_0
                        None, # slice_used_1
                        None, # dma_in_used
                        None, # dma_out_used
                        None, # cdma_used
                        None, # css_used
                    ))

    df = pd.DataFrame(rows)

    # Ensure consistent column order
    desired_columns = [
        "action_id", "job_id", "operation", "invocation_name", "original_operator", "total_time",
        "slice_id", "slice_used_0_in_program", "slice_used_1_in_program",
        "dma_in_used_in_program", "dma_out_used_in_program", "cdma_used_in_program", "css_used_in_program",
        "timestamp_start", "timestamp_end", "location"
    ]
    # Only keep columns that are actually in the DataFrame
    existing_columns = [col for col in desired_columns if col in df.columns]
    df = df[existing_columns]
    
    if output_file.endswith('.csv'):
        df.to_csv(output_file, index=False, sep=';')
    elif output_file.endswith('.xlsx'):
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            
            # Use human-friendly column names            
            human_friendly_columns = {
                "action_id": "Action ID",
                "job_id": "Job ID",
                "operation": "Action in program",
                "invocation_name": "Invocation Name (if applicable)",
                "total_time": "Total Time [us]",
                "slice_id": "Slice ID (if applicable)",
                "slice_used_0_in_program": "Slice 0 Used in NSS Program",
                "slice_used_1_in_program": "Slice 1 Used in NSS Program",
                "dma_in_used_in_program": "DMA In Used in NSS Program",
                "dma_out_used_in_program": "DMA Out Used in NSS Program",
                "cdma_used_in_program": "CDMA Used in CSS Program",
                "css_used_in_program": "CSS Used in CSS Program",
                "timestamp_start": "Timestamp Start",
                "timestamp_end": "Timestamp End",
                "location": "Location",
                "original_operator": "Original Operator"
            }

            df.rename(columns=human_friendly_columns, inplace=True)
            sheet_name = "Detailed Performance Data"
            df.to_excel(writer, index=False, sheet_name=sheet_name)        

            for worksheet in writer.sheets.values():            
                for idx, col in enumerate(df.columns):
                    max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                    worksheet.set_column(idx, idx, max_len)

    # Generate Perfetto trace directly
    if perfetto_file and perfetto_rows:
        # Calculate overall start and end from the collected rows
        overall_start = min(r[2] for r in perfetto_rows)
        overall_end = max(r[3] for r in perfetto_rows)
        
        trace_writer = perfetto_logger.PerfettoTraceWriter(perfetto_file)
        perfetto_logger.log_runtime_profile_data("Host Profile", trace_writer, perfetto_rows, overall_start, overall_end)
        
        # Compute Metrics and Render Overview
        metrics_result = perfetto_logger.compute_runtime_metrics(perfetto_rows, overall_start, overall_end)
        perfetto_logger.render_overview_tracks("Host Profile", metrics_result['overall_start'], metrics_result['metrics'], trace_writer)
        
        trace_writer.close()

def write_annotated_profile(profiling_data, program_ops, output_file, perfetto_file=None):
    rows = []
    perfetto_rows = []
    slice_active = [False, False]
    dma_in_active = False
    dma_out_active = False

    start_time = profiling_data.iloc[0]["time_since_open"] - profiling_data.iloc[0]["time_since_start"]

    last_slice_program_name = [None, None]

    for idx, (ops, args) in enumerate(program_ops):

        if idx >= len(profiling_data):
            print(f"Warning: Profiling data has fewer entries ({len(profiling_data)}) than program ops ({len(program_ops)}). Stopping annotation.")
            break

        data_after = profiling_data.iloc[idx]

        if idx > 0:
            data_before = profiling_data.iloc[idx - 1]
        else:
            data_before = {"time_since_open": data_after["time_since_open"] - data_after["time_since_start"]}

        slice_used = [slice_active[0], slice_active[1]]
        dma_out_used = dma_out_active
        dma_in_used = dma_in_active

        for op in ops:
            operation = op.operation.name

            if operation == "torq_hw.slice_start" or operation == "torq_hw.slice_wait":
                slice_id = op.operation.attributes["id"].value

            if operation.startswith("torq_hw.slice_start"):
                slice_active[slice_id] = True
                slice_used[slice_id] = True
            elif operation.startswith("torq_hw.slice_wait"):
                slice_active[slice_id] = False                
            if operation == "torq_hw.dma_in_start":
                dma_in_active = True
                dma_in_used = True
            elif operation == "torq_hw.dma_in_wait":
                dma_in_active = False                
            if operation == "torq_hw.dma_out_start":
                dma_out_active = True
                dma_out_used = True
            elif operation == "torq_hw.dma_out_wait":
                dma_out_active = False

        for op in ops:
            operation = op.operation.name
            slice_id = None
            invocation_name = None

            if operation == "torq_hw.slice_start" or operation == "torq_hw.slice_wait":
                slice_id = op.operation.attributes["id"].value
                
                if operation == "torq_hw.slice_start":
                    invocation_arg = BlockArgument(op.operands[0])
                    invocation = args[invocation_arg.arg_number]                    
                    invocation_name = str(invocation.owner.attributes["name"].value)                                        
                    invocation_name = invocation_name[len("slice_program_torq_hl."):]
                    last_slice_program_name[slice_id] = invocation_name
                else:
                    invocation_name = last_slice_program_name[slice_id]
                

            else:
                slice_id = None

            row = {
                "nss_program_index": idx,
                "operation": operation,
                "slice_id": slice_id,
                "invocation_name": invocation_name,
                "nss_program_timestamp_start": data_before['time_since_open'] - start_time,
                "nss_program_timestamp_end": data_after['time_since_open'] - start_time,
                "nss_program_timestamp_total": data_after['time_since_open'] - data_before['time_since_open'],
                "slice_used_0_in_program": slice_used[0],
                "slice_used_1_in_program": slice_used[1],
                "dma_in_used_in_program": dma_in_used,
                "dma_out_used_in_program": dma_out_used
            }

            rows.append(row)

            # Collect data for Perfetto
            if perfetto_file:
                ts_start = data_before['time_since_open'] - start_time
                ts_end = data_after['time_since_open'] - start_time
                perfetto_rows.append((
                    str(idx), # action_id (using index as proxy)
                    "", # job_id
                    ts_start,
                    ts_end,
                    "", # location
                    "", # original_operator
                    invocation_name,
                    operation,
                    slice_id,
                    slice_used[0],
                    slice_used[1],
                    dma_in_used,
                    dma_out_used
                ))

    df = pd.DataFrame(rows)
    
    if output_file.endswith('.csv'):
        df.to_csv(output_file, index=False, sep=';')
    elif output_file.endswith('.xlsx'):
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            
            # Use human-friendly column names            
            human_friendly_columns = {
                "nss_program_index": "NSS Program Index",
                "operation": "torq-hw Operation in program",
                "slice_id": "Slice ID (if applicable)",
                "invocation_name": "Invocation Name (if applicable)",
                "nss_program_timestamp_start": "NSS Program Start Timestamp [us]",
                "nss_program_timestamp_end": "NSS Program End Timestamp [us]",
                "nss_program_timestamp_total": "NSS Program Total Time [us]",
                "slice_used_0_in_program": "Slice 0 Used in NSS Program",
                "slice_used_1_in_program": "Slice 1 Used in NSS Program",
                "dma_in_used_in_program": "DMA In Used in NSS Program",
                "dma_out_used_in_program": "DMA Out Used in NSS Program"
            }

            df.rename(columns=human_friendly_columns, inplace=True)
            sheet_name = "Detailed Performance Data"
            df.to_excel(writer, index=False, sheet_name=sheet_name)            

            for worksheet in writer.sheets.values():            
                for idx, col in enumerate(df.columns):
                    max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                    worksheet.set_column(idx, idx, max_len)

    # Generate Perfetto trace directly
    if perfetto_file and perfetto_rows:
        # Calculate overall start and end from the collected rows
        overall_start = min(r[2] for r in perfetto_rows)
        overall_end = max(r[3] for r in perfetto_rows)
        
        trace_writer = perfetto_logger.PerfettoTraceWriter(perfetto_file)
        perfetto_logger.log_runtime_profile_data("NSS Profile", trace_writer, perfetto_rows, overall_start, overall_end)
        
        # Compute Metrics and Render Overview
        metrics_result = perfetto_logger.compute_runtime_metrics(perfetto_rows, overall_start, overall_end)
        perfetto_logger.render_overview_tracks("NSS Profile", metrics_result['overall_start'], metrics_result['metrics'], trace_writer)
        
        trace_writer.close()

def parse_profiling_to_dict(profiling_data):
    profiling_data.columns = profiling_data.columns.str.strip().str.lower()
    
    result = {}
    for _, row in profiling_data.iterrows():
        action_idx = int(row['actionindex'])
        event = row['event']
        timestamp = int(row['timestamp(us)'])
        location = row['location']

        if action_idx not in result:
            result[action_idx] = {
                "timestamp_start": None,
                "timestamp_end": None,
                "location": set()
            }

        # Set start and end timestamps based on event name
        if event.endswith("START"):
            result[action_idx]["timestamp_start"] = timestamp
        elif event.endswith("END"):
            result[action_idx]["timestamp_end"] = timestamp

        # Add location to the set
        result[action_idx]["location"].add(location)
    
    for action_id, entry in result.items():
        start = entry.get("timestamp_start")
        end = entry.get("timestamp_end")
        if start is not None and end is not None:
            entry["total_time"] = end - start
        else:
            entry["total_time"] = None
            
    return result

def annotate_host_profile_from_files(mlir_file, profile_file, output_file, original_mlir_file=None, perfetto_file=None):
    """Wrapper function for easy external use."""
    profiling_data = pd.read_csv(profile_file, sep=',')
    profiling_dict = parse_profiling_to_dict(profiling_data)
    
    original_mlir_lines = None
    if original_mlir_file:
        try:
            with open(original_mlir_file, 'r') as f:
                original_mlir_lines = f.readlines()
        except Exception as e:
            print(f"Warning: Could not read original MLIR file: {e}")

    with Context() as ctx:
        parsed_module = parse_mlir_file(mlir_file)
        dispatches = find_dispatches(parsed_module)
        
        if len(dispatches) != 1:
            raise ValueError(f"Expected exactly one dispatch, found {len(dispatches)}")
        
        actions_ops = extract_ops_based_on_action_ids(dispatches[0])
        nss_program_ops = extract_program_ops(dispatches[0])

        write_host_annotated_profile(profiling_dict, actions_ops, nss_program_ops, output_file, original_mlir_lines, perfetto_file)


def annotate_nss_profile_from_files(mlir_file, profile_file, output_file, perfetto_file=None):
    """Wrapper function for easy external use."""
    profiling_data = pd.read_csv(profile_file, sep=';')
    profiling_data.columns = profiling_data.columns.str.strip().str.lower()

    with Context() as ctx:
        parsed_module = parse_mlir_file(mlir_file)
        
        dispatches = find_dispatches(parsed_module)

        if len(dispatches) != 1:
            raise ValueError(f"Expected exactly one dispatch in the module, found {len(dispatches)}")

        nss_program_ops = extract_program_ops(dispatches[0])

        print(f"Found {len(nss_program_ops)} NSS programs in the dispatch")

        write_annotated_profile(profiling_data, nss_program_ops, output_file, perfetto_file)

    print(f"Annotated profile written to {output_file}")