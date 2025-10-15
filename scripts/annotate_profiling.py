#!/usr/bin/env python3

from iree.compiler.ir import Context, Module, BlockArgument
import argparse
import pandas as pd
from collections import defaultdict


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
                             "torq_hw.css_start", "torq_hw.css_wait"]
            
            task_op_order = {x: idx for idx, x in enumerate(task_op_order)}

            def sort_key(op):                
                return task_op_order[op.operation.name]
            
            ordered_task_ops = sorted(unordered_task_ops, key=sort_key)

            task_ops.extend(ordered_task_ops)

        all_task_ops.append([task_ops, args[op.operation.operands[0]]])

    return all_task_ops


def write_annotated_profile(profiling_data, program_ops, output_file):
    rows = []
    slice_active = [False, False]
    dma_in_active = False
    dma_out_active = False

    start_time = profiling_data.iloc[0]["time_since_open"] - profiling_data.iloc[0]["time_since_start"]

    last_slice_program_name = [None, None]

    for idx, (ops, args) in enumerate(program_ops):

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

        

def main():
    parser = argparse.ArgumentParser(description="Parse an MLIR file.")
    parser.add_argument("mlir_file", type=str, help="Path to the executable-targets phase dump file")
    parser.add_argument("profile_file", type=str, help="Path to the runtime profile file.")
    parser.add_argument("annotated_profile_file", type=str, help="Path to the output")
    args = parser.parse_args()

    profiling_data = pd.read_csv(args.profile_file, sep=';')

    with Context() as ctx:
        parsed_module = parse_mlir_file(args.mlir_file)
        
        dispatches = find_dispatches(parsed_module)

        if len(dispatches) != 1:
            raise ValueError(f"Expected exactly one dispatch in the module, found {len(dispatches)}")

        nss_program_ops = extract_program_ops(dispatches[0])

        print(f"Found {len(nss_program_ops)} NSS programs in the dispatch")

        write_annotated_profile(profiling_data, nss_program_ops, args.annotated_profile_file)

    print(f"Annotated profile written to {args.annotated_profile_file}")


if __name__ == "__main__":
    main()