import streamlit as st
from zipfile import ZipFile
import numpy as np
import hashlib
import sys
import os
from contextlib import contextmanager


class Buffer:
    def __init__(self, executable, **kwargs):

        self.executable = executable

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.id = int(self.id)
        self.allocation_action = int(self.allocation_action)
        self.deallocation_action = int(self.deallocation_action)
        self.last_use_action = int(self.last_use_action)
        self.address = int(self.address)
        self.size = int(self.size)

    def load(self, action):
        return self.executable.buffer_dump.load_buffer(self, action)


class Executable:
    def __init__(self, buffer_dump, name):
        self.buffer_dump = buffer_dump
        self.name = name
        self.buffers = []
        self.action_count = 0

    def add_buffer(self, buffer):
        self.buffers.append(buffer)
        self.action_count = max(self.action_count, buffer.allocation_action, buffer.deallocation_action)
    

class BufferDump:    

    def __init__(self, file_name):
        self.file_name = file_name

        self.is_zip = not (isinstance(self.file_name, str) and os.path.isdir(self.file_name))
        
        self.root = None
        self.executables = {}        
        self.load_executables()        

    def get_executable_names(self):
        executables = []

        if not self.is_zip:
            return os.listdir(self.file_name)
        else:
            with ZipFile(self.file_name, 'r') as zip_file:
                file_list = zip_file.namelist()
                        
                for file_name in file_list:                
                    if not file_name.endswith('/'):
                        continue

                    parts = file_name.strip('/').split('/')

                    if len(parts) == 1:
                        self.root = parts[0]
                        continue

                    if len(parts) == 2:
                        executables.append(parts[1])

        return executables        
    
    @contextmanager
    def open_file(self, path):
        if not self.is_zip:
            with open(self.file_name + "/" + path, 'rb') as fp:
                yield fp
        else:  
            with ZipFile(self.file_name, 'r') as zip_file:          
                yield zip_file.open(self.root + '/' + path)

    def load_executable(self, executable_name):
        executable = Executable(self, executable_name)

        with self.open_file(executable_name + '/buffers.csv') as f:
            headers = f.readline().strip().decode('utf-8').split(';')

            for row in f.readlines():
                row = row.decode('utf-8').strip().split(';')
                buffer = Buffer(executable, **{header: value for header, value in zip(headers, row)})
                executable.add_buffer(buffer)

        return executable


    def load_executables(self):            
        for executable_name in self.get_executable_names():
            self.executables[executable_name] = self.load_executable(executable_name)
                    
    def load_buffer(self, buffer, action_id):      
        with self.open_file(buffer.executable.name + '/action' + str(action_id) + '/buffer_' + str(buffer.id) + '.npy') as fp:        
            return  np.load(fp, allow_pickle=False)
    
    def sha1_buffer(self, buffer, action_id):        
        buffer_data = self.load_buffer(buffer, action_id)
        return hashlib.sha1(buffer_data.tobytes()).hexdigest()
    
    def exists(self, buffer, action_id):
        try:
            self.load_buffer(buffer, action_id)
            return True
        except FileNotFoundError:
            return False

    def has_changed(self, action, buffer):

        if action < 2:
            return False
        
        if action <= buffer.allocation_action or action >= buffer.deallocation_action:
            return False


        buffer_after_action = self.load_buffer(buffer, action)
        buffer_before_action  = self.load_buffer(buffer, action - 1)

        return np.any(buffer_after_action != buffer_before_action)


def show_buffer(buffer, action):
    
    buffer_data = buffer.load(action)

    if len(buffer_data.shape) > 2:
        idx = []
        
        for dim in range(len(buffer_data.shape) - 2):        
            if buffer_data.shape[dim] > 1:                
                idx.append(st.slider("Dimension " + str(dim), 0, buffer_data.shape[dim] - 1, 0, key=f"dim_{buffer.id}_{action}_{dim}"))        
            else:
                st.caption("Dimension " + str(dim))
                st.write(1)
                idx.append(0)

        st.caption("Buffer data at index [" + ", ".join([str(i) for i in idx]) + ",:,:] after action " + str(action) + " executed")
        
        squeezed = buffer_data[tuple(idx)].squeeze()
        if len(squeezed.shape) == 0:
            st.write(str(squeezed))
        else:
            st.dataframe(squeezed)
    
    else:

        st.caption("Buffer data after action " + str(action) + " executed")

        st.dataframe(buffer_data)


def show_buffer_difference(buffer, action1, action2):

    buffer_data_1 = buffer.load(action1)
    buffer_data_2 = buffer.load(action2)

    show_any_buffer_difference(buffer_data_1, buffer_data_2)


def show_any_buffer_difference(buffer_data_1, buffer_data_2):

    buffer_data = buffer_data_2 - buffer_data_1

    if len(buffer_data.shape) > 2:
        idx = []
        
        for dim in range(len(buffer_data.shape) - 2):        
            if buffer_data.shape[dim] > 1:                
                idx.append(st.slider("Dimension " + str(dim), 0, buffer_data.shape[dim] - 1, 0))        
            else:
                st.caption("Dimension " + str(dim))
                st.write(1)
                idx.append(0)

        st.caption("Buffer difference at index [" + ", ".join([str(i) for i in idx]) + ",:,:]")

        buffer_data_slice = buffer_data[tuple(idx)].squeeze()
    
    else:

        st.caption("Buffer data difference")

        buffer_data_slice = buffer_data

    # create numpy array with a three channels image from buffer_data_slice with color of a pixel ref if the value in the original data is not zero, otherwise white

    if len(buffer_data_slice.shape) > 1:
        buffer_size = buffer_data_slice.shape[0] * buffer_data_slice.shape[1]
    else:
        buffer_size = buffer_data_slice.shape[0]

    map_width = st.slider("Map width (pixels)", 1, buffer_size, 64)

    map_size = int(np.ceil(buffer_size / map_width) * map_width)

    buffer_map = np.full((map_size, ), np.nan)
    buffer_map[:buffer_data_slice.size] = buffer_data_slice.flatten()    
    buffer_map = buffer_map.reshape(-1, map_width)
    buffer_map = np.expand_dims(buffer_map, axis=-1)

    # set color to red if the value is not zero, otherwise white, nans are set to gray
    buffer_map = np.where(
        np.isnan(buffer_map),
        np.array([80, 80, 80], dtype=np.uint8),
        np.where(
            buffer_map == 0,
            np.array([0, 255, 0], dtype=np.uint8),
            np.array([255, 0, 0], dtype=np.uint8)
        )
    )

    # upscale the image with nearest neighbor interpolation x4
    buffer_map = np.repeat(np.repeat(buffer_map, 4, axis=0), 4, axis=1)

    st.image(buffer_map, caption="Buffer difference image (red difference, green unchanged, gray outside buffer)")

    st.dataframe(buffer_data_slice)


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


@st.cache_data
def parse_code(file_path):
    from iree.compiler.ir import Context, Module, BlockArgument, OpResult

    with open(file_path, 'r') as mlir_file:
        mlir_content = mlir_file.read()

    with Context() as ctx:
        module = Module.parse(mlir_content)
        actions = {}
        buffer_actions = {}
        program_arguments = {}
        for dispatch in find_dispatches(module):

            dispatch_name = dispatch.operation.attributes["sym_name"].value

            buffer_actions[dispatch_name] = {}
            actions_buffers = {}

            for region in dispatch.operation.regions:
                for block in region:
                    for child in block.operations:
                        if "torq-buffer-ids" in child.operation.attributes:                            
                            buffer_ids = child.operation.attributes["torq-buffer-ids"]
                            mem_ref_result_id = 0
                            for result in child.operation.results:
                                if str(result.type).startswith("memref<"):
                                    buffer_actions[dispatch_name][buffer_ids[mem_ref_result_id]] = str(result)
                                    actions_buffers[result] = buffer_ids[mem_ref_result_id]
                                    mem_ref_result_id += 1

            actions[dispatch_name] = []
            program_arguments[dispatch_name] = {}
            invocation_to_args = {}

            for region in dispatch.operation.regions:
                for block in region:
                    for child in block.operations:
                        
                        if "torq-action-id" in child.operation.attributes:
                            action_id = child.operation.attributes["torq-action-id"].value
                            if action_id != len(actions[dispatch_name]):
                                raise ValueError(f"Unexpected action id {action_id} in dispatch {dispatch_name}, expected {len(actions[dispatch_name])}")
                                                        
                            action_desc = [("Action", str(child))]

                            program_arguments[dispatch_name][action_id] = []

                            if child.operation.name in ["torq_hl.start_program", "torq_hl.wait_program"]:                                                                
                                invocation = child.operation.operands[0]
                                invocation_op = invocation.owner                                
                                action_desc.append(("Where the invocation is", str(invocation_op)))

                                program_op = invocation_op.operands[0].owner
                                action_desc.append(("Where the program is", str(program_op)))

                                if child.operation.name == "torq_hl.start_program":
                                    code_sections = child.operation.attributes["operandSegmentSizes"][1]
                                    invocation_to_args[invocation_op] = child.operation.operands[(code_sections + 1):]

                                for operand in invocation_to_args[invocation_op]:
                                    buffer_id = None
                                    if "torq-buffer-ids" in operand.owner.attributes:                                        
                                        if str(operand.type).startswith("memref<"):                                            
                                            buffer_id = actions_buffers[operand]
                                    program_arguments[dispatch_name][action_id].append(buffer_id)

                            actions[dispatch_name].append(action_desc)

    return actions, buffer_actions, program_arguments

st.set_page_config(page_title="Torq Buffer dump viewer", layout="wide")

st.title("Torq Buffer dump viewer")

st.markdown("""
            This is a simple viewer for the Torq buffer dump files.

            To analyze the buffers dumps of a model use the following steps:
            
            1. Compile a model with debug information enabled using the  `--torq-enable-buffer-debug-info` flag
            
            2. Execute the model with buffer dumps enabled using the `--torq_dump_buffers_dir=${PATH}` command line option

            3. Zip the contents of the dump directory and upload it here to view the contents

            If you started this tool from the command line, you can also pass the path to the dump directory as an argument.

            Optionally, you can also pass the path to the source `executable-targets` IR of the model as a second argument to see the code locations of each action.
            
            """)

action_descs = None
buffer_actions = None

if len(sys.argv) > 1:
    uploaded_file = sys.argv[1]
    st.write("File name: " + uploaded_file)

    if len(sys.argv) > 2:
        code_path = sys.argv[2]
        st.write("Code: " + code_path)
        action_descs, buffer_actions, program_arguments = parse_code(code_path)
else:
    uploaded_file = st.file_uploader("Buffer dump zip file")

if uploaded_file is None:
    st.stop()

buffer_dump = BufferDump(uploaded_file)

analysis_mode = st.radio("Analysis mode", ["View buffers changes for each action", "View buffer state across actions", "Compare two buffer states"])

if analysis_mode == "View buffer state across actions":

    executable = st.selectbox("Dispatch", buffer_dump.executables.keys())

    buffer = st.selectbox("Buffer", buffer_dump.executables[executable].buffers, format_func=lambda x: str(x.id) + " (" + x.shape + " " + x.type + ")")

    st.subheader("Buffer information")

    description = [
        ["Type", buffer.type],
        ["Shape", buffer.shape],
        ["Size in bytes", str(buffer.size) + " (`" + hex(buffer.size) + "`)"],
        ["Address", str(buffer.address) + " (`" + hex(buffer.address) + "`)"],
        ["Allocation action", buffer.allocation_action],
        ["Last use action", buffer.last_use_action],
        ["Deallocation action", buffer.deallocation_action],
        ["Allocation location", "``" + buffer.allocation_location + "``"],
        ["Deallocation location", "``" + buffer.deallocation_location + "``"],
    ]    

    for desc in description:        
        st.write(f"**{desc[0]}**: {desc[1]}")

    if buffer_actions is not None:
        st.markdown("**Action that creates this buffer**")
        st.code(buffer_actions[executable][buffer.id], wrap_lines=True)

    st.subheader("Buffer contents")

    if buffer.allocation_action == buffer.last_use_action:
        action = buffer.allocation_action
    else:
        action = st.slider("After action", buffer.allocation_action, buffer.last_use_action)    

    show_buffer(buffer, action)

elif analysis_mode == "View buffers changes for each action":

    executable = st.selectbox("Dispatch", buffer_dump.executables.keys())

    executable_obj = buffer_dump.executables[executable]

    st.subheader("Buffer changes")
    action = st.slider("action", 1, executable_obj.action_count)

    buffer_to_arg = {}

    if action_descs is not None:        
        st.markdown("**Action**")
        st.code(action_descs[executable][action][0][1], wrap_lines=True)
        for action_desc in action_descs[executable][action][1:]:
            st.markdown(action_desc[0] + ":")
            st.code(action_desc[1], wrap_lines=True)        
        buffer_to_arg = {x: idx for idx, x in enumerate(program_arguments[executable][action])}

    job_id = action - 1
    
    st.caption(f"Buffers summary after action {action} executed")

    active_buffers = []
    active_buffer_objs = []

    for buffer in executable_obj.buffers:
        if buffer.allocation_action <= action <= buffer.deallocation_action:

            dump_exists = buffer_dump.exists(buffer, action)

            if buffer.allocation_action == action and buffer.deallocation_action == action:
                state = "🟣 Allocated and deallocated"
            elif buffer.allocation_action == action:
                state = "🟢 Allocated"
            elif buffer.deallocation_action == action:
                state = "🔴 Deallocated"
            elif not dump_exists:
                state = "⚫ No dump available"
            elif buffer_dump.has_changed(action, buffer):
                state = "🟡 Changed"
            else:
                state = "⚪ Unchanged"

            arg = ""
            if buffer.id in buffer_to_arg:
                arg = "arg" + str(buffer_to_arg[buffer.id])

            active_buffers.append({
                "id": buffer.id,
                "argument": arg,
                "shape": buffer.shape,
                "type": buffer.type,
                "size": buffer.size,
                "start address": hex(buffer.address),
                "end address": hex(buffer.address + buffer.size),
                "state": state,
                "sha1": buffer_dump.sha1_buffer(buffer, action) if dump_exists else "n/a"
            })

            active_buffer_objs.append(buffer)

    buffer_selection = st.dataframe(active_buffers, selection_mode="single-row", hide_index=True, key="id", on_select="rerun")
    
    if len(buffer_selection['selection']['rows']) == 0:
        st.stop()

    selected_buffer = active_buffer_objs[buffer_selection['selection']['rows'][0]]

    if not buffer_dump.exists(selected_buffer, action):
        st.warning("No dump available in this action. Showing last dump available.")
        show_buffer(selected_buffer, selected_buffer.last_use_action)
        st.stop()

    view = st.radio("Buffer contents", ["Before action", "After action", "Difference"])

    if view == "Before action":        
        if action == selected_buffer.allocation_action:
            st.warning("No before action available for the allocation action of the buffer.")
        else:            
            show_buffer(selected_buffer, action - 1)
    elif view == "After action":
        if action == selected_buffer.deallocation_action:
            st.warning("No after action available for the deallocation action of the buffer.")
        else:
            show_buffer(selected_buffer, action)
    else:
        if action == selected_buffer.allocation_action or action == selected_buffer.deallocation_action:
            st.warning("No difference available for the allocation or deallocation action of the buffer.")
        else:            
            show_buffer_difference(selected_buffer, action - 1, action)

else:

    def buffer_picker(index):
        executable = st.selectbox(f"Dispatch {index}", buffer_dump.executables.keys(), key=f"executable{index}")
        buffer = st.selectbox(f"Buffer {index}", buffer_dump.executables[executable].buffers, format_func=lambda x: str(x.id) + " (" + x.shape + " " + x.type + ")", key=f"buffer{index}")
        
        if buffer.allocation_action == buffer.last_use_action:
            st.write("After action: " + str(buffer.allocation_action))
            return (buffer, buffer.allocation_action)
        else:
            action = st.slider("After action", buffer.allocation_action, buffer.last_use_action, key=f"action{index}")
            return (buffer, action)

    st.subheader("Compare two buffer states")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Buffer 1**")
        buffer1, action1 = buffer_picker(1)

    with col3:
        st.write("**Buffer 2**")
        buffer2, action2 = buffer_picker(2)

    col1, col2, col3 = st.columns(3)

    with col1:
        show_buffer(buffer1, action1)

    with col2:
        if buffer1.shape != buffer2.shape:
            st.error("Buffers must have the same shape to compare them.")
            st.stop()
        
        show_any_buffer_difference(buffer1.load(action1), buffer2.load(action2))

    with col3:
        show_buffer(buffer2, action2)
    
    


