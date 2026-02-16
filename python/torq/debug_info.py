import logging
import os
from typing import List, Optional, Dict, Set
from dataclasses import dataclass, field
from iree.compiler.ir import Context, Operation, Value, Location, Block, BlockArgument, Module
import pandas as pd
from collections import OrderedDict
from enum import Enum
from abc import ABC, abstractmethod
from functools import cached_property
import re
import torq.utils.location as loc_utils


logger = logging.getLogger("torq.model_profiler.annotation")

# Clock frequency in MHz for cycle-to-time conversion
CLOCK_FREQ_MHZ = 800

def cycles_to_ns(cycles: int, frequency_mhz: int):
    """Convert cycles to nanoseconds given a clock frequency in MHz."""

    return int(cycles * 1000 / frequency_mhz)


@dataclass
class EventData:
    """
    Represents a row in the runtime profiling log for a given event.

    The profiling log contains two events per action: one for the beginning of the action and one for the end.

    """

    action_id: int
    event: str
    timestamp_ns: int


@dataclass
class InvocationEventData:
    invocation_id: int
    dispatch_name: str
    event_data: List[EventData] = field(default_factory=list)


def parse_profiling_log(profiling_file: str) -> List[InvocationEventData]:
    """
    Parse the profiling log produced by the runtime (with --torq_profile_host) and return 
    a list of InvocationEventData objects representing the events in the log.
    """


    profiling_data = pd.read_csv(profiling_file, sep=',')
    
    invocations = {}

    for _, row in profiling_data.iterrows():

        dispatch_name = str(row['dispatch_name']).strip()
        invocation_id = int(row['invocation_id'])

        invocation_data = invocations.get(invocation_id)

        if invocation_data is None:
            invocation_data = InvocationEventData(invocation_id=invocation_id, dispatch_name=dispatch_name)
            invocations[invocation_id] = invocation_data

        entry = EventData(
            action_id=int(row['action_id']),
            event=row['event'],
            timestamp_ns=int(row['timestamp_us']) * 1000)
        
        invocation_data.event_data.append(entry)

    return invocations.values()


class ActionType(str, Enum):
    """
    Type of runtime action as present in runtime profiling log
    """
    ALLOC = "ALLOC"
    DEALLOC = "DEALLOC"
    NSS_START = "NSS_START"
    NSS_WAIT = "NSS_WAIT"
    HOST_START = "HOST_START"
    HOST_WAIT = "HOST_WAIT"
    HOST_COPY = "HOST_COPY"
    DISPATCH = "DISPATCH"


@dataclass
class ActionProfilingData:
    """
    Profiling information for a given runtime action, this is derived from the original
    profiling log by matching the begin and end events for each action id.
    """
    action_id: int
    action_type: ActionType
    start_time_ns: int
    end_time_ns: int

    @property
    def total_time_ns(self) -> int:
        return self.end_time_ns - self.start_time_ns


def generate_action_profiling_data(raw_entries: List[EventData]) -> OrderedDict[int, ActionProfilingData]:
    """
    Generate ActionProfilingData objects for each action by matching the start and end events in the profiling log.
    """

    result = {}

    for entry in raw_entries:        

        action_name, event_type = entry.event.rsplit("_", 1)

        action_entry = result.get(entry.action_id)

        if action_entry is None:          

            action_entry = ActionProfilingData(
                action_id=entry.action_id,
                action_type=ActionType(action_name),
                start_time_ns=None,
                end_time_ns=None
            )

            result[entry.action_id] = action_entry

        if event_type == "BEGIN":
            action_entry.start_time_ns = entry.timestamp_ns            
        elif event_type == "END":
            action_entry.end_time_ns = entry.timestamp_ns
        else:
            raise ValueError(f"Unknown event type {event_type} in profiling log")
    
    return result



class ActionDebugInfo:

    def __init__(self, dispatch: 'DispatchDebugInfo', operation: Operation):
        self.dispatch = dispatch
        self.operation = operation
        self.workunit = None

    @property
    def is_start_nss_action(self) -> bool:        
        return self.operation.name == "torq_hl.start_program" and str(self.operation.operands[0].type) == "!torq_hl.invocation<nss>"

    @property
    def nss_program(self) -> Optional[Value]:
        if not self.is_start_nss_action:
            return None
        
        invocation = self.operation.operands[0]
        return invocation.owner.operands[0].owner

    @property
    def nss_invocation_args(self) -> Optional[List[Value]]:
        if not self.is_start_nss_action:
            return None
        
        args_offset = 1 + self.operation.attributes["operandSegmentSizes"][1]        
        return self.operation.operands[args_offset:]

    @property
    def action_id(self) -> int:
        return self.operation.attributes["torq-action-id"].value
    
    @property
    def start_time_ns(self) -> Optional[int]:
        return self.dispatch.get_operation_start_time_ns(self.operation)
        
    @property
    def end_time_ns(self) -> Optional[int]:
        return self.dispatch.get_operation_end_time_ns(self.operation)        

    @property
    def total_time_ns(self) -> Optional[int]:
        start_time = self.start_time_ns
        end_time = self.end_time_ns

        if start_time is None or end_time is None:
            return None
        
        return end_time - start_time
    
    @property
    def location(self) -> Location:
        return self.operation.location


class Executor(Enum):
    HOST = "HOST"
    HOST_COPY = "HOST_COPY"
    NSS = "NSS"
    CSS = "CSS"
    SLICE = "SLICE"
    CDMA = "CDMA"
    DMA_IN = "DMA_IN"
    DMA_OUT = "DMA_OUT"
    NSS_CFG = "NSS_CFG"


class WorkUnitDebugInfo(ABC):
    """
    A work unit is a unit of execution on a given executor (e.g. Slice / NSS CFG / Host )

    A work unit starts at a given start_time, is ready at a given ready_time and 
    releases the executor at a given end_time.
    """

    def __init__(self, dispatch: 'DispatchDebugInfo', start_operation: Operation, end_operation: Operation):
        self.dispatch = dispatch
        self.start_operation = start_operation
        self.end_operation = end_operation

    @property
    def executor_instance_id(self) -> int:
        return 0

    @property
    @abstractmethod
    def executor(self) -> Executor:
        pass

    @property
    def location(self) -> Location:
        return self.start_operation.location

    @property
    def start_time_ns(self) -> Optional[int]:
        return self.dispatch.get_operation_start_time_ns(self.start_operation)
    
    @property
    def end_time_ns(self) -> Optional[int]:
        return self.dispatch.get_operation_end_time_ns(self.end_operation)

    @property
    def total_time_ns(self) -> Optional[int]:
        start_time = self.start_time_ns
        end_time = self.end_time_ns

        if start_time is not None and end_time is not None:
            return end_time - start_time
        
        return None
    
    @property
    def ready_time_ns(self) -> Optional[int]:
        """
        Time at which the work unit has completed its execution, this may be earlier
        than the end_time in case the execution results are ready but need to be
        acknowledged by the caller to free the executor.
        """
        start_time = self.start_time_ns
        async_duration = self.dispatch.get_operation_async_duration_ns(self.start_operation)

        if start_time is not None and async_duration is not None:
            return start_time + async_duration
        
        return None
    
    @classmethod
    def pretty_name(cls):
        if hasattr(cls, "_pretty_name"):
            return cls._pretty_name
        
        name = cls.__name__.replace("WorkUnitDebugInfo", "")        
        cls._pretty_name = re.sub(r'([A-Z])', r' \1', name).strip()

        return cls._pretty_name

    @property
    def pretty_print(self):
        return self.pretty_name() + " | " + self.dispatch.debug_info.pretty_print_location(self.location)


class HostProgramWorkUnitDebugInfo(WorkUnitDebugInfo):

    def __init__(self, dispatch, start_operation, end_operation):
        super().__init__(dispatch, start_operation, end_operation)

    @property
    def invocation(self) -> Value:
        return self.start_operation.operands[0]
    
    @property
    def executor(self) -> Executor:
        return Executor.HOST


class HostCopyWorkUnitDebugInfo(WorkUnitDebugInfo):
    
    def __init__(self, dispatch, start_operation, end_operation):
        super().__init__(dispatch, start_operation, end_operation)

    @property
    def executor(self) -> Executor:
        return Executor.HOST_COPY


class NssProgramWorkUnitDebugInfo(WorkUnitDebugInfo):
    
    def __init__(self, dispatch, start_operation, end_operation):
        super().__init__(dispatch, start_operation, end_operation)
        self.slice_used = [False, False]
        self.dma_out_used = False
        self.dma_in_used = False
        self.cdma_used = False
        self.css_used = False
        self.related_workunits = set()
        self.start_action : Optional[ActionDebugInfo] = None
        self.end_action : Optional[ActionDebugInfo] = None

    @property
    def invocation(self) -> Value:
        return self.start_operation.operands[0]

    @property
    def executor(self) -> Executor:
        return Executor.NSS

    @property
    def related_slice_invocations_names(self) -> List[str]:
        invocation_names = []

        for related_workunit in self.related_workunits:
            if isinstance(related_workunit, SliceProgramWorkUnitDebugInfo):
                invocation_names.append(related_workunit.invocation_name)

        return invocation_names

    @property
    def program_name(self) -> str:
        invocation = self.start_operation.operands[0]
        return invocation.owner.attributes["name"].value

    @property
    def job_id(self) -> Optional[int]:
        invocation = self.start_operation.operands[0]        
        return invocation.owner.operation.attributes["torq-job-id"].value

    @property
    def pretty_print(self):
        details = self.pretty_name() + " | " + self.program_name + " | " + self.dispatch.debug_info.pretty_print_location(self.location)

        invocation_names = self.related_slice_invocations_names

        if len(invocation_names) > 0:
            details += f"| Torq Kernels: [{', '.join(invocation_names)}]"

        details += f" | Job: {self.job_id}"
        details += f" | Start Action: {self.start_action.action_id if self.start_action else 'N/A'}"
        details += f" | Wait Action: {self.end_action.action_id if self.end_action else 'N/A'}"

        return details


class NssCfgWorkUnitDebugInfo(WorkUnitDebugInfo):
    
    def __init__(self, dispatch, task_op):
        super().__init__(dispatch, task_op, task_op)
    
    @property
    def executor(self) -> Executor:
        return Executor.NSS_CFG
    
    @property
    def task_id(self) -> int:
        return self.start_operation.attributes["torq-task-id"].value

    @property
    def pretty_print(self):
        return f"{self.pretty_name()} | Task: {self.task_id} | " + self.dispatch.debug_info.pretty_print_location(self.location)
    

class NssManagedWorkUnitDebugInfo(WorkUnitDebugInfo):
    """
    Work units that are managed by the NSS are started and stopped
    by operations inside a torq_hw.nss_task operation so the 
    start_time/end_time is found differently than other 
    work units where the timing is directly on the start 
    and end operations (e.g. host programs and host copies)
    """

    @property
    def start_time_ns(self) -> Optional[int]:
        return self.dispatch.get_operation_start_time_ns(self.start_operation.parent)
    
    @property
    def end_time_ns(self) -> Optional[int]:        
        return self.dispatch.get_operation_end_time_ns(self.end_operation.parent)


class CssProgramWorkUnitDebugInfo(NssManagedWorkUnitDebugInfo):
    
    def __init__(self, dispatch, start_operation, end_operation):
        super().__init__(dispatch, start_operation, end_operation)
    
    @property
    def executor(self) -> Executor:
        return Executor.CSS


class SliceProgramWorkUnitDebugInfo(NssManagedWorkUnitDebugInfo):
    
    def __init__(self, dispatch, start_operation, end_operation):
        super().__init__(dispatch, start_operation, end_operation)

    @property
    def executor_instance_id(self) -> int:
        return self.start_operation.attributes["id"].value

    @property
    def executor(self) -> Executor:
        return Executor.SLICE

    @property
    def invocation_name(self) -> str:
        invocation = self.dispatch.resolve_argument(self.start_operation, 0)
        return invocation.owner.attributes["name"].value[len("slice_program_torq_hl.") :]
    
    @property
    def pretty_print(self):
        return f"{self.pretty_name()} | {self.invocation_name}"


class CdmaWorkUnitDebugInfo(NssManagedWorkUnitDebugInfo):
    
    def __init__(self, dispatch, start_operation, end_operation):
        super().__init__(dispatch, start_operation, end_operation)

    @property
    def executor(self) -> Executor:
        return Executor.CDMA


class DmaInWorkUnitDebugInfo(NssManagedWorkUnitDebugInfo):
    
    def __init__(self, dispatch, start_operation, end_operation):
        super().__init__(dispatch, start_operation, end_operation)

    @property
    def executor(self) -> Executor:
        return Executor.DMA_IN


class DmaOutWorkUnitDebugInfo(NssManagedWorkUnitDebugInfo):
    
    def __init__(self, dispatch, start_operation, end_operation):
        super().__init__(dispatch, start_operation, end_operation)

    @property
    def executor(self) -> Executor:
        return Executor.DMA_OUT


@dataclass
class PendingWorkUnits:
    host_program: Optional[HostProgramWorkUnitDebugInfo] = None
    nss_program: Optional[NssProgramWorkUnitDebugInfo] = None
    slice_programs: List[Optional[SliceProgramWorkUnitDebugInfo]] = field(default_factory=lambda: [None, None])
    css_program: Optional[CssProgramWorkUnitDebugInfo] = None
    cdma: Optional[CdmaWorkUnitDebugInfo] = None
    dma_in: Optional[DmaInWorkUnitDebugInfo] = None
    dma_out: Optional[DmaOutWorkUnitDebugInfo] = None


class OriginalLocationDebugInfo:

    def __init__(self, dispatch: 'DispatchDebugInfo', location: loc_utils.FileLineColLocation):
        self.dispatch = dispatch
        self.location = location
        self.workunits : OrderedDict[WorkUnitDebugInfo] = OrderedDict()


class BlockDebugInfo:
    
    def __init__(self, dispatch: 'DispatchDebugInfo', block: Block, args: List[Value]):
        self.dispatch = dispatch
        self.block = block
        self.args = args
    
    def get_argument(self, value: Value) -> Value:
        try:
            arg = BlockArgument(value)
            arg_index = arg.arg_number
            return self.args[arg_index]
        except ValueError:
            return value


@dataclass
class TimeDebugInfo:
    """When this operation will start execution"""
    start_time_ns: Optional[int]

    """When this operation will end execution"""
    end_time_ns: Optional[int]

    """How much time the workunit triggered by this operation will take to complete from the start_time_ns timestamp"""
    async_duration_ns: Optional[int] = None


class DispatchDebugInfo:
    def __init__(self, debug_info: 'DebugInfo', module: Operation):
        # ! IMPORTANT: we need to keep a reference to the module or it
        # will be garbage collected and all operations will lose their
        # parent and thus become invalid
        self.module = module

        self.debug_info = debug_info
        self.dispatch = module.body.operations[0]                

        self.operation_times_ns = {}
        self.actions = self._load_actions()                        
        self.nss_blocks_info = self._load_nss_blocks_info()
        self.workunits = self._load_host_workunits()
        self.original_locations = self._load_original_locations()        

    def get_operation_start_time_ns(self, operation: Operation) -> Optional[int]:
        time_info = self.operation_times_ns.get(operation)
        if time_info is not None:
            return time_info.start_time_ns
        return None
    
    def get_operation_end_time_ns(self, operation: Operation) -> Optional[int]:
        time_info = self.operation_times_ns.get(operation)
        if time_info is not None:
            return time_info.end_time_ns
        return None
    
    def get_operation_async_duration_ns(self, operation: Operation) -> Optional[int]:
        time_info = self.operation_times_ns.get(operation)
        if time_info is not None:
            return time_info.async_duration_ns
        return None

    def _load_actions(self) -> OrderedDict[int, ActionDebugInfo]:
        actions = []                
        for region in self.dispatch.operation.regions:
            for block in region:
                for op in block.operations:
                    if "torq-action-id" in op.operation.attributes:                        
                        action = ActionDebugInfo(self, op)                        
                        actions.append(action)                        

        return OrderedDict([(x.action_id, x) for x in sorted(actions, key=lambda a: a.action_id)])        


    def _load_nss_blocks_info(self) -> Dict[Block, BlockDebugInfo]:
        blocks = {}

        for action in self.actions.values():
            
            if not action.is_start_nss_action:
                continue

            program = action.nss_program

            block_info = BlockDebugInfo(self, program.regions[0].blocks[0], action.nss_invocation_args)
            blocks[block_info.block] = block_info

            finished = False

            while not finished:
                for op in block_info.block.operations:                    
                    if op.name == "torq_hl.next":
                        next_block_args = [block_info.get_argument(operand) for operand in op.operation.operands]        
                        block_info = BlockDebugInfo(self, op.successors[0], next_block_args)
                        blocks[op.successors[0]] = block_info
                    elif op.name == "torq_hl.return":
                        finished = True
                        break

        return blocks

    def _load_host_workunits(self) -> List[WorkUnitDebugInfo]:

        workunits = []

        pending = PendingWorkUnits()

        for action in self.actions.values():
            if action.operation.name == "torq_hl.start_program":
                invocation = action.operation.operands[0]
                invocation_type = str(invocation.type)

                if invocation_type == "!torq_hl.invocation<host>":
                    workunit = HostProgramWorkUnitDebugInfo(self, action.operation, None)
                    pending.host_program = workunit
                elif invocation_type == "!torq_hl.invocation<nss>":
                    workunit = NssProgramWorkUnitDebugInfo(self, action.operation, None)
                    pending.nss_program = workunit

                    pending.nss_program.start_action = action
                    pending.nss_program.slice_used = [x is not None for x in pending.slice_programs]
                    pending.nss_program.dma_in_used = pending.dma_in is not None
                    pending.nss_program.dma_out_used = pending.dma_out is not None
                    pending.nss_program.cdma_used = pending.cdma is not None
                    pending.nss_program.css_used = pending.css_program is not None

                    self._load_nss_workunits(invocation, workunits, pending)
                else:
                    raise ValueError(f"Unknown invocation type {invocation_type} at action id {action.action_id}")
                
                action.workunit = workunit

                workunits.append(workunit)

            elif action.operation.name == "torq_hl.wait_program":

                invocation = action.operation.operands[0]
                invocation_type = str(invocation.type)
                
                if invocation_type == "!torq_hl.invocation<host>":
                    workunit = pending.host_program
                    pending.host_program = None
                elif invocation_type == "!torq_hl.invocation<nss>":
                    workunit = pending.nss_program
                    workunit.end_action = action
                    pending.nss_program = None
                else:
                    raise ValueError(f"Unknown invocation type {invocation_type} at action id {action.action_id}")

                workunit.end_operation = action.operation
                action.workunit = workunit

            elif action.operation.name == "torq_hl.host_copy":
                workunit = HostCopyWorkUnitDebugInfo(self, action.operation, action.operation)
                action.workunit = workunit

                workunits.append(workunit)

            else:
                # skip all other operations (e.g. memrefs alloc/dealloc and subviews)
                continue

        return workunits

    def _get_task_operations_of_type(self, op, task_type):
        return [o for o in op.regions[0].blocks[0].operations if o.name == task_type]

    def _load_nss_workunits(self, invocation: Value, workunits, pending: PendingWorkUnits):

        nss_program = invocation.owner.operands[0].owner

        for block in nss_program.regions[0].blocks:
            for op in block.operations:

                if op.name == "torq_hw.nss_task":

                    workunits.append(NssCfgWorkUnitDebugInfo(self, op))

                    for slice_start in self._get_task_operations_of_type(op, "torq_hw.slice_start"):
                        workunit = SliceProgramWorkUnitDebugInfo(self, slice_start, None)
                        pending.slice_programs[workunit.executor_instance_id] = workunit
                        pending.nss_program.slice_used[workunit.executor_instance_id] = True
                        pending.nss_program.related_workunits.add(workunit)
                        
                    for dma_in_start in self._get_task_operations_of_type(op, "torq_hw.dma_in_start"):
                        pending.dma_in = DmaInWorkUnitDebugInfo(self, dma_in_start, None)
                        pending.nss_program.dma_in_used = True
                        pending.nss_program.related_workunits.add(pending.dma_in)
                        
                    for dma_out_start in self._get_task_operations_of_type(op, "torq_hw.dma_out_start"):
                        pending.dma_out = DmaOutWorkUnitDebugInfo(self, dma_out_start, None)
                        pending.nss_program.dma_out_used = True
                        pending.nss_program.related_workunits.add(pending.dma_out)
                        
                    for cdma_start in self._get_task_operations_of_type(op, "torq_hw.cdma_start"):
                        pending.cdma = CdmaWorkUnitDebugInfo(self, cdma_start, None)
                        pending.nss_program.cdma_used = True
                        pending.nss_program.related_workunits.add(pending.cdma)

                    for css_start in self._get_task_operations_of_type(op, "torq_hw.css_start"):
                        pending.css_program = CssProgramWorkUnitDebugInfo(self, css_start, None)
                        pending.nss_program.css_used = True
                        pending.nss_program.related_workunits.add(pending.css_program)

                    for slice_wait in self._get_task_operations_of_type(op, "torq_hw.slice_wait"):
                        slice_id = slice_wait.attributes["id"].value
                        workunit = pending.slice_programs[slice_id]
                        pending.slice_programs[slice_id] = None
                        workunit.end_operation = slice_wait
                        workunits.append(workunit)
                        pending.nss_program.related_workunits.add(workunit)

                    for dma_in_end in self._get_task_operations_of_type(op, "torq_hw.dma_in_wait"):
                        workunit = pending.dma_in
                        pending.dma_in = None
                        workunit.end_operation = dma_in_end
                        workunits.append(workunit)
                        pending.nss_program.related_workunits.add(workunit)

                    for dma_out_end in self._get_task_operations_of_type(op, "torq_hw.dma_out_wait"):
                        workunit = pending.dma_out
                        pending.dma_out = None
                        workunit.end_operation = dma_out_end
                        workunits.append(workunit)
                        pending.nss_program.related_workunits.add(workunit)

                    for cdma_end in self._get_task_operations_of_type(op, "torq_hw.cdma_wait"):
                        workunit = pending.cdma
                        pending.cdma = None
                        workunit.end_operation = cdma_end
                        workunits.append(workunit)
                        pending.nss_program.related_workunits.add(workunit)

                    for css_end in self._get_task_operations_of_type(op, "torq_hw.css_wait"):
                        workunit = pending.css_program
                        pending.css_program = None
                        workunit.end_operation = css_end
                        workunits.append(workunit)
                        pending.nss_program.related_workunits.add(workunit)

                else:
                    # skip all other operations (e.g. memref allocs/deallocs and subviews and next operations)
                    continue


    def _load_original_locations(self) -> List[OriginalLocationDebugInfo]:
        original_locations_debug_info = {}

        for workunit in self.workunits:

            original_location = workunit.location            

            all_locations = loc_utils.extract_all_file_locations(str(original_location))

            for file_location in all_locations:

                debug_info = original_locations_debug_info.get(file_location)

                if debug_info is None:
                    debug_info = OriginalLocationDebugInfo(self, file_location)
                    original_locations_debug_info[file_location] = debug_info
                
                debug_info.workunits[workunit] = None

        # return a list of original locations sorted by filename and then line and column number to ensure a deterministic order
        sorted_debug_info = sorted(original_locations_debug_info.values(), key=lambda d: (d.location.file, d.location.line, d.location.col))

        return sorted_debug_info


    def resolve_argument(self, operation: Operation, operand_index: int) -> Value:
        operand = operation.operands[operand_index]

        try:
            block_arg = BlockArgument(operand)
            block_info = self.nss_blocks_info[block_arg.owner]
            return block_info.get_argument(operand)
        except ValueError:
            return operand


    def load_runtime_events(self, events_data: List[EventData]):
        """
        Annotates the operations in the dispatch with the timing information from a profiling log
        that was created by running torq-run-module with the --torq_profile_host option
        """

        actions_profile = generate_action_profiling_data(events_data)

        # Annotate the action operations with the timing data from the profiling log
        for action_id, profile_data in actions_profile.items():
            
            # the action id may be -1 if the log inforamtion corresponds to DISPATCH_{BEGIN/END} event
            if action_id == -1:
                continue

            action = self.actions.get(action_id)

            self.operation_times_ns[action.operation] = TimeDebugInfo(start_time_ns=profile_data.start_time_ns, end_time_ns=profile_data.end_time_ns)

    def _update_time_ns(self, operation: Operation, frequency_mhz: int = CLOCK_FREQ_MHZ):

        start_time_ns = None
        if "torq-start-time-cycles" in operation.attributes:
            start_cycles = operation.attributes["torq-start-time-cycles"].value
            start_time_ns = cycles_to_ns(start_cycles, frequency_mhz)

        end_time_ns = None

        if "torq-end-time-cycles" in operation.attributes:
            end_cycles = operation.attributes["torq-end-time-cycles"].value
            end_time_ns = cycles_to_ns(end_cycles, frequency_mhz)

        async_duration_ns = None

        if "torq-async-duration-cycles" in operation.attributes:
            async_duration_cycles = operation.attributes["torq-async-duration-cycles"].value
            async_duration_ns = cycles_to_ns(async_duration_cycles, frequency_mhz)        

        self.operation_times_ns[operation] = TimeDebugInfo(start_time_ns=start_time_ns, end_time_ns=end_time_ns, async_duration_ns=async_duration_ns)


    def infer_runtime_profile_from_cycles(self, frequency_mhz: int = CLOCK_FREQ_MHZ):

        for workunit in self.workunits:
            self._update_time_ns(workunit.start_operation, frequency_mhz)
            self._update_time_ns(workunit.end_operation, frequency_mhz)

    @cached_property
    def start_time_ns(self):
        start_times = [workunit.start_time_ns for workunit in self.workunits if workunit.start_time_ns is not None]
        return min(start_times) if start_times else None

    @cached_property
    def end_time_ns(self):
        start_times = [workunit.start_time_ns for workunit in self.workunits if workunit.start_time_ns is not None]
        return min(start_times) if start_times else None

    @property
    def total_time_ns(self):
        if self.start_time_ns is not None and self.end_time_ns is not None:
            return self.end_time_ns - self.start_time_ns
        return None


class DebugInfo:
    """
    Represents the debug information for a model stored in a directory, this can be generated during
    compilation by passing the --torq-debug-info flag to the compiler.
    """

    def __init__(self, debug_info_path: str):
        self.context = Context()
        self.path = debug_info_path
        self._dispatches = {}
        self._cached_input_files = {}

    def get_dispatch(self, dispatch_name: str) -> DispatchDebugInfo:
        """
        Get the dispatch debug info for a given dispatch name
        """

        if dispatch_name in self._dispatches:
            return self._dispatches[dispatch_name]

        dispatch_file = f"{self.path}/{dispatch_name}.mlirb"

        if not os.path.exists(dispatch_file):
            raise ValueError(f"Dispatch debug info {dispatch_file} not found")

        with open(dispatch_file, 'rb') as mlir_file:
            mlir_content = mlir_file.read()
            
        module = Module.parse(mlir_content, context=self.context)    

        self._dispatches[dispatch_name] = DispatchDebugInfo(self, module)

        return self._dispatches[dispatch_name]

    @property
    def dispatch_names(self):
        return [f[:-6] for f in os.listdir(self.path) if f.endswith(".mlirb")]

    def pretty_print_location(self, loc: Location):
        ploc = loc_utils.parse_location(str(loc))

        def pretty_print_parsed_location(ploc: loc_utils.LocationAST):

            if isinstance(ploc, loc_utils.CallsiteLocation):
                file_location = ploc.callee
            elif isinstance(ploc, loc_utils.FileLineColLocation):
                file_location = ploc
            elif isinstance(ploc, loc_utils.FusedLocation):
                return "+".join([pretty_print_parsed_location(l) for l in ploc.locations])
            else:
                return str(loc)

            return f"@L{file_location.line}C{file_location.col}"
        
        return pretty_print_parsed_location(ploc)

    def _extract_operator(self, line_no: int, line: str): 
        line = line.strip()

        if not line:
            return None
        
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
                return match.group(1) + "@L" + str(line_no)

        # Handle TOSA operators (both quoted and unquoted forms)
        if line.startswith('"tosa.'):
            match = re.search(r'"([^"]+)"', line)
            if match:
                return match.group(1) + "@L" + str(line_no)
            
        elif line.startswith('tosa.'):
            # Return the first token (e.g., "tosa.conv2d")
            return line.split()[0] + "@L" + str(line_no)

        return None

    def get_original_operators(self, location: Location) -> List[str]:

        file_locations = loc_utils.extract_all_file_locations(str(location))

        operators = []

        for file_location in file_locations:

            lines = self._cached_input_files.get(file_location.file)

            if lines is None:
                with open(file_location.file.strip('"'), "r") as f:
                    lines = f.readlines()

                self._cached_input_files[file_location.file] = lines

            line = lines[int(file_location.line) - 1]  # line numbers are 1-based in the location

            operator = self._extract_operator(int(file_location.line), line)

            if operator is not None:
                operators.append(operator)            

        return operators
