import os
import logging
import time
import subprocess
from collections.abc import Iterable, Mapping
from typing import Literal

import numpy.typing as npt
try:
    import iree.runtime as iree_rt
    IREE_RT_AVAILABLE = True
except ImportError:
    IREE_RT_AVAILABLE = False

from .base import InferenceRunner
from .utils import TensorInfo, get_inputs_and_outputs, random_inputs_from_info

__all__ = [
    "run_vmfb",
    "profile_vmfb_inference_time",
    "VMFBInferenceRunner",
]


def run_vmfb(
    model_path: str | os.PathLike,
    inputs: Iterable[str],
    outputs: Iterable[str],
    device: str = "torq",
    n_threads: int | None = None,
    iree_binary: str | os.PathLike = None,
) -> float:
    """Run a VMFB model via ``iree-run-module`` and return wall-clock time in ms.

    Args:
        model_path: Path to the ``.vmfb`` file.
        inputs: Input descriptors forwarded as ``--input`` flags.
        outputs: Output descriptors forwarded as ``--output`` flags.
        device: IREE device URI.
        n_threads: Worker thread count (defaults to ``os.cpu_count()``).
        iree_binary: Path to ``iree-run-module`` binary, resolves from PATH if not provided.

    Returns:
        Elapsed wall-clock time in milliseconds.

    Raises:
        RuntimeError: If the ``iree-run-module`` subprocess exits with an error.
    """
    n_threads = n_threads if (isinstance(n_threads, int) and n_threads > 0) else os.cpu_count()
    cmd: list[str] = [
        str(iree_binary) or "iree-run-module",
        f"--device={device}",
        f"--task_topology_group_count={n_threads}",
        f"--task_topology_max_group_count={n_threads}",
        f"--module={model_path}"
    ]
    cmd.extend([f"--input={inp}" for inp in inputs])
    cmd.extend([f"--output={out}" for out in outputs])

    st = time.perf_counter_ns()
    result = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.DEVNULL,   # discard all stdout
        stderr=subprocess.PIPE       # capture stderr only
    )
    et = time.perf_counter_ns()

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to run VMFB model '{str(model_path)}':\n"
            + result.stderr.strip()
        )

    return (et - st) / 1e6


def profile_vmfb_inference_time(
    model_path: str | os.PathLike,
    inputs: Iterable[npt.NDArray] | None = None,
    *,
    n_iters: int = 5,
    do_warmup: bool = True,
    function: str = "main",
    device: str = "torq",
    n_threads: int | None = None,
    load_model_to_mem: bool = True,
    runtime_flags: Iterable[str] = None,
):
    """Load a VMFB model and run inference ``n_iters`` times for profiling.

    Args:
        model_path: Path to the ``.vmfb`` file.
        inputs: Input arrays; generated randomly from model metadata when *None*.
        n_iters: Number of timed inference iterations.
        do_warmup: If True, run one untimed warmup pass first.
        function: Exported function name inside the module.
        device: IREE device URI.
        n_threads: Worker thread count.
        load_model_to_mem: Load model into memory during initialization.
        runtime_flags: Extra IREE runtime flags.

    Returns:
        Average wall-clock inference time in milliseconds.

    Raises:
        ValueError: If *inputs* is None and reflection metadata is unavailable.
    """
    runner = VMFBInferenceRunner(
        model_path,
        function=function,
        device_uri=device,
        n_threads=n_threads,
        load_model_to_mem=load_model_to_mem,
        runtime_flags=runtime_flags
    )
    if not inputs:
        if runner.inputs_info is None:
            raise ValueError("Input tensor info unavailable from model reflection data; please provide inputs explicitly")
        inputs = random_inputs_from_info(runner.inputs_info)
    if do_warmup:
        runner.infer(inputs)
    total_time_ms: float = 0.0
    for _ in range(n_iters):
        runner.infer(inputs)
        total_time_ms += runner.infer_time_ms
    return total_time_ms / n_iters


class VMFBInferenceRunner(InferenceRunner):
    """InferenceRunner backed by the IREE runtime for ``.vmfb`` modules."""

    def __init__(
        self,
        model_path: str | os.PathLike,
        *,
        function: str = "main",
        device_uri: str = "torq",
        n_threads: int | None = None,
        load_method: Literal["preload", "mmap"] = "preload",
        load_model_to_mem: bool = True,
        runtime_flags: Iterable[str] | None = None
    ):
        """InferenceRunner backed by the IREE runtime for ``.vmfb`` modules.

        Args:
            model_path: Path to the ``.vmfb`` file.
            function: Exported function name inside the module.
            device_uri: IREE device identifier.
            n_threads: Worker thread count.
            load_method: ``"preload"`` copies into memory; ``"mmap"`` memory-maps the file.
            load_model_to_mem: Load model into memory during initialization.
            runtime_flags: Extra IREE runtime flags.

        Raises:
            RuntimeError: If the IREE Runtime python API is not installed.
        """
        if not IREE_RT_AVAILABLE:
            raise RuntimeError("IREE Runtime python API not available in environment")

        super().__init__(model_path)
        self._logger = logging.getLogger(__class__.__name__)
        self._function = function
        self._device_uri = device_uri
        self._load_method = load_method
        self._logger.debug("Using IREE runtime at '%s'", str(iree_rt.__file__))

        flags = set()
        if isinstance(n_threads, int) and n_threads > 0:
            flags.add(f"--task_topology_group_count={n_threads}")
            flags.add(f"--task_topology_max_group_count={n_threads}")
            self._logger.debug("Using %d threads for inference", n_threads)
        if device_uri == "torq":
            function = "main"
            flags.add("--torq_hw_type=astra_machina")
        if runtime_flags:
            for flag in runtime_flags:
                flags.add(flag)
        
        if flags:
            iree_rt.flags.parse_flags(*flags)

        self._invoker = None
        self._inputs_info = None
        self._outputs_info = None
        if load_model_to_mem:
            self._load_invoker()
        self._logger.info("Loaded VMFB model '%s'", str(self._model_path))

    @property
    def inputs_info(self) -> list[TensorInfo] | None:
        return self._inputs_info

    @property
    def outputs_info(self) -> list[TensorInfo] | None:
        return self._outputs_info

    def _load_invoker(self):
        """Load the VMFB module and resolve the target function invoker.

        Raises:
            ValueError: If the requested function is not found in the module.
        """
        if self._load_method == "mmap":
            module = iree_rt.load_vm_flatbuffer_file(self._model_path, driver=self._device_uri)
            vm_module = module.vm_module
            self._logger.debug("'%s' loaded via mmap", str(self._model_path))
        else:
            instance = iree_rt.VmInstance()
            device = iree_rt.get_device(self._device_uri)
            hal_module = iree_rt.create_hal_module(instance, device)
            with open(self._model_path, "rb") as f:
                fb = f.read()
            vm_module = iree_rt.VmModule.from_flatbuffer(instance, fb, warn_if_copy=False)
            _ctx = iree_rt.SystemContext(vm_modules=[hal_module, vm_module])
            module = _ctx.modules[vm_module.name]
            self._logger.debug("'%s' loaded via memory copy", str(self._model_path))

        if self._function not in vm_module.function_names:
            raise ValueError(f"Function '{self._function}' not found in '{self._model_path}'")
        self._invoker = module[self._function]
        io_info = get_inputs_and_outputs(self._invoker, self._function)
        if io_info:
            self._inputs_info, self._outputs_info = io_info

    def _infer(self, inputs: Iterable[npt.NDArray] | Mapping[str, npt.NDArray]) -> list[npt.NDArray]:
        """Run a single inference pass and return host-side output arrays."""
        if isinstance(inputs, Mapping):
            inputs = inputs.values()
        if self._invoker is None:
            self._load_invoker()
        result: iree_rt.DeviceArray | tuple[iree_rt.DeviceArray] = self._invoker(*inputs)
        if isinstance(result, tuple):
            return [r.to_host() for r in result]
        return [result.to_host()]
