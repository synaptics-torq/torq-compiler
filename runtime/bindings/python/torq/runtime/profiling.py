"""Resource profiling utilities for inference workloads.

Provides DRAM and CPU sampling backed by ``/proc`` interfaces available
on Yocto Linux targets.
"""

import os
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Literal

import numpy.typing as npt

from .utils import random_inputs_from_info
from .vmfb import VMFBInferenceRunner

__all__ = [
    "ProfileStats",
    "ResourceSampler",
    "profile_vmfb_resources",
]

TORQ_INFERENCE_TIME_PATH = "/sys/class/misc/torq/statistics/torq_inference_time"


@dataclass
class ProfileStats:
    """Profiling statistics."""

    avg_inference_time_ms: float = 0.0
    avg_dram_footprint_bytes: int = 0
    peak_dram_footprint_bytes: int = 0
    avg_anon_mem_bytes: int = 0
    peak_anon_mem_bytes: int = 0
    avg_cpu_percent: float = 0.0
    """Average CPU utilisation percentage. Measured process-wide; includes
    the Python interpreter, the sampling thread, and any other activity
    in the process."""
    npu_inference_time_us: int | None = None
    """NPU execution time accumulated by the Torq driver during profiling."""
    avg_npu_load_percent: float | None = None
    """Average NPU utilisation percentage during the profiling window."""

    def summary(self) -> str:
        """Return a human-readable summary of the profiling results."""
        def _mb(b: int) -> str:
            return f"{b / (1024 * 1024):.1f} MB"

        lines = [
            f"Avg inference time:  {self.avg_inference_time_ms:.3f} ms",
            "",
            "Process-wide resource usage (includes Python overhead):",
            f"  Avg DRAM footprint:  {_mb(self.avg_dram_footprint_bytes)}  (includes mmap'd model file)",
            f"  Peak DRAM footprint: {_mb(self.peak_dram_footprint_bytes)}",
            f"  Avg memory usage:    {_mb(self.avg_anon_mem_bytes)}  (heap/stack only, excludes model file)",
            f"  Peak memory usage:   {_mb(self.peak_anon_mem_bytes)}",
            f"  Avg CPU usage:       {self.avg_cpu_percent:.1f}%",
        ]
        if self.avg_npu_load_percent is not None:
            lines.append(f"  Avg NPU usage:       {self.avg_npu_load_percent:.1f}%")
        else:
            lines.append("  Avg NPU usage:       unavailable")
        return "\n".join(lines)


def _read_process_rss(pid: int) -> int:
    """Return resident set size in bytes for *pid* via ``/proc``."""
    try:
        with open(f"/proc/{pid}/statm", "r") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        return 0


def _read_process_anon_rss(pid: int) -> int:
    """Return anonymous RSS in bytes for *pid* via ``/proc``.

    This excludes file-backed pages (e.g. mmap'd model data) and measures
    only heap, stack, and other anonymous allocations.
    """
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) * 1024  # value is in kB
    except (OSError, ValueError):
        pass
    return 0


def _read_process_cpu_times(pid: int) -> tuple[float, float] | None:
    """Return (user+sys jiffies, total system jiffies) for *pid*."""
    try:
        with open(f"/proc/{pid}/stat", "r") as f:
            parts = f.read().split()
        utime = int(parts[13])
        stime = int(parts[14])
        with open("/proc/stat", "r") as f:
            cpu_line = f.readline()
        total = sum(int(x) for x in cpu_line.split()[1:])
        return (utime + stime, total)
    except (OSError, IndexError, ValueError):
        return None


def _read_npu_inference_time_us(path: str | os.PathLike) -> int | None:
    """Return cumulative Torq NPU inference time in microseconds."""
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


class ResourceSampler:
    """Background thread that samples process resources and optional NPU load.

    Usage::

        sampler = ResourceSampler(os.getpid())
        sampler.start()
        # ... workload ...
        sampler.stop()
        print(sampler.avg_rss, sampler.peak_rss)
        print(sampler.avg_anon_rss, sampler.peak_anon_rss)
        print(sampler.avg_cpu_percent)
        print(sampler.avg_npu_load_percent)
    """

    def __init__(
        self,
        pid: int,
        interval: float = 0.01,
        npu_inference_time_path: str | os.PathLike | None = TORQ_INFERENCE_TIME_PATH,
    ):
        self._pid = pid
        self._interval = interval
        self._npu_inference_time_path = npu_inference_time_path
        self._npu_start_us: int | None = None
        self._npu_end_us: int | None = None
        self._npu_start_time_ns: int | None = None
        self._npu_end_time_ns: int | None = None
        self._stop = threading.Event()
        self._rss_samples: list[int] = []
        self._anon_rss_samples: list[int] = []
        self._cpu_start: tuple[float, float] | None = None
        self._cpu_end: tuple[float, float] | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._cpu_start = _read_process_cpu_times(self._pid)
        if self._npu_inference_time_path is not None:
            self._npu_start_us = _read_npu_inference_time_us(self._npu_inference_time_path)
            self._npu_start_time_ns = perf_counter_ns()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        if self._npu_inference_time_path is not None:
            self._npu_end_time_ns = perf_counter_ns()
            self._npu_end_us = _read_npu_inference_time_us(self._npu_inference_time_path)
        self._cpu_end = _read_process_cpu_times(self._pid)

    def _run(self):
        while not self._stop.is_set():
            rss = _read_process_rss(self._pid)
            if rss > 0:
                self._rss_samples.append(rss)
            anon = _read_process_anon_rss(self._pid)
            if anon > 0:
                self._anon_rss_samples.append(anon)
            self._stop.wait(self._interval)

    @property
    def avg_rss(self) -> int:
        return int(sum(self._rss_samples) / len(self._rss_samples)) if self._rss_samples else 0

    @property
    def peak_rss(self) -> int:
        return max(self._rss_samples) if self._rss_samples else 0

    @property
    def avg_anon_rss(self) -> int:
        return int(sum(self._anon_rss_samples) / len(self._anon_rss_samples)) if self._anon_rss_samples else 0

    @property
    def peak_anon_rss(self) -> int:
        return max(self._anon_rss_samples) if self._anon_rss_samples else 0

    @property
    def avg_cpu_percent(self) -> float:
        if self._cpu_start and self._cpu_end:
            proc_delta = self._cpu_end[0] - self._cpu_start[0]
            total_delta = self._cpu_end[1] - self._cpu_start[1]
            if total_delta > 0:
                ncpus = os.cpu_count() or 1
                return (proc_delta / total_delta) * 100.0 * ncpus
        return 0.0

    @property
    def npu_inference_time_us(self) -> int | None:
        if self._npu_start_us is None or self._npu_end_us is None:
            return None
        if self._npu_end_us < self._npu_start_us:
            return None
        return self._npu_end_us - self._npu_start_us

    @property
    def avg_npu_load_percent(self) -> float | None:
        if (
            self.npu_inference_time_us is None
            or self._npu_start_time_ns is None
            or self._npu_end_time_ns is None
        ):
            return None
        window_us = (self._npu_end_time_ns - self._npu_start_time_ns) / 1000.0
        if window_us <= 0:
            return None
        return (self.npu_inference_time_us * 100.0) / window_us


def profile_vmfb_resources(
    model_path: str | os.PathLike | VMFBInferenceRunner,
    inputs: Iterable[npt.NDArray] | None = None,
    *,
    n_iters: int = 5,
    do_warmup: bool = True,
    function: str = "main",
    device: str = "torq",
    n_threads: int | None = None,
    load_method: Literal["preload", "mmap"] = "preload",
    load_model_to_mem: bool = True,
    runtime_flags: Iterable[str] = None,
    device_io: bool = False,
) -> ProfileStats:
    """Profile VMFB inference including DRAM, CPU, and NPU statistics.

    Args:
        model_path: Path to the ``.vmfb`` file or an existing
            :class:`~torq.runtime.VMFBInferenceRunner`.
        inputs: Input arrays; generated randomly from model metadata when *None*.
        n_iters: Number of timed inference iterations.
        do_warmup: If True, run one untimed warmup pass first.
        function: Exported function name inside the module.
        device: IREE device URI.
        load_method: ``"preload"`` copies into memory; ``"mmap"`` memory-maps the file.
        n_threads: Worker thread count.
        load_model_to_mem: Load model into memory during initialization.
        runtime_flags: Extra IREE runtime flags.
        device_io: Pass inputs and outputs as :class:`~iree.runtime.DeviceArray` objects instead of NumPy arrays.

    Returns:
        A :class:`ProfileStats` with average inference time, DRAM usage,
        CPU utilisation, and NPU utilisation when available.

    Raises:
        ValueError: If *inputs* is None and reflection metadata is unavailable.
    """
    if isinstance(model_path, VMFBInferenceRunner):
        runner = model_path
    else:
        runner = VMFBInferenceRunner(
            model_path,
            function=function,
            device_uri=device,
            n_threads=n_threads,
            load_method=load_method,
            load_model_to_mem=load_model_to_mem,
            runtime_flags=runtime_flags,
            device_outputs=device_io,
        )
    if not inputs:
        if runner.inputs_info is None:
            raise ValueError(
                "Input tensor info unavailable from model reflection data; "
                "please provide inputs explicitly"
            )
        inputs = random_inputs_from_info(runner.inputs_info)
    if device_io:
        inputs = list(inputs.values()) if isinstance(inputs, Mapping) else inputs
        inputs = [runner.allocate_device_array(inp) for inp in inputs]
    if do_warmup:
        runner.infer(inputs)

    sampler = ResourceSampler(os.getpid())
    sampler.start()
    total_time_ms: float = 0.0
    for _ in range(n_iters):
        runner.infer(inputs)
        total_time_ms += runner.infer_time_ms
    sampler.stop()

    return ProfileStats(
        avg_inference_time_ms=total_time_ms / n_iters,
        avg_dram_footprint_bytes=sampler.avg_rss,
        peak_dram_footprint_bytes=sampler.peak_rss,
        avg_anon_mem_bytes=sampler.avg_anon_rss,
        peak_anon_mem_bytes=sampler.peak_anon_rss,
        avg_cpu_percent=sampler.avg_cpu_percent,
        npu_inference_time_us=sampler.npu_inference_time_us,
        avg_npu_load_percent=sampler.avg_npu_load_percent,
    )
