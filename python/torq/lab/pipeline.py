# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ModelPipeline: compile/run/profile orchestration over local and remote targets."""

import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from torq.lab import io, tools
from torq.lab.types import (
    CompileResult,
    LabError,
    MlirIoSpec,
    PipelineConfig,
    RemoteTarget,
    RunResult,
)


def build_compile_command(config, tool, vmfb_path, debug_dir, phases_dir, compile_profile) -> List[str]:
    """Assemble the ``torq-compile`` command line from a PipelineConfig."""
    cmds = [str(tool), str(config.model_path), "-o", str(vmfb_path)]
    cmds.append(f"--torq-hw={config.chip}")

    if config.dump_ir:
        ir_dir = Path(debug_dir) / "ir"
        cmds.extend([
            "--mlir-print-ir-after-all",
            f"--mlir-print-ir-tree-dir={ir_dir}",
            "--mlir-elide-elementsattrs-if-larger=50",
        ])

    if config.dump_phases:
        cmds.append(f"--dump-compilation-phases-to={phases_dir}")

    # The cmodel/fpga host needs native host binaries
    if config.runtime_hw_type in ["sim", "aws_fpga"]:
        cmds.extend(["--torq-target-host-triple=native"])    

    cmds += list(config.compiler_options)

    # Debug info is always emitted; profiling annotates cycle-time attributes.
    cmds.append(f"--torq-debug-info={debug_dir}")
    if config.profile_compile:
        cmds.extend(["--torq-enable-profiling", f"--torq-dump-profiling={compile_profile}"])
    elif config.profile_runtime:
        cmds.append("--torq-enable-profiling")

    return cmds


def build_run_command(config, tool, vmfb_path, func_name, output_args, input_args, host_profile) -> List[str]:
    """Assemble the ``torq-run-module`` command line from a PipelineConfig."""
    cmds = [
        str(tool),
        f"--module={vmfb_path}",
        f"--function={func_name}",
        *output_args,
        f"--torq_hw_type={config.runtime_hw_type}",
        *list(config.runtime_options),
        *input_args,
    ]
    if config.profile_runtime:
        cmds.append(f"--torq_profile_host={host_profile}")
    return cmds


def _diagnostics(proc) -> str:
    try:
        return proc.stderr.decode("utf-8", errors="replace")
    except Exception:
        return ""


class ModelPipeline:
    """High-level compile/run/profile orchestration around a PipelineConfig."""

    def __init__(self, config: PipelineConfig, remote: Optional[RemoteTarget] = None):
        self.config = config
        self.remote = remote
        self.work_dir = Path(config.work_dir)
        self.inputs_dir = self.work_dir / "inputs"
        self.outputs_dir = self.work_dir / "outputs"
        self.debug_dir = self.work_dir / "debug"
        self.phases_dir = self.work_dir / "phases"
        self.profiles_dir = self.work_dir / "profiles"
        self.remote_dir = self.work_dir / "remote"

    # -- helpers ---------------------------------------------------------

    def _vmfb_path(self) -> Path:
        if self.config.vmfb_path:
            return Path(self.config.vmfb_path)
        return self.work_dir / "model.vmfb"

    def _spec_source(self) -> Optional[Path]:
        if self.config.spec_source:
            return Path(self.config.spec_source)
        model_path = Path(self.config.model_path)
        if model_path.suffix == ".mlir":
            return model_path
        sibling = model_path.with_suffix(".mlir")
        return sibling if sibling.exists() else None

    def _io_spec(self) -> Optional[MlirIoSpec]:
        src = self._spec_source()
        if src and src.exists():
            return io.parse_mlir_io_spec(src)
        return None

    def _materialize_inputs(self, spec) -> Tuple[List[np.ndarray], List[Path]]:
        if self.config.input_npy:
            inputs = [np.load(p, allow_pickle=False) for p in self.config.input_npy]
        elif self.config.random_inputs:
            if spec is None:
                raise LabError(
                    "--random-inputs requires an MLIR spec source to know input shapes/dtypes"
                )
            inputs = io.generate_random_inputs(spec)
        else:
            return [], []
        paths = io.write_inputs(inputs, self.inputs_dir)
        return inputs, paths

    def _collect_profiles(self):
        host = self.profiles_dir / "host_profile.csv"
        host = host if host.exists() else None
        annotated = trace = None
        if self.profiles_dir.exists():
            xlsx = sorted(self.profiles_dir.glob("annotated_profile*.xlsx"))
            annotated = xlsx[0] if xlsx else None
            pbs = sorted(self.profiles_dir.glob("trace*.pb"))
            trace = pbs[0] if pbs else None
        return host, annotated, trace

    def _collect_compile_profiles(self):
        """Locate the compile-time Perfetto trace and viewer, if produced."""
        trace = None
        if self.profiles_dir.exists():
            pbs = sorted(self.profiles_dir.glob("*_compile.pb"))
            trace = pbs[0] if pbs else None
        return trace, self._viewer_path()

    def _viewer_path(self) -> Optional[Path]:
        viewer = self.profiles_dir / "perfetto_viewer.html"
        return viewer if viewer.exists() else None

    def _has_debug_info(self) -> bool:
        """True when the compile emitted debug info to annotate against."""
        return self.debug_dir.is_dir() and any(self.debug_dir.iterdir())

    def _finalize_profiles(self) -> None:
        """Annotate the host profile, add compile trace(s), and render the viewer.

        No-op unless a profiling mode is enabled and the compile produced debug
        info to annotate against. Requires the ``[profile]`` extra; the
        ``torq.lab.profiling`` helpers raise ``LabError`` when it is missing.

        Safe to call after both ``compile`` (compile trace) and ``run`` (host
        annotation): the compile trace is written only once, so a subsequent run
        adds the annotated run trace and re-renders a combined viewer.
        """
        if not (self.config.profile_runtime or self.config.profile_compile):
            return
        if not self._has_debug_info():
            return
        from torq.lab import profiling

        host_profile = self.profiles_dir / "host_profile.csv"
        if self.config.profile_runtime and host_profile.exists():
            profiling.annotate_run_profile(self.debug_dir, host_profile, self.profiles_dir)
        if self.config.profile_compile and not any(self.profiles_dir.glob("*_compile.pb")):
            profiling.write_compile_trace(self.debug_dir, self.profiles_dir)

        pb_files = sorted(self.profiles_dir.glob("*.pb"))
        if pb_files:
            profiling.write_perfetto_report(pb_files, self.profiles_dir / "perfetto_viewer.html")

    # -- stages ----------------------------------------------------------

    def compile(self) -> CompileResult:
        """Compile ``config.model_path`` (an ``.mlir``) to a VMFB.

        Creates the work-dir layout, invokes ``torq-compile`` with flags derived
        from the config (chip, IR/phase dumps, profiling), and returns a
        :class:`~torq.lab.types.CompileResult`. Raises
        :class:`~torq.lab.types.LabError` / ``ToolError`` on tool failure.
        """
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        if self.config.dump_ir:
            (self.debug_dir / "ir").mkdir(parents=True, exist_ok=True)
        if self.config.dump_phases:
            self.phases_dir.mkdir(parents=True, exist_ok=True)
        compile_profile = None
        if self.config.profile_compile:
            self.profiles_dir.mkdir(parents=True, exist_ok=True)
            compile_profile = self.profiles_dir / "compile_profile.csv"

        tool = tools.find_compile_tool(self.config.compile_tool)
        vmfb = self._vmfb_path()
        cmds = build_compile_command(
            self.config, tool, vmfb, self.debug_dir, self.phases_dir, compile_profile
        )
        start = time.perf_counter()
        proc = tools.run_tool(cmds, timeout=self.config.timeout, cwd=self.work_dir)
        elapsed = time.perf_counter() - start

        # Compile-time profiling produces its trace and viewer here, without
        # needing a run (unlike runtime profiling, which is finalized in run()).
        compile_trace = perfetto_viewer = None
        if self.config.profile_compile:
            self._finalize_profiles()
            compile_trace, perfetto_viewer = self._collect_compile_profiles()

        return CompileResult(
            vmfb_path=vmfb,
            command=cmds,
            elapsed=elapsed,
            debug_dir=self.debug_dir,
            phases_dir=self.phases_dir if self.config.dump_phases else None,
            compile_profile=compile_profile,
            compile_trace=compile_trace,
            perfetto_viewer=perfetto_viewer,
            diagnostics=_diagnostics(proc),
        )

    def run(self, vmfb_path=None) -> RunResult:
        """Run a compiled VMFB and collect its outputs.

        Uses ``vmfb_path`` when given, else ``config.vmfb_path`` or
        ``<work-dir>/model.vmfb``. Materializes inputs (from ``input_npy`` or
        ``random_inputs``), resolves the entry function and output specs from the
        MLIR IO spec, executes locally or on ``self.remote``, loads outputs back
        into numpy arrays, and finalizes any profiling artifacts. Returns a
        :class:`~torq.lab.types.RunResult`.
        """
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        vmfb = Path(vmfb_path) if vmfb_path else self._vmfb_path()

        spec = self._io_spec()
        spec_src = self._spec_source()
        func_name = self.config.function or (
            io.parse_func_name(spec_src) if spec_src and spec_src.exists() else "main"
        )

        inputs, input_paths = self._materialize_inputs(spec)
        input_args = io.build_input_args(input_paths, inputs, spec) if inputs else []

        if spec is not None:
            output_args = io.create_output_args(self.outputs_dir, spec.outputs)
            output_paths = [Path(p) for p in io.create_output_paths(self.outputs_dir, spec.outputs)]
        else:
            output_args, output_paths = [], []

        host_profile = None
        if self.config.profile_runtime:
            self.profiles_dir.mkdir(parents=True, exist_ok=True)
            host_profile = self.profiles_dir / "host_profile.csv"

        if self.remote is not None:
            return self._run_remote(vmfb, func_name, input_args, input_paths, output_args, output_paths, spec)

        tool = tools.find_run_tool(self.config.run_tool)
        cmds = build_run_command(
            self.config, tool, vmfb, func_name, output_args, input_args, host_profile
        )
        start = time.perf_counter()
        proc = tools.run_tool(cmds, timeout=self.config.timeout, cwd=self.work_dir)
        wall = time.perf_counter() - start

        outputs = io.load_outputs(spec.outputs, output_paths) if (spec and output_paths) else []
        self._finalize_profiles()
        host, annotated, trace = self._collect_profiles()
        return RunResult(
            command=cmds,
            output_paths=output_paths,
            outputs=outputs,
            host_profile=host,
            annotated_profile=annotated,
            trace=trace,
            perfetto_viewer=self._viewer_path(),
            wall_time=wall,
            diagnostics=_diagnostics(proc),
        )

    def _run_remote(self, vmfb, func_name, input_args, input_paths, output_args, output_paths, spec) -> RunResult:
        from torq.lab.remote import RemoteExecutor

        self.remote_dir.mkdir(parents=True, exist_ok=True)
        runtime_opts = [f"--torq_hw_type={self.config.runtime_hw_type}", *list(self.config.runtime_options)]
        if self.config.profile_runtime:
            self.profiles_dir.mkdir(parents=True, exist_ok=True)
            runtime_opts.append(f"--torq_profile_host={self.profiles_dir / 'host_profile.csv'}")

        executor = RemoteExecutor(
            self.remote, vmfb, func_name, input_args, output_args, runtime_opts,
            timeout=self.config.timeout,
        )
        outcome = executor.run()

        outputs = io.load_outputs(spec.outputs, output_paths) if (spec and output_paths) else []
        self._finalize_profiles()
        host, annotated, trace = self._collect_profiles()
        return RunResult(
            command=outcome.command,
            output_paths=output_paths,
            outputs=outputs,
            host_profile=host,
            annotated_profile=annotated,
            trace=trace,
            perfetto_viewer=self._viewer_path(),
            wall_time=outcome.wall_time,
            diagnostics=outcome.output,
        )

    def compile_run(self) -> Tuple[CompileResult, RunResult]:
        """Compile the model, then run the freshly built VMFB in one call."""
        compile_result = self.compile()
        run_result = self.run(compile_result.vmfb_path)
        return compile_result, run_result

    # -- reporting -------------------------------------------------------

    def expected_outputs(self) -> Optional[List[np.ndarray]]:
        """Load the ``expected_output_npy`` arrays, or ``None`` if none configured."""
        if not self.config.expected_output_npy:
            return None
        return [np.load(p, allow_pickle=False) for p in self.config.expected_output_npy]

    def compare(self, run_result: RunResult):
        """Compare run outputs against the configured expected outputs, if any.

        Returns ``None`` only when no expected outputs are configured. When they
        are, the comparison always runs, so a run that produced no outputs (or a
        mismatched output count) is reported as a failure instead of silently
        passing.
        """
        expected = self.expected_outputs()
        if expected is None:
            return None
        from torq.lab.compare import compare_outputs

        return compare_outputs(run_result.outputs, expected)

    def write_manifest(self, **kwargs) -> Path:
        """Build and write ``manifest.json`` into the work dir."""
        from torq.lab.manifest import build_manifest
        from torq.lab.manifest import write_manifest as _write

        manifest = build_manifest(config=self.config, remote=self.remote, **kwargs)
        return _write(self.work_dir, manifest)
