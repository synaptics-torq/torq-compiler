# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Remote (SSH/ADB) execution: stage artifacts, rewrite paths, run, pull back.

Builds on the transport in ``torq.lab.transport``.  The staging,
path-rewriting, and pullback mechanics are self-contained here (no board
locking, kernel-module update, once-per-session globals, or print side
effects); the test-rig runner lives in
``torq.testing.remote_testing.RemoteTestRunner``.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import List, Optional

from torq.lab.transport import remote_command_runner_factory

from torq.lab.types import RemoteTarget

logger = logging.getLogger("torq.lab.remote")

# Runtime options whose value is a local path that must be staged or pulled back.
_PATH_OPTIONS = {
    "--torq_desc_data_dir": "input_dir",
    "--torq_dump_mem_data_dir": "output_dir",
    "--torq_dump_io_data_dir": "output_dir",
    "--torq_dump_buffers_dir": "output_dir",
    "--torq_profile": "output_file",
    "--torq_profile_host": "output_file",
}


def parse_board_wall_time(output: Optional[str]) -> Optional[float]:
    """Parse seconds from the shell ``time`` builtin output (``real  0m1.234s``)."""
    if not output:
        return None
    m = re.search(r"^real\s+(\d+)m([\d.]+)s", output, re.MULTILINE)
    if m:
        return int(m.group(1)) * 60.0 + float(m.group(2))
    return None


@dataclass
class RemoteRunOutcome:
    command: List[str]
    wall_time: Optional[float] = None
    output: str = ""
    pulled_files: List[Path] = field(default_factory=list)


class RemoteExecutor:
    """Run a compiled module on a remote board over SSH or ADB."""

    def __init__(
        self,
        target: RemoteTarget,
        vmfb_path,
        function_name: str,
        input_args: List[str],
        output_args: List[str],
        runtime_opts: List[str],
        *,
        remote_dir_name: Optional[str] = None,
        timeout: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.target = target
        self.vmfb_path = Path(vmfb_path)
        self.function_name = function_name
        self.input_args = list(input_args)
        self.output_args = list(output_args)
        self.runtime_opts = list(runtime_opts)
        self.timeout = timeout
        self._logger = logger or logging.getLogger("torq.lab.remote")
        name = remote_dir_name or self.vmfb_path.stem
        self.remote_root = PurePosixPath("/tmp") / name

    def _rewrite_arg_with_remote_path(self, arg: str):
        if "@" not in arg:
            return arg, None, None
        prefix, path_str = arg.split("@", 1)
        local_path = Path(path_str.strip())
        remote_path = self.remote_root / local_path.name
        return f"{prefix}@{remote_path}", local_path, remote_path

    def _rewrite_runtime_opts(self, runner):
        output_files = {}
        output_dirs = {}
        remote_opts = []
        for opt in self.runtime_opts:
            rewritten = False
            for prefix, kind in _PATH_OPTIONS.items():
                if not opt.startswith(prefix + "="):
                    continue
                local_path = Path(opt.split("=", 1)[1].strip())
                remote_path = self.remote_root / local_path.name
                if kind == "input_dir":
                    runner.copy_files(str(local_path), str(self.remote_root), recursive=True, board_dst=True)
                elif kind == "output_dir":
                    runner.run_cmd(["mkdir", "-p", str(remote_path)])
                    output_dirs[str(remote_path)] = local_path
                elif kind == "output_file":
                    output_files[str(remote_path)] = local_path
                remote_opts.append(f"{prefix}={remote_path}")
                rewritten = True
                break
            if not rewritten:
                remote_opts.append(opt)
        return remote_opts, output_files, output_dirs

    def _build_invoke(self, runner) -> tuple:
        """Return (invocation-prefix, runner_path) after optionally staging the runner."""
        runner_path = self.target.remote_runner_path
        if self.target.stage_runner:
            runner_path = runner_path or str(self.remote_root / "torq-run-module")
            runner.run_cmd(["mkdir", "-p", str(PurePosixPath(runner_path).parent)])
            runner.copy_files(str(self.target.stage_runner), runner_path, board_dst=True)
            runner.run_cmd(["chmod", "+x", runner_path])
        if runner_path:
            return [runner_path], runner_path
        # Discover the runner via PATH after sourcing the board test environment.
        return [
            "IREE_RUN_MODULE=torq-run-module ; "
            "if [ -f /usr/local/bin/setup-test-environment ] ; "
            "then source /usr/local/bin/setup-test-environment ; fi ; $IREE_RUN_MODULE"
        ], None

    def run(self) -> RemoteRunOutcome:
        remote_root = str(self.remote_root)
        remote_model = str(self.remote_root / self.vmfb_path.name)
        runner = remote_command_runner_factory(
            self.target.address,
            int(self.timeout or 15),
            ssh_multiplex=True,
            ssh_port=self.target.port,
            ssh_private_key=self.target.private_key,
        )

        pulled: List[Path] = []
        with runner as r:
            r.run_cmd(["mkdir", "-p", remote_root])
            invoke, _ = self._build_invoke(r)

            self._logger.info("Staging model to board: %s -> %s", self.vmfb_path, remote_root)
            r.copy_files(str(self.vmfb_path), remote_root, board_dst=True)

            remote_input_args: List[str] = []
            staged = set()
            for arg in self.input_args:
                remote_arg, local_path, _ = self._rewrite_arg_with_remote_path(arg)
                if local_path is not None and local_path not in staged:
                    r.copy_files(str(local_path), remote_root, board_dst=True)
                    staged.add(local_path)
                remote_input_args.append(remote_arg)

            remote_opts, output_files, output_dirs = self._rewrite_runtime_opts(r)
            remote_output_args: List[str] = []
            for arg in self.output_args:
                remote_arg, local_path, remote_path = self._rewrite_arg_with_remote_path(arg)
                if local_path is not None and remote_path is not None:
                    output_files[str(remote_path)] = local_path
                remote_output_args.append(remote_arg)

            cmd = [
                *invoke,
                f"--module={remote_model}",
                f"--function={self.function_name}",
                *remote_opts,
                *remote_output_args,
                *remote_input_args,
            ]
            # Measure wall time on the board itself, excluding SSH transport.
            timed_cmd = [f"time {{ {' '.join(cmd)} ; }}"]
            self._logger.info("Running remote: %s", " ".join(cmd))
            output = r.run_cmd(timed_cmd)
            wall_time = parse_board_wall_time(output)

            for remote_path, local_path in output_files.items():
                Path(local_path).parent.mkdir(parents=True, exist_ok=True)
                r.copy_files(remote_path, str(local_path), board_dst=False)
                pulled.append(Path(local_path))
            for remote_dir, local_dir in output_dirs.items():
                Path(local_dir).parent.mkdir(parents=True, exist_ok=True)
                r.copy_files(remote_dir, str(Path(local_dir).parent), recursive=True, board_dst=False)
                pulled.append(Path(local_dir))

        return RemoteRunOutcome(command=cmd, wall_time=wall_time, output=output or "", pulled_files=pulled)
