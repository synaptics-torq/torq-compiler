# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Remote (SSH/ADB) execution: staging, path rewriting, transport, and pull-back.

The transport runners (SSH/ADB) are an implementation detail of remote pipeline
execution; the staging, path-rewriting, and pullback mechanics wrap them. The
test-rig runner lives in ``torq.testing.remote_testing.RemoteTestRunner``.
"""

import logging
import os
import platform
import re
import shlex
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import List, Optional
from uuid import uuid4

from torq.lab import LabError

@dataclass
class RemoteTarget:
    """A remote board reachable over SSH or ADB.

    ``address`` accepts the same forms as the transport factory: ``adb`` (first
    device), an ADB serial, ``user@host``, or a bare hostname/IP for SSH.
    """

    address: str
    port: int = 22
    private_key: Optional[str] = None
    remote_runner_path: Optional[str] = None
    stage_runner: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return {
            "address": self.address,
            "port": self.port,
            "private_key": self.private_key,
            "remote_runner_path": self.remote_runner_path,
            "stage_runner": self.stage_runner,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RemoteTarget":
        return cls(
            address=d["address"],
            port=d.get("port", 22),
            private_key=d.get("private_key"),
            remote_runner_path=d.get("remote_runner_path"),
            stage_runner=d.get("stage_runner"),
        )

    @classmethod
    def from_manifest(cls, manifest: dict) -> Optional["RemoteTarget"]:
        """Reconstruct the remote target from a manifest's config section, if any."""
        section = manifest.get("config", manifest)
        remote = section.get("remote")
        return cls.from_dict(remote) if remote else None


def _is_known_adb_device(device_id: str) -> bool:
    """Return True if *device_id* appears in the current `adb devices` list."""
    try:
        result = subprocess.run(
            ["adb", "devices"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        for line in result.stdout.strip().splitlines()[1:]:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1] == "device" and parts[0] == device_id:
                return True
    except Exception:
        pass
    return False


def _is_adb_address(addr: str) -> bool:
    """Return True if *addr* is the auto-detect keyword or a known ADB device."""
    return addr.lower() == "adb" or _is_known_adb_device(addr)


class RemoteCommandError(Exception):

    def __init__(self, cmd: str, output: str):
        self.cmd = cmd
        self.output = output
        super().__init__(f"Error running remote command \"{cmd}\":\n\t{output}")


class RemoteCommandRunner(ABC):

    def __init__(self, board_addr: str, timeout: int = 5, logger: logging.Logger | None = None):
        self.board_addr = board_addr
        self.timeout = timeout
        self._logger = logger or logging.getLogger(__class__.__name__)

    @abstractmethod
    def run_cmd(self, cmd: str | list[str]) -> str | None: ...

    @abstractmethod
    def copy_files(self, src: str, dst: str, recursive: bool = False, board_dst: bool = False, verbose: bool = False) -> None: ...

    @abstractmethod
    def _cleanup(self) -> None: ...

    def _format_cmd(self, cmd: str | list[str] | tuple[str, ...]) -> str:
        if isinstance(cmd, (list, tuple)):
            return " ".join(cmd)
        return str(cmd)

    def _format_output(self, output: str | bytes | None, timeout: int | None = None) -> str:
        if output is None:
            return f"Command timed out after {timeout}s" if timeout is not None else ""
        if isinstance(output, bytes):
            return output.decode(errors="replace")
        return output

    def close(self) -> None:
        self._cleanup()

    def __enter__(self) -> "RemoteCommandRunner":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._cleanup()
        return False


class SSHCommandRunner(RemoteCommandRunner):

    def __init__(
        self,
        board_ip: str,
        timeout: int = 5,
        multiplex: bool = False,
        keep_alive: int = 10,
        logger: logging.Logger | None = None,
        port: int = 22,
        private_key: str | None = None
    ):
        super().__init__(board_ip, timeout, logger=logger)
        self.multiplex = bool(multiplex)
        self.keep_alive = int(keep_alive)
        self.ssh_options = [
            "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={self.timeout}",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
        ]

        if private_key:
            self.ssh_options += ["-i", private_key]

        self.ssh_socket = None
        self.port = port
        if self.multiplex:
            socket_name = f"ssh_mux_{board_ip.replace('.', '_')}_{os.getpid()}_{str(uuid4())}"
            # n MacOs use /tmp since gettempdir() path is too long
            tempdir = "/tmp" if platform.system() == "Darwin" else tempfile.gettempdir()
            self.ssh_socket = os.path.join(tempdir, socket_name)
            self._init_connection()

    def _init_connection(self) -> None:
        if not self.multiplex or not self.ssh_socket:
            return
        subprocess.Popen([
            "ssh", "-MNf",
            "-o", "ControlMaster=yes",
            "-o", f"ControlPath={self.ssh_socket}",
            "-o", f"ControlPersist={self.keep_alive}s",
            "-p", str(self.port),
            self.board_addr
        ] + self.ssh_options,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
        # Wait for the multiplex socket to be created so that subsequent
        # commands can reuse the connection rather than racing the master.
        for _ in range(100):
            if os.path.exists(self.ssh_socket):
                return
            time.sleep(0.05)

    def _cleanup(self) -> None:
        if not self.multiplex or not self.ssh_socket:
            return
        try:
            subprocess.run(
                [
                    "ssh", "-O", "exit",
                    "-o", f"ControlPath={self.ssh_socket}",
                    "-p", str(self.port)
                ] + self.ssh_options + [
                    self.board_addr,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            pass
        try:
            if os.path.exists(self.ssh_socket):
                os.remove(self.ssh_socket)
        except OSError:
            pass

    def check_multiplex(self):
        if self.multiplex:
            cmd = [
                "ssh", "-O", "check",
                "-o", f"ControlPath={self.ssh_socket}",
                "-p", str(self.port),
                self.board_addr
            ]
            try:
                out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=self.timeout)
                self._logger.warning("Multiplex connection is active: %s", out.strip())
            except subprocess.CalledProcessError as e:
                self._logger.warning("Multiplex connection is not active: %s", (e.output or "").strip())
            except Exception as e:
                self._logger.warning("Multiplex check failed: %s", e)
        else:
            self._logger.warning("Multiplex connection is not enabled.")

    def run_cmd(self, cmd: str | list[str]) -> str | None:
        if isinstance(cmd, str):
            cmd = shlex.split(cmd)
        try:
            if self.multiplex:
                full_cmd = [
                    "ssh", "-T",
                    "-o", "ControlMaster=no",
                    "-o", f"ControlPath={self.ssh_socket}",
                    "-p", str(self.port),
                ] + self.ssh_options + [self.board_addr] + cmd
            else:
                full_cmd = [
                    "ssh", "-T",
                    "-p", str(self.port),
                ] + self.ssh_options + [
                    self.board_addr,
                ] + cmd
            result = subprocess.check_output(
                full_cmd,
                text=True,
                stderr=subprocess.STDOUT,
                timeout=self.timeout,
            )
            if not self.multiplex:
                self._logger.info(
                    "Successfully executed command \"%s\" on %s",
                    " ".join(cmd), self.board_addr
                )
            return result
        except subprocess.TimeoutExpired as e:
            raise RemoteCommandError(
                self._format_cmd(e.cmd),
                self._format_output(e.output, self.timeout),
            ) from e
        except subprocess.CalledProcessError as e:
            self.check_multiplex()
            raise RemoteCommandError(
                self._format_cmd(e.cmd),
                self._format_output(e.stdout),
            ) from e

    def copy_files(self, src: str, dst: str, recursive: bool = False, board_dst: bool = False, verbose: bool = False) -> None:
        cmd = ["scp"]
        if recursive:
            cmd.append("-r")
        if board_dst:
            dst = f"{self.board_addr}:{dst}"
        else:
            src = f"{self.board_addr}:{src}"
        if self.multiplex:
            cmd.extend([
                "-o", "ControlMaster=no",
                "-o", f"ControlPath={self.ssh_socket}",
                "-P", str(self.port),
                src,
                dst
            ])
        else:
            cmd.extend(self.ssh_options + [
                "-P", str(self.port),
                src,
                dst
            ])
        try:
            if verbose:
                subprocess.check_call(cmd, timeout=self.timeout)
            else:
                subprocess.check_output(
                    cmd,
                    text=True,
                    stderr=subprocess.STDOUT
                ) # we cannot set a timeout because it may take very long to upload files
            if not self.multiplex:
                self._logger.info("Copied \"%s\" to \"%s\"", src, dst)
        except subprocess.CalledProcessError as e:
            self.check_multiplex()
            raise RemoteCommandError(
                self._format_cmd(e.cmd),
                self._format_output(e.stdout),
            ) from e


class ADBCommandRunner(RemoteCommandRunner):
    """
    Run commands on a remote device using ADB instead of SSH.
    Notes:
      - Requires `adb` in PATH.
    """

    def __init__(
        self,
        target_device: str = "",
        timeout: int = 15,
        logger: logging.Logger | None = None
    ):
        super().__init__("", timeout, logger=logger)


        # The target passed to `adb -s`.
        self._target = (target_device or "").strip() or None

        # Initialize connection
        self._ensure_server()
        self._init_connection()

    def _adb_cmd_prefix(self) -> list[str]:
        prefix = ["adb"]
        host = os.environ.get("ADB_SERVER_HOST")
        port = os.environ.get("ADB_SERVER_PORT")
        if host:
            prefix += ["-H", host]
        if port:
            prefix += ["-P", port]
        if self._target:
            prefix += ["-s", self._target]
        return prefix

    def _ensure_server(self) -> None:
        """Start the ADB server if not running."""
        try:
            subprocess.run(
                ["adb", "start-server"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise RemoteCommandError(
                self._format_cmd(e.cmd),
                self._format_output("Timed out starting ADB server", self.timeout),
            ) from e

    def _init_connection(self) -> None:
        try:
            # Wait for the specific device to be ready.
            subprocess.run(
                [*self._adb_cmd_prefix(), "wait-for-device"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise RemoteCommandError(
                self._format_cmd(e.cmd),
                self._format_output("Timed out waiting for ADB device", self.timeout),
            ) from e

    def _cleanup(self) -> None:
        pass

    def run_cmd(self, cmd: str | list[str]) -> str | None:
        """
        Execute a shell command on the device via `adb shell`.
        Returns stdout as a string
        """
        # Normalize input
        if isinstance(cmd, str):
            cmd = shlex.split(cmd)

        # For adb shell we pass a single string for the remote shell to interpret.
        # If strict quoting is required, pass a single string in the original call.
        cmd = " ".join(cmd)
        full_cmd = [*self._adb_cmd_prefix(), "shell", cmd]
        print("[ADB]", cmd)

        try:
            result = subprocess.check_output(
                full_cmd,
                text=True,
                stderr=subprocess.STDOUT,
                timeout=self.timeout,
            )
            self._logger.info(
                'Successfully executed command "%s" on %s',
                cmd, self.board_addr
            )
            return result
        except subprocess.TimeoutExpired as e:
            print("Command timed out:")
            print(e.output)
            raise RemoteCommandError(
                self._format_cmd(e.cmd),
                self._format_output(e.output, self.timeout),
            ) from e
        except subprocess.CalledProcessError as e:
            print("Command failed:")
            print(e.output)
            raise RemoteCommandError(
                self._format_cmd(e.cmd),
                self._format_output(e.stdout),
            ) from e

    def copy_files(self, src: str, dst: str, recursive: bool = False, board_dst: bool = False, verbose: bool = False) -> None:
        """
        Copy files using ADB:
          - board_dst=True  -> push  (local src -> device dst)
          - board_dst=False -> pull  (device src -> local dst)
        """
        if board_dst:
            # local -> device
            cmd = [*self._adb_cmd_prefix(), "push", src, dst]
        else:
            # device -> local
            cmd = [*self._adb_cmd_prefix(), "pull", src, dst]

        try:
            subprocess.check_output(
                cmd,
                text=True,
                stderr=subprocess.STDOUT,
                timeout=self.timeout,
            )
            direction = "to device" if board_dst else "from device"
            self._logger.info('Copied "%s" %s "%s"', src, direction, dst)
        except subprocess.TimeoutExpired as e:
            raise RemoteCommandError(
                self._format_cmd(e.cmd),
                self._format_output(e.output, self.timeout),
            ) from e
        except subprocess.CalledProcessError as e:
            raise RemoteCommandError(
                self._format_cmd(e.cmd),
                self._format_output(e.stdout),
            ) from e


def _get_first_adb_device() -> str | None:
    """Return the first connected ADB device ID, or None if none found."""
    try:
        result = subprocess.run(
            ["adb", "devices"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        for line in result.stdout.strip().splitlines()[1:]:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1] == "device":
                return parts[0]
    except Exception:
        pass
    return None


def remote_command_runner_factory(
    board_address: str,
    timeout: int = 15,
    logger: logging.Logger | None = None,
    *,
    ssh_multiplex: bool = True,
    ssh_keep_alive: int = 10,
    ssh_port: int = 22,
    ssh_private_key: str | None = None
) -> RemoteCommandRunner:
    # "adb" means auto-detect the first connected ADB device.
    if board_address.lower() == "adb":
        first_device = _get_first_adb_device()
        if not first_device:
            raise ValueError("No ADB device detected. Connect a device and run 'adb devices'.")
        return ADBCommandRunner(target_device=first_device, timeout=timeout, logger=logger)

    # ADB device IDs (e.g. sl2619, sl2619-dev-board-NNN, SL16x0).
    if _is_adb_address(board_address):
        return ADBCommandRunner(
            target_device=board_address,
            timeout=timeout,
            logger=logger
        )

    # Everything else (user@ip or bare hostnames like sl2619-dev-board-80) is SSH.
    return SSHCommandRunner(
        board_ip=board_address,
        timeout=timeout, logger=logger,
        multiplex=ssh_multiplex,
        keep_alive=ssh_keep_alive,
        port=ssh_port,
        private_key=ssh_private_key
    )


# The named stages a remote run can fail at, in the order they occur.
_STAGES = ("connect", "stage-runner", "stage-model", "execute", "pull-results")


class RemoteStageError(LabError):
    """A named remote-execution stage failed, wrapping the underlying transport error."""

    def __init__(self, stage: str, detail: str):
        self.stage = stage
        self.detail = detail
        super().__init__(f"remote run failed at stage '{stage}': {detail}")

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
        keep_remote_dir: bool = True,
    ):
        self.target = target
        self.vmfb_path = Path(vmfb_path)
        self.function_name = function_name
        self.input_args = list(input_args)
        self.output_args = list(output_args)
        self.runtime_opts = list(runtime_opts)
        self.timeout = timeout
        self._logger = logger or logging.getLogger("torq.lab.remote")
        self.keep_remote_dir = keep_remote_dir
        name = remote_dir_name or self.vmfb_path.stem
        self.remote_root = PurePosixPath("/tmp") / name

    @contextmanager
    def _stage(self, name: str):
        """Wrap a block of remote calls, reporting a transport failure as this named stage."""
        assert name in _STAGES, f"unknown remote stage {name!r}, expected one of {_STAGES}"
        try:
            yield
        except RemoteStageError:
            raise
        except Exception as exc:
            raise RemoteStageError(name, str(exc)) from exc

    def _preflight_text(self, runner) -> str:
        runner_source = self.target.stage_runner or self.target.remote_runner_path or "PATH"
        return (
            f"preflight: target={self.target.address}:{self.target.port} "
            f"transport={type(runner).__name__} runner={runner_source} "
            f"vmfb={self.vmfb_path} inputs={len(self.input_args)} "
            f"remote_dir={self.remote_root}"
        )

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
        """Run the staged VMFB on the board, reporting a failure by named stage.

        A transport failure at any point is re-raised as a :class:`RemoteStageError`
        naming which of ``connect``/``stage-runner``/``stage-model``/``execute``/
        ``pull-results`` it happened at, instead of surfacing the raw
        scp/ssh/adb command as the primary diagnostic (that command is still in
        ``.detail`` for follow-up debugging). A compact preflight is logged
        before anything is transferred.
        """
        remote_root = str(self.remote_root)
        remote_model = str(self.remote_root / self.vmfb_path.name)

        with self._stage("connect"):
            runner = remote_command_runner_factory(
                self.target.address,
                int(self.timeout or 15),
                ssh_multiplex=True,
                ssh_port=self.target.port,
                ssh_private_key=self.target.private_key,
            )

        self._logger.info(self._preflight_text(runner))

        pulled: List[Path] = []
        with runner as r:
            with self._stage("connect"):
                r.run_cmd(["mkdir", "-p", remote_root])

            with self._stage("stage-runner"):
                invoke, _ = self._build_invoke(r)

            with self._stage("stage-model"):
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

            with self._stage("execute"):
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

            with self._stage("pull-results"):
                for remote_path, local_path in output_files.items():
                    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
                    r.copy_files(remote_path, str(local_path), board_dst=False)
                    pulled.append(Path(local_path))
                for remote_dir, local_dir in output_dirs.items():
                    Path(local_dir).parent.mkdir(parents=True, exist_ok=True)
                    r.copy_files(remote_dir, str(Path(local_dir).parent), recursive=True, board_dst=False)
                    pulled.append(Path(local_dir))

            if not self.keep_remote_dir:
                try:
                    r.run_cmd(["rm", "-rf", remote_root])
                except Exception as exc:
                    self._logger.warning("Could not remove remote scratch dir %s: %s", remote_root, exc)

        return RemoteRunOutcome(command=cmd, wall_time=wall_time, output=output or "", pulled_files=pulled)
