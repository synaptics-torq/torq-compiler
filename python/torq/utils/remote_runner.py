import logging
import os
import shlex
import subprocess
import tempfile
import platform
import sys
from abc import ABC, abstractmethod
from uuid import uuid4


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
    def copy_files(self, src: str, dst: str, recursive: bool = False, board_dst: bool = False) -> None: ...

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
        port: int = 22
    ):
        super().__init__(board_ip, timeout, logger=logger)
        self.multiplex = bool(multiplex)
        self.keep_alive = int(keep_alive)
        self.ssh_options = [
            "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={self.timeout}",
        ]
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
                    "-o", "BatchMode=yes",
                    "-o", f"ConnectTimeout={self.timeout}",
                    "-p", str(self.port),
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
            raise RemoteCommandError(
                self._format_cmd(e.cmd),
                self._format_output(e.stdout),
            ) from e
        
    def copy_files(self, src: str, dst: str, recursive: bool = False, board_dst: bool = False) -> None:
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
            cmd.extend([
                "-o", "BatchMode=yes",
                "-o", f"ConnectTimeout={self.timeout}",
                "-P", str(self.port),
                src,
                dst
            ])
        try:
            subprocess.check_output(
                cmd,
                text=True,
                stderr=subprocess.STDOUT,
                timeout=self.timeout,
            )
            if not self.multiplex:
                self._logger.info("Copied \"%s\" to \"%s\"", src, dst)
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


if __name__ == "__main__":
    pass
