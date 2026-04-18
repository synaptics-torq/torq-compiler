import datetime
import getpass
import hashlib
import logging
import os
import re
import socket
import subprocess
import tarfile
import tempfile
import time as _time
from pathlib import Path, PurePosixPath

from ..utils.remote_runner import RemoteCommandError, remote_command_runner_factory


def _parse_board_wall_time(output: str | None) -> float | None:
    """Parse wall-clock time from the shell ``time`` built-in output.

    Looks for a line like ``real  0m1.234s`` and returns the value in seconds.
    Returns *None* when the pattern is not found.
    """
    if not output:
        return None
    m = re.search(r'^real\s+(\d+)m([\d.]+)s', output, re.MULTILINE)
    if m:
        return int(m.group(1)) * 60.0 + float(m.group(2))
    return None


TOPDIR = Path(__file__).parent.parent.parent.parent
SOC_BUILD_DIR = Path(os.environ.get('IREE_SOC_BUILD_DIR', str(TOPDIR.parent / 'iree-build-soc')))

# Base directory on the remote board where per-user runner binaries are stored
REMOTE_RUNNER_BASE = "/home/root/iree-build-soc"

# Path to the cross-compiled torq-run-module binary in the SoC build tree.
LOCAL_RUNNER_PATH = SOC_BUILD_DIR / 'runtime' / 'tools' / 'torq-run-module'

# Default path to the NPU kernel module on the remote board.
DEFAULT_REMOTE_KO_PATH = "/usr/lib/modules/6.12.11/updates/syna_npu.ko"

# Name of the NPU kernel module (used with rmmod / insmod).
NPU_DRIVER_MODULE = "syna_npu"


def _md5_local(path: Path) -> str:
    """Return the hex MD5 digest of a local file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_remote_runner_path() -> str:
    """Build the default remote runner path using the local Linux username.

    e.g. for user 'aeldhose' -> /home/root/iree-build-soc/aeldhose/torq-run-module
    """
    username = getpass.getuser()
    return f"{REMOTE_RUNNER_BASE}/{username}/torq-run-module"


# Lock directory on the remote board for session-level exclusive access.
# mkdir is atomic on POSIX — it fails with EEXIST if another session holds
# the lock.  Lock metadata (user, host, pid, timestamp) is stored inside.
BOARD_LOCK_DIR = "/tmp/torq_board_test.lock"
BOARD_LOCK_INFO = f"{BOARD_LOCK_DIR}/info"


# Tracks (board_addr, port, remote_runner_path) tuples that have already been
# set up during this process so the setup is performed at most once per session.
_boards_setup: set[tuple[str, int, str]] = set()


def check_board_liveness(
    board_addr: str,
    port: int,
    timeout: int = 10,
) -> None:
    """Verify the board is reachable over SSH.

    Runs `true` on the remote host with a short timeout.  Raises
    ``RuntimeError`` immediately if the board cannot be reached so that the
    pytest session fails fast before any tests are collected or run.
    """
    cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={timeout}",
        "-o", "StrictHostKeyChecking=no",
        "-p", str(port),
        board_addr,
        "true",
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Board {board_addr}:{port} did not respond within {timeout}s.\n"
            "  The board may be powered off, rebooting, or have a new IP after a reflash.\n"
            "  To find it on the network:\n"
            "    python3 scripts/scan_boards.py\n"
            "  To search for a specific hostname:\n"
            "    python3 scripts/scan_boards.py --find <hostname>\n"
            "  To set a hostname on a board:\n"
            "    python3 scripts/set_board_hostname.py <ip> <hostname>"
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip() if e.stderr else ""
        raise RuntimeError(
            f"Board {board_addr}:{port} is unreachable (exit {e.returncode}).\n"
            f"  SSH error: {stderr}\n"
            "  The board may have been reflashed and received a new IP.\n"
            "  To find it on the network:\n"
            "    python3 scripts/scan_boards.py\n"
            "  To search for a specific hostname:\n"
            "    python3 scripts/scan_boards.py --find <hostname>\n"
            "  To set a hostname on a board:\n"
            "    python3 scripts/set_board_hostname.py <ip> <hostname>"
        )

    # Validate that the board's hostname matches the expected pattern
    # (sl2619-dev-board-NNN).  This catches cases where the IP was
    # reassigned to a different device after a reflash.
    import re as _re
    hostname_cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={timeout}",
        "-o", "StrictHostKeyChecking=no",
        "-p", str(port),
        board_addr,
        "hostname",
    ]
    try:
        result = subprocess.run(
            hostname_cmd,
            check=True,
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        remote_hostname = result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        raise RuntimeError(
            f"Board {board_addr}:{port} is reachable but failed to read its hostname.\n"
            "  Cannot verify board identity."
        )

    if not _re.match(r"^sl2619-dev-board-\d+$", remote_hostname):
        raise RuntimeError(
            f"Board {board_addr}:{port} is reachable but its hostname is '{remote_hostname}',\n"
            f"  which does not match the expected pattern 'sl2619-dev-board-NNN'.\n"
            "  This IP may have been reassigned to a different device.\n\n"
            "  To set the correct hostname on the board:\n"
            f"    python3 scripts/set_board_hostname.py {board_addr.split('@')[-1]} sl2619-dev-board-<number>\n"
            "  To find boards on the network:\n"
            "    python3 scripts/scan_boards.py"
        )


def acquire_board_lock(
    board_addr: str,
    port: int,
    timeout: int = 10,
    logger: logging.Logger | None = None,
) -> None:
    """Acquire a session-level exclusive lock on the remote board.

    Uses ``mkdir`` as an atomic lock primitive — it fails with EEXIST if
    another session already holds the lock.  Lock metadata (user, host,
    pid, timestamp) is written inside the directory so that anyone
    blocked by the lock can see who owns it.
    """
    log = logger or logging.getLogger("RemoteTestRunner")
    username = getpass.getuser()
    hostname = socket.gethostname()
    pid = os.getpid()
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    lock_info = f"user={username}\\nhost={hostname}\\npid={pid}\\ntime={timestamp}"

    runner = remote_command_runner_factory(
        board_addr, timeout, ssh_multiplex=True, ssh_port=port
    )
    with runner as r:
        try:
            r.run_cmd(["mkdir", BOARD_LOCK_DIR])
        except RemoteCommandError:
            # Lock directory already exists — read who holds it.
            try:
                info = r.run_cmd(["cat", BOARD_LOCK_INFO])
            except RemoteCommandError:
                info = ""

            # Extract the username from the lock info for a clear message.
            lock_owner = "unknown"
            for line in info.splitlines():
                if line.startswith("user="):
                    lock_owner = line.split("=", 1)[1]
                    break

            detail = "\n    ".join(info.strip().splitlines()) if info.strip() else "(lock directory exists but info file is missing)"
            raise RuntimeError(
                f"Board {board_addr} is locked by '{lock_owner}'.\n\n"
                f"  Lock info:\n    {detail}\n\n"
                "  If the lock is stale (e.g. previous session crashed), release it with:\n"
                f"    python3 scripts/reset_board_lock.py {board_addr}\n"
                "  Or with --port if non-default:\n"
                f"    python3 scripts/reset_board_lock.py {board_addr} --port {port}"
            )

        # mkdir succeeded — we own the lock.  Write metadata using chained
        # echo commands (busybox-safe, no printf escaping issues).
        # If the write fails, remove the lock directory so we don't leave
        # a stale lock with no info file.
        write_cmd = (
            f"echo 'user={username}' > {BOARD_LOCK_INFO} && "
            f"echo 'host={hostname}' >> {BOARD_LOCK_INFO} && "
            f"echo 'pid={pid}' >> {BOARD_LOCK_INFO} && "
            f"echo 'time={timestamp}' >> {BOARD_LOCK_INFO}"
        )
        try:
            r.run_cmd([write_cmd])
        except RemoteCommandError:
            # Clean up the orphaned lock directory.
            try:
                r.run_cmd(["rm", "-rf", BOARD_LOCK_DIR])
            except RemoteCommandError:
                pass
            raise RuntimeError(
                f"Failed to write lock info to {BOARD_LOCK_INFO} on {board_addr}. "
                "The lock directory has been cleaned up — retry the command."
            )
        log.info(
            "Acquired board lock on %s (user=%s, host=%s, pid=%d)",
            board_addr, username, hostname, pid,
        )


def release_board_lock(
    board_addr: str,
    port: int,
    timeout: int = 10,
    logger: logging.Logger | None = None,
) -> None:
    """Release the session-level lock on the remote board."""
    log = logger or logging.getLogger("RemoteTestRunner")
    runner = remote_command_runner_factory(
        board_addr, timeout, ssh_multiplex=True, ssh_port=port
    )
    with runner as r:
        try:
            r.run_cmd(["rm", "-rf", BOARD_LOCK_DIR])
        except RemoteCommandError:
            log.warning("Failed to release board lock on %s (board may be unreachable).", board_addr)
            return
    log.info("Released board lock on %s", board_addr)


def setup_dev_board(
    board_addr: str,
    port: int,
    remote_runner_path: str,
    local_ko_path: str | None = None,
    timeout: int = 5,
    logger: logging.Logger | None = None,
) -> None:
    """Prepare a remote dev board for testing.

    Checks board liveness, copies the locally built torq-run-module
    binary if it differs from what is on the board (MD5 comparison),
    and reloads the NPU kernel module.

    If *local_ko_path* is provided the .ko file is transferred to the
    board before reloading; otherwise the default on-board module at
    ``DEFAULT_REMOTE_KO_PATH`` is used.

    Only performs work on the first call per (board, port, path) tuple;
    subsequent calls are no-ops.
    """
    key = (board_addr, port, remote_runner_path)
    if key in _boards_setup:
        return

    log = logger or logging.getLogger("RemoteTestRunner")

    # Fail fast if the board is not reachable before doing any real work.
    log.info("Checking board liveness: %s:%s ...", board_addr, port)
    check_board_liveness(board_addr, port, timeout=10)
    log.info("Board %s:%s is reachable.", board_addr, port)

    local_bin = LOCAL_RUNNER_PATH if LOCAL_RUNNER_PATH.exists() else None

    remote_runner = remote_command_runner_factory(
        board_addr, timeout, ssh_multiplex=True, ssh_port=port
    )
    with remote_runner as runner:
        remote_runner_dir = str(PurePosixPath(remote_runner_path).parent)
        runner.run_cmd(["mkdir", "-p", remote_runner_dir])

        if local_bin is not None:
            local_md5 = _md5_local(local_bin)
            log.info("Local torq-run-module: %s (md5 %s)", local_bin, local_md5)

            # Check whether the remote binary already matches.
            try:
                remote_out = runner.run_cmd(["md5sum", remote_runner_path])
                remote_md5 = remote_out.split()[0] if remote_out else ""
                log.info("Remote torq-run-module: %s (md5 %s)", remote_runner_path, remote_md5)
            except RemoteCommandError:
                # File absent or md5sum unavailable — force copy.
                remote_md5 = ""
                log.info("Remote torq-run-module not found at %s, will copy.", remote_runner_path)

            if local_md5 == remote_md5:
                log.info(
                    "torq-run-module on %s is up to date (md5 %s), skipping copy.",
                    board_addr, local_md5,
                )
            else:
                log.info(
                    "Deploying local torq-run-module %s -> %s:%s (md5 %s)",
                    local_bin, board_addr, remote_runner_path, local_md5,
                )
                # Pack into a tarball and transfer, then extract on the board.
                with tempfile.TemporaryDirectory() as tmp:
                    tar_path = os.path.join(tmp, "torq-run-module.tar.gz")
                    remote_tar_path = str(
                        PurePosixPath(remote_runner_path).parent / "torq-run-module.tar.gz"
                    )
                    with tarfile.open(tar_path, "w:gz") as tf:
                        tf.add(str(local_bin), arcname=local_bin.name)
                    runner.copy_files(tar_path, remote_tar_path, board_dst=True)
                runner.run_cmd(
                    ["tar", "-xzf", remote_tar_path,
                     "-C", str(PurePosixPath(remote_runner_path).parent)]
                )
                runner.run_cmd(["rm", "-f", remote_tar_path])
                runner.run_cmd(["chmod", "+x", remote_runner_path])
        else:
            log.warning(
                "Local torq-run-module not found at %s. "
                "Assuming %s already exists on the board. "
                "Build it with: cmake --build <soc-build-dir> --target torq-run-module",
                LOCAL_RUNNER_PATH,
                remote_runner_path,
            )

        # --- NPU kernel module reload ---
        # When a local .ko is provided, compare its hash against the
        # default on-board module.  If they differ the on-board copy is
        # replaced and the board is rebooted so the new module is loaded
        # at boot; pytest then fails intentionally so the user re-runs
        # after the board comes back up.
        if local_ko_path:
            local_ko = Path(local_ko_path)
            if not local_ko.exists():
                raise FileNotFoundError(
                    f"Kernel module not found at {local_ko}. "
                    "Check the --torq-ko-path value."
                )

            local_ko_md5 = _md5_local(local_ko)
            log.info("Local ko: %s (md5 %s)", local_ko, local_ko_md5)

            # Hash of the default on-board ko.
            try:
                remote_ko_out = runner.run_cmd(["md5sum", DEFAULT_REMOTE_KO_PATH])
                remote_ko_md5 = remote_ko_out.split()[0] if remote_ko_out else ""
                log.info("On-board ko: %s (md5 %s)", DEFAULT_REMOTE_KO_PATH, remote_ko_md5)
            except RemoteCommandError:
                remote_ko_md5 = ""
                log.info("On-board ko not found at %s, will replace.", DEFAULT_REMOTE_KO_PATH)

            if local_ko_md5 != remote_ko_md5:
                log.info(
                    "Ko hash mismatch — replacing %s on the board and rebooting.",
                    DEFAULT_REMOTE_KO_PATH,
                )
                # Ensure the target directory exists and copy the new ko.
                remote_ko_dir = str(PurePosixPath(DEFAULT_REMOTE_KO_PATH).parent)
                runner.run_cmd(["mkdir", "-p", remote_ko_dir])
                runner.copy_files(str(local_ko), DEFAULT_REMOTE_KO_PATH, board_dst=True)

                # Reboot the board so the new module is loaded at boot.
                try:
                    runner.run_cmd(["reboot"])
                except RemoteCommandError:
                    pass  # reboot kills the SSH session — expected.

                raise RuntimeError(
                    f"Kernel module at {DEFAULT_REMOTE_KO_PATH} was out of date "
                    f"(local md5 {local_ko_md5}, board md5 {remote_ko_md5}).\n\n\n"
                    "  The new module has been copied and the board is rebooting.\n"
                    "  Wait ~30s for the board to come back up, then re-run pytest.\n"
                    "  If the board's IP changes after reboot, find it with:\n"
                    "    python3 scripts/scan_boards.py\n"
                    "  Or search by hostname:\n"
                    "    python3 scripts/scan_boards.py --find <hostname>\n"
                    "  To set a hostname so it survives reflashes:\n"
                    "    python3 scripts/set_board_hostname.py <ip> <hostname>"
                )

            log.info(
                "Ko hash matches on-board module — already loaded, continuing.",
            )
        else:
            log.info("No --torq-ko-path provided, skipping kernel module check.")

    _boards_setup.add(key)


class RemoteTestRunner:

    def __init__(
        self,
        vmfb_path: str | os.PathLike,
        function_name: str,
        input_data_args: list[str],
        output_data_args: list[str],
        *runtime_opts,
        board_addr: str | None = None,
        remote_dir_name: str | None = None,
        recompute_cache: bool = False,
        logger: logging.Logger | None = None,
        port : int = 22,
        remote_runner_path: str | None = None,
        update_runtime: bool = False,
    ):
        self.vmfb_path = Path(vmfb_path)
        self.function_name = function_name
        self.input_data_args = list(input_data_args)
        self.output_data_args = list(output_data_args)
        self.runtime_opts = list(runtime_opts)
        self.board_addr = board_addr
        self.port = port
        self._recompute_cache = recompute_cache
        self._logger = logger or logging.getLogger(__class__.__name__)
        self.remote_runner_path = remote_runner_path or _default_remote_runner_path()
        self._update_runtime = update_runtime

        if not self.board_addr:
            raise ValueError("RemoteTestRunner requires a board address.")

        if remote_dir_name:
            self.remote_root = PurePosixPath("/tmp") / remote_dir_name
        else:
            self.remote_root = PurePosixPath("/tmp") / self.vmfb_path.stem

    def _rewrite_arg_with_remote_path(self, arg: str) -> tuple[str, Path | None, PurePosixPath | None]:
        if "@" not in arg:
            return arg, None, None
        prefix, path_str = arg.split("@", 1)
        local_path = Path(path_str.strip())
        remote_path = self.remote_root / local_path.name
        return f"{prefix}@{remote_path}", local_path, remote_path

    def _rewrite_runtime_opts(
        self,
        runner: RemoteCommandError,
    ) -> tuple[list[str], dict[str, Path], dict[str, Path]]:
        path_options = {
            "--torq_desc_data_dir": "input_dir",
            "--torq_dump_mem_data_dir": "output_dir",
            "--torq_dump_io_data_dir": "output_dir",
            "--torq_dump_buffers_dir": "output_dir",
            "--torq_profile": "output_file",
            "--torq_profile_host": "output_file",
        }

        output_files: dict[str, Path] = {}
        output_dirs: dict[str, Path] = {}
        remote_opts: list[str] = []

        for opt in self.runtime_opts:
            rewritten = False
            for prefix, kind in path_options.items():
                if not opt.startswith(prefix + "="):
                    continue
                local_value = opt.split("=", 1)[1].strip()
                local_path = Path(local_value)
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

    def run(self, timeout: int = 5) -> float | None:
        """Run the model on the remote board.

        Returns the wall-clock time in seconds for the remote execution,
        or *None* if something prevents measurement.
        """
        remote_root = str(self.remote_root)
        remote_model_path = str(self.remote_root / self.vmfb_path.name)
        remote_runner = remote_command_runner_factory(
            self.board_addr,
            int(timeout),
            ssh_multiplex=True,
            ssh_port=self.port
        )

        with remote_runner as runner:
            if self._recompute_cache:
                runner.run_cmd(["rm", "-rf", remote_root])
            runner.run_cmd(["mkdir", "-p", remote_root])

            # Set up the board (copy runner binary, ko check, etc.) — no-op after first call.
            if self._update_runtime:
                setup_dev_board(
                    self.board_addr,
                    self.port,
                    self.remote_runner_path,
                    timeout=int(timeout),
                    logger=self._logger,
                )

            runner.copy_files(str(self.vmfb_path), remote_root, board_dst=True)
            remote_input_args: list[str] = []
            staged_inputs: set[Path] = set()
            for arg in self.input_data_args:
                remote_arg, local_path, _ = self._rewrite_arg_with_remote_path(arg)
                if local_path is not None and local_path not in staged_inputs:
                    runner.copy_files(str(local_path), remote_root, board_dst=True)
                    staged_inputs.add(local_path)
                remote_input_args.append(remote_arg)

            remote_runtime_opts, output_files, output_dirs = self._rewrite_runtime_opts(runner)
            remote_output_args: list[str] = []
            for arg in self.output_data_args:
                remote_arg, local_path, remote_path = self._rewrite_arg_with_remote_path(arg)
                if local_path is not None and remote_path is not None:
                    output_files[str(remote_path)] = local_path
                remote_output_args.append(remote_arg)

            if self._update_runtime:
                # Board lock is held at session level (acquired in
                # pytest_sessionstart, released in pytest_sessionfinish).
                # Invoke the runner by its absolute path.
                cmd = [
                    f"if [ -f /usr/local/bin/setup-test-environment ] ; then source /usr/local/bin/setup-test-environment ; fi ;",
                    f"{self.remote_runner_path}",
                    f"--module={remote_model_path}",
                    f"--function={self.function_name}",
                    *remote_runtime_opts,
                    *remote_output_args,
                    *remote_input_args,
                ]
            else:
                # FPGA / legacy path: no locking, discover runner via PATH.
                cmd = [
                    "IREE_RUN_MODULE=torq-run-module ;  if [ -f /usr/local/bin/setup-test-environment ] ; then source /usr/local/bin/setup-test-environment ; fi ; $IREE_RUN_MODULE",
                    f"--module={remote_model_path}",
                    f"--function={self.function_name}",
                    *remote_runtime_opts,
                    *remote_output_args,
                    *remote_input_args,
                ]
            self._logger.info("Running remote test: %s", " ".join(cmd))
            if self._update_runtime:
                # Wrap the command with the shell 'time' built-in so that
                # the wall-clock measurement happens on the board itself,
                # excluding any SSH transport overhead.  The entire command
                # must be inside a compound block so that 'time' covers the
                # full invocation, not just the first statement.
                cmd_str = " ".join(cmd)
                timed_cmd = [f"time {{ {cmd_str} ; }}"]
                output = runner.run_cmd(timed_cmd)
                wall_time = _parse_board_wall_time(output)
                if wall_time is not None:
                    self._logger.info("Board wall time: %.3fs", wall_time)
                else:
                    self._logger.warning(
                        "Could not parse board-side wall time from output, "
                        "'time' output not found in command output"
                    )
            else:
                runner.run_cmd(cmd)
                wall_time = None

            for remote_path, local_path in output_files.items():
                runner.copy_files(remote_path, str(local_path), board_dst=False)
            for remote_dir, local_dir in output_dirs.items():
                local_dir.parent.mkdir(parents=True, exist_ok=True)
                runner.copy_files(remote_dir, str(local_dir.parent), recursive=True, board_dst=False)

        return wall_time
