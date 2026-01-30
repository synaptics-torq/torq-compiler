import logging
import os
from pathlib import Path, PurePosixPath

from ..utils.remote_runner import SSHCommandRunner


class SoCTestRunner:

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
    ):
        self.vmfb_path = Path(vmfb_path)
        self.function_name = function_name
        self.input_data_args = list(input_data_args)
        self.output_data_args = list(output_data_args)
        self.runtime_opts = list(runtime_opts)
        self.board_addr = board_addr
        self._recompute_cache = recompute_cache
        self._logger = logger or logging.getLogger(__class__.__name__)

        if not self.board_addr:
            raise ValueError("SoCTestRunner requires a board address.")

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
        runner: SSHCommandRunner,
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

    def run(self, timeout: int = 5) -> None:
        remote_root = str(self.remote_root)
        remote_model_path = str(self.remote_root / self.vmfb_path.name)

        with SSHCommandRunner(self.board_addr, timeout=int(timeout), multiplex=True) as runner:
            if self._recompute_cache:
                runner.run_cmd(["rm", "-r", remote_root])
            runner.run_cmd(["mkdir", "-p", remote_root])

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

            cmd = [
                "iree-run-module",
                f"--module={remote_model_path}",
                f"--function={self.function_name}",
                *remote_runtime_opts,
                *remote_output_args,
                *remote_input_args,
            ]
            self._logger.info("Running SoC test: %s", " ".join(cmd))
            runner.run_cmd(cmd)

            for remote_path, local_path in output_files.items():
                runner.copy_files(remote_path, str(local_path), board_dst=False)
            for remote_dir, local_dir in output_dirs.items():
                local_dir.parent.mkdir(parents=True, exist_ok=True)
                runner.copy_files(remote_dir, str(local_dir.parent), recursive=True, board_dst=False)
