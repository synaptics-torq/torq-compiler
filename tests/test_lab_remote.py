# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Remote execution tests using a fake transport (no hardware)."""

from pathlib import Path

from torq.lab.pipeline import ModelPipeline
from torq.lab.remote import RemoteExecutor, parse_board_wall_time
from torq.lab.types import PipelineConfig, RemoteTarget

TOSA_MLIR = """
module {
  func.func @main(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
    return %arg0 : tensor<1x4xf32>
  }
}
"""


class FakeRunner:
    """Records run_cmd/copy_files; returns a canned 'time' output; fakes pullback."""

    def __init__(self, time_output="real\t0m2.500s\nuser\t0m0.100s\n", pull_payload=b"\x00" * 16):
        self.calls = []
        self.copies = []
        self.time_output = time_output
        self.pull_payload = pull_payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run_cmd(self, cmd):
        self.calls.append(cmd)
        if isinstance(cmd, list) and cmd and isinstance(cmd[0], str) and cmd[0].startswith("time {"):
            return self.time_output
        return ""

    def copy_files(self, src, dst, recursive=False, board_dst=False, verbose=False):
        self.copies.append((str(src), str(dst), recursive, board_dst))
        if not board_dst:  # pull: materialize the local destination
            p = Path(dst)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(self.pull_payload)


def test_parse_board_wall_time():
    assert parse_board_wall_time("real\t0m2.500s\n") == 2.5
    assert parse_board_wall_time("real 1m3.000s") == 63.0
    assert parse_board_wall_time("no time here") is None
    assert parse_board_wall_time(None) is None


def test_rewrite_arg_with_remote_path():
    ex = RemoteExecutor(RemoteTarget("host"), "/l/model.vmfb", "main", [], [], [])
    arg, local, remote = ex._rewrite_arg_with_remote_path("--input=1x4xf32=@/local/in.bin")
    assert arg == "--input=1x4xf32=@/tmp/model/in.bin"
    assert local == Path("/local/in.bin")
    assert str(remote) == "/tmp/model/in.bin"
    assert ex._rewrite_arg_with_remote_path("--flag") == ("--flag", None, None)


def test_rewrite_runtime_opts():
    ex = RemoteExecutor(
        RemoteTarget("host"), "/l/model.vmfb", "main", [], [],
        ["--torq_profile_host=/l/host.csv", "--torq_dump_buffers_dir=/l/buf", "--plain=1"],
    )
    fake = FakeRunner()
    opts, out_files, out_dirs = ex._rewrite_runtime_opts(fake)
    assert "--torq_profile_host=/tmp/model/host.csv" in opts
    assert "--torq_dump_buffers_dir=/tmp/model/buf" in opts
    assert "--plain=1" in opts
    assert out_files == {"/tmp/model/host.csv": Path("/l/host.csv")}
    assert out_dirs == {"/tmp/model/buf": Path("/l/buf")}
    assert ["mkdir", "-p", "/tmp/model/buf"] in fake.calls


def test_remote_executor_run(tmp_path, monkeypatch):
    vmfb = tmp_path / "model.vmfb"
    vmfb.write_bytes(b"x")
    in_bin = tmp_path / "in.bin"
    in_bin.write_bytes(b"y")
    local_out = tmp_path / "out" / "outputs" / "output_0.bin"
    local_prof = tmp_path / "out" / "profiles" / "host_profile.csv"

    fake = FakeRunner()
    monkeypatch.setattr("torq.lab.remote.remote_command_runner_factory", lambda *a, **k: fake)

    ex = RemoteExecutor(
        RemoteTarget(address="user@host", port=2222),
        vmfb, "main",
        [f"--input=1x4xf32=@{in_bin}"],
        [f"--output=@{local_out}"],
        ["--torq_hw_type=astra_machina", f"--torq_profile_host={local_prof}"],
        timeout=15,
    )
    outcome = ex.run()

    assert outcome.wall_time == 2.5
    assert "--module=/tmp/model/model.vmfb" in outcome.command
    assert "--function=main" in outcome.command
    assert "--input=1x4xf32=@/tmp/model/in.bin" in outcome.command
    assert "--output=@/tmp/model/output_0.bin" in outcome.command
    assert "--torq_profile_host=/tmp/model/host_profile.csv" in outcome.command
    # vmfb + input staged to the board
    assert any(src == str(vmfb) and board for (src, _dst, _r, board) in fake.copies)
    assert any(src == str(in_bin) and board for (src, _dst, _r, board) in fake.copies)
    # outputs + profile pulled back locally
    assert local_out.exists()
    assert local_prof.exists()


def test_pipeline_run_remote(tmp_path, monkeypatch):
    model = tmp_path / "model.mlir"
    model.write_text(TOSA_MLIR)
    vmfb = tmp_path / "model.vmfb"
    vmfb.write_bytes(b"x")

    fake = FakeRunner()
    monkeypatch.setattr("torq.lab.remote.remote_command_runner_factory", lambda *a, **k: fake)

    config = PipelineConfig(
        model_path=model, work_dir=tmp_path / "out",
        random_inputs=True, profile_runtime=True, runtime_hw_type="astra_machina",
    )
    result = ModelPipeline(config, remote=RemoteTarget(address="user@host")).run(vmfb)

    assert result.wall_time == 2.5
    assert len(result.outputs) == 1
    assert result.outputs[0].shape == (1, 4)
    assert result.host_profile is not None and result.host_profile.exists()
