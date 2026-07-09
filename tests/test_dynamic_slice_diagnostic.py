"""Regression test for graceful handling of data-dependent slice offsets.

A `stablehlo.dynamic_slice` with a runtime start index lowers to a
`memref.subview` with a dynamic strided-layout offset. Torq address
resolution needs a compile-time-constant offset, so this cannot be
supported today (tracked separately as dynamic-address support). What we
guarantee here is that the compiler rejects it with a clear, located
diagnostic instead of aborting with an ICE (`LLVM ERROR: Unsupported
dynamic offsets`).
"""

import subprocess

import pytest

# Minimal repro: the dim-1 start index (%arg1) is a runtime value, so the
# resulting slice has a data-dependent memory offset.
_DYNAMIC_SLICE_MLIR = """\
func.func @main(%arg0: tensor<1x128x1x256xbf16>, %arg1: tensor<i32>,
                %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>)
    -> tensor<1x9x1x256xbf16> {
  %0 = stablehlo.dynamic_slice %arg0, %arg1, %arg2, %arg3, %arg4,
      sizes = [1, 9, 1, 256]
    : (tensor<1x128x1x256xbf16>, tensor<i32>, tensor<i32>, tensor<i32>,
       tensor<i32>) -> tensor<1x9x1x256xbf16>
  return %0 : tensor<1x9x1x256xbf16>
}
"""


@pytest.mark.ci
def test_dynamic_slice_reports_graceful_diagnostic(tmp_path, torq_compiler):
    src = tmp_path / "dynamic_slice.mlir"
    src.write_text(_DYNAMIC_SLICE_MLIR)

    result = subprocess.run(
        [
            str(torq_compiler.file_path),
            "--iree-input-type=auto",
            "--iree-hal-target-backends=torq",
            str(src),
            "-o",
            str(tmp_path / "dynamic_slice.vmfb"),
        ],
        capture_output=True,
        text=True,
    )

    # Compilation must fail: a data-dependent offset has no static address.
    assert result.returncode != 0, "expected compilation to fail"
    # It must fail with our clear, located diagnostic ...
    assert "data-dependent (dynamic) memory offset" in result.stderr, result.stderr
    # ... and NOT abort with an internal-compiler-error crash.
    assert "LLVM ERROR" not in result.stderr, result.stderr
    assert "Please report issues" not in result.stderr, result.stderr
