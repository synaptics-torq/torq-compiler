import re
from pathlib import Path

import pytest

from .test_lram_loads import dispatch_after


MODEL = Path(__file__).parents[1] / "testdata" / "linalg_ops" / "shared-keepdims-reduction-i8.mlir"


@pytest.mark.ci
def test_shared_keepdims_reduction_is_recovered_and_tiled(torq_compiler, tmp_path):
    """The keepdims reduction is shared with a returned output and feeds an oversized add.
    Tiling the group rewrites the reduction, which stays full-size for the returned output.
    Compile fully, including LRAM assignment, and ensure the add has no full-size result.

    This covers the rewrite of a reduction kept alive by a side user. It passes without the
    residual tiling too: the full-size reduction fits, so nothing is left to tile."""
    dump = dispatch_after(
        torq_compiler,
        tmp_path,
        MODEL,
        "SL2610",
        "torq-tile-and-fuse",
        [],
    )

    assert "scf.for" in dump, f"the oversized group was not tiled:\n{dump}"
    full_size_generics = re.findall(
        r"(?ms)^\s*%[^\n]*=\s*linalg\.generic\b(?:(?!^\s*\}).)*"
        r"^\s*\}\s*->\s*tensor<1x1024x32x32xi8>",
        dump,
    )
    assert not full_size_generics, (
        "a full-size Linalg result remains after tile-and-fuse:\n"
        + "\n".join(full_size_generics)
        + f"\nFull dispatch:\n{dump}"
    )
