# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Fake torq-compile / torq-run-module binaries for hardware-free tests.

The fakes record their argv to a file (path from an env var) so tests can assert
on the assembled command line, and produce the expected artifacts:
  * fake torq-compile writes a dummy VMFB at the ``-o`` path.
  * fake torq-run-module writes zero-filled outputs (sizes from an env var) for
    each ``--output=@`` arg and an optional host-profile CSV.
"""

import stat
from pathlib import Path

_FAKE_COMPILE = r"""#!/usr/bin/env python3
import os, sys, pathlib
argv = sys.argv[1:]
rec = os.environ.get("FAKE_COMPILE_RECORD")
if rec:
    pathlib.Path(rec).write_text("\n".join(argv))
out = None
for i, a in enumerate(argv):
    if a == "-o":
        out = argv[i + 1]
if out:
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_bytes(b"FAKEVMFB")
"""

_FAKE_RUN = r"""#!/usr/bin/env python3
import os, sys, pathlib
argv = sys.argv[1:]
rec = os.environ.get("FAKE_RUN_RECORD")
if rec:
    pathlib.Path(rec).write_text("\n".join(argv))
outputs = [a.split("@", 1)[1] for a in argv if a.startswith("--output=@")]
sizes = [int(s) for s in os.environ.get("FAKE_OUTPUT_SIZES", "").split(",") if s]
for i, out in enumerate(outputs):
    n = sizes[i] if i < len(sizes) else 4
    p = pathlib.Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00" * n)
for a in argv:
    if a.startswith("--torq_profile_host="):
        p = pathlib.Path(a.split("=", 1)[1])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("phase,us\ntotal,1\n")
"""


def _write_exe(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def write_fake_compile(directory) -> Path:
    return _write_exe(Path(directory) / "torq-compile", _FAKE_COMPILE)


def write_fake_run(directory) -> Path:
    return _write_exe(Path(directory) / "torq-run-module", _FAKE_RUN)
