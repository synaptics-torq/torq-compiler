"""Helpers for writing pytest test-generation caches safely.

The torch/onnx/tflite test modules have to inspect a model before they can
build test cases: they load the model, run a forward pass with hooks attached,
and record every layer's input/output shapes. That preprocessing can take
tens of seconds, so the result is cached on disk under
``.pytest_cache/d/<cache_name>/<key>/``.

Cache flow
----------
1. A test module asks for the cached metadata for a given model key.
2. If ``manifest.json`` exists and its version/size metadata still matches the
   source model, the cached cases are returned immediately.
3. Otherwise the model is re-processed, the cache directory is populated with
   the new artifacts, and ``manifest.json`` is written last.

``manifest.json`` is the cache entry point. It is a JSON file that records the
cache format version, the source model's size/mtime (or version string), and
the list of per-layer test cases with their layer names and input/output
shapes. Other artifacts, such as per-layer ``.onnx`` or ``.tflite`` files,
live in the same directory and are referenced from the manifest.

Why locking and atomic writes are needed
----------------------------------------
pytest-xdist launches multiple independent Python processes. Each process
imports the test modules and runs ``pytest_generate_tests`` for the same
model, so several workers may try to generate the same cache at the same time.
That creates two problems:

* **Multiple writers.** Only one worker should generate the cache; the rest
  should wait and then read the finished result. Callers must therefore
  acquire a ``FileLock`` for the model key before reading or writing the cache.
  The first worker that acquires the lock creates the cache; the others block
  until it finishes and then read the cached ``manifest.json``.

* **Readers must never see a partial cache.** The cache must never contain a
  half-written manifest or model file. We therefore avoid deleting or replacing
  the whole cache directory while another worker may be reading it. On CI
  runners that use NFS, ``shutil.rmtree`` on a directory that contains a lock
  file held by another process can fail with ``OSError: Directory not empty``.

How to use these helpers
------------------------
While holding the cache lock (and with the lock file kept *outside* the cache
key directory), write every artifact through a temporary file and rename it
into place:

    tmp = key_dir / f"manifest.json.tmp.{os.getpid()}"
    tmp.write_text(...)
    tmp.rename(key_dir / "manifest.json")

POSIX ``rename`` is atomic: a reader sees either the old file or the new file,
never a partially written one. ``atomic_write_json_file`` wraps this for JSON
manifests; front ends that write binary artifacts (e.g. ONNX models) should
follow the same temp-file + rename pattern.
"""

import json
import os
from pathlib import Path


def atomic_write_json_file(key_dir: Path, filename: str, data) -> None:
    """Atomically write ``filename`` as JSON inside ``key_dir``.

    Steps:
      1. Make sure ``key_dir`` exists.
      2. Write to ``<filename>.tmp.<pid>`` in the same directory.
      3. Rename the temp file to ``<filename>``.

    This assumes the caller already holds the cache lock for this key, so no
    other process is writing the same file at the same time.
    """
    key_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = key_dir / f"{filename}.tmp.{os.getpid()}"
    tmp_path.unlink(missing_ok=True)
    tmp_path.write_text(json.dumps(data, indent=2))
    tmp_path.rename(key_dir / filename)


def atomic_write_json_manifest(key_dir: Path, manifest: dict) -> None:
    """Atomically write ``manifest.json`` inside ``key_dir``."""
    atomic_write_json_file(key_dir, "manifest.json", manifest)
