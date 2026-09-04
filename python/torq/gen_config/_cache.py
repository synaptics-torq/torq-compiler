# Copyright 2025-2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Content-versioned artifact cache for the standalone discovery path.

The cache root defaults to ``${XDG_CACHE_HOME:-~/.cache}/torq-gen-config``
(the per-user XDG cache directory, so repeated runs reuse artifacts without
dropping a cache directory into the current working directory) and can be
overridden with the ``TORQ_GEN_CONFIG_CACHE_DIR`` environment variable.  Callers take a plain
:class:`Cache` (for the cache root directory and the small-value store)
and/or an explicit ``recompute: bool`` flag (for ``--recompute-cache``).
The versioning semantics:

- an artifact's version string chains from the version strings of its inputs,
  so changing any upstream input invalidates all downstream artifacts;
- stale artifacts are recomputed on demand, fresh ones are reused;
- ``recompute=True`` forces a rebuild of everything;
- concurrent processes are serialized with :class:`filelock.FileLock`.

Only the mechanics live here (dataclasses, hashing, locks, drivers); the
in-tree test framework wraps them in fixtures via
``torq.testing.versioned_fixtures``.  This module must not import
``torq.testing`` so the wheel stays lean.
"""

import hashlib
import inspect
import json
import logging
import os
import pickle
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Union

from filelock import FileLock

logger = logging.getLogger("torq.gen_config.cache")

#: Environment variable overriding the cache root directory.
CACHE_DIR_ENV_VAR = "TORQ_GEN_CONFIG_CACHE_DIR"

#: Name of the application cache directory inside the XDG cache home.
DEFAULT_CACHE_DIR_NAME = "torq-gen-config"


def default_cache_root() -> Path:
    """Default cache root: ``${XDG_CACHE_HOME:-~/.cache}/torq-gen-config``."""
    xdg_cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(xdg_cache) / DEFAULT_CACHE_DIR_NAME


class Cache:
    """Root directory and small-value store for the standalone cache.

    ``mkdir(name)`` returns (creating if needed) the ``<root>/d/<name>``
    subdirectory for artifact storage; ``get()``/``set()`` persist small
    JSON-serializable values under ``<root>/v/values.json``, guarded by a
    FileLock so concurrent processes cannot corrupt the store.
    """

    def __init__(self, root_dir: Optional[Union[str, Path]] = None) -> None:
        if root_dir is None:
            root_dir = os.environ.get(CACHE_DIR_ENV_VAR) or default_cache_root()
        # Resolve to an absolute path: derived artifact paths are handed to
        # subprocesses (e.g. torq-compile) that run with a different CWD.
        self._root = Path(root_dir).resolve()
        (self._root / "d").mkdir(parents=True, exist_ok=True)
        (self._root / "v").mkdir(parents=True, exist_ok=True)

    @property
    def root_dir(self) -> Path:
        """Root directory of the cache."""
        return self._root

    def mkdir(self, name: str) -> Path:
        """Return (creating if needed) the artifact subdirectory ``name``."""
        path = self._root / "d" / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def _values_path(self) -> Path:
        return self._root / "v" / "values.json"

    def _load_values(self) -> Dict[str, Any]:
        if not self._values_path.exists():
            return {}
        with open(self._values_path, "r") as f:
            return json.load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Read a small JSON-serializable value, or ``default`` if absent."""
        with FileLock(str(self._values_path) + ".lock"):
            values = self._load_values()
        return values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Store a small JSON-serializable value."""
        with FileLock(str(self._values_path) + ".lock"):
            values = self._load_values()
            values[key] = value
            with open(self._values_path, "w") as f:
                json.dump(values, f)


def _dataclass_dict_deep(obj: Any) -> Any:
    """Recursively convert dataclasses to plain dicts for serialization."""
    if isinstance(obj, dict):
        return {k: _dataclass_dict_deep(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_dataclass_dict_deep(v) for v in obj]
    elif is_dataclass(obj):
        return _dataclass_dict_deep(asdict(obj))
    else:
        return obj


def _hash_data(data: Any) -> str:
    """Return the hash of an object by serializing it to JSON using sorted keys."""
    hash_data = json.dumps(_dataclass_dict_deep(data), sort_keys=True).encode("utf-8")
    hash_obj = hashlib.sha256()
    hash_obj.update(hash_data)
    return hash_obj.hexdigest()


def _hash_file(cache: Cache, file_path: Path) -> str:
    """Return the hash of a file, caching the result based on the file's mtime."""
    cached_hashes = cache.get("torq_file_hashes", {})

    file_stat = file_path.stat()

    # check if we have a cached hash for this file and if the mtime matches
    if str(file_path) in cached_hashes:
        cached_entry = cached_hashes[str(file_path)]
        if cached_entry["mtime"] == file_stat.st_mtime:
            logger.debug(f"[file content hash cache hit] {file_path} -> {cached_entry['hash']}")
            return cached_entry["hash"]

    # compute the hash
    hash_obj = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            hash_obj.update(data)

    file_hash = hash_obj.hexdigest()

    # cache the hash
    cached_hashes[str(file_path)] = {"mtime": file_stat.st_mtime, "hash": file_hash}
    cache.set("torq_file_hashes", cached_hashes)

    logger.debug(f"[file content hash computed] {file_path} -> {file_hash}")

    return file_hash


def versioned_static_file(cache: Cache, name: str, file_path: Union[str, Path]) -> "VersionedFile":
    """Version an existing file by its content hash (mtime-memoized in ``cache``)."""
    path = Path(file_path)
    version = name + _hash_file(cache, path)
    logger.debug(f"[static file] {name} -> {path}")
    logger.debug(f"[static file version] {name} -> {version}")
    return VersionedFile(path, version)


@dataclass
class VersionedFile:
    """A versioned file; the content is not loaded into memory automatically.

    Any two versioned artifacts with the same version string are considered
    equivalent (even if they are of different types).
    """

    file_path: Path
    version: str

    def valid(self, recompute: bool = False) -> bool:
        return self.file_path.exists() and not recompute

    @staticmethod
    def build(
        cache: Cache, name: str, suffix: str, input_versions: Sequence[str]
    ) -> "VersionedFile":
        files_dir = cache.mkdir("versioned_fixtures") / name
        files_dir.mkdir(parents=True, exist_ok=True)

        # namespace the version with the name so two artifacts with the
        # same input versions don't collide
        version = name + "." + _hash_data(list(input_versions))

        file_path = files_dir / f"{version}.{suffix}"

        return VersionedFile(file_path, version)


@dataclass
class VersionedDirectory:
    """A versioned directory generated on demand.

    A ``valid`` sentinel file inside the directory marks a completed
    generation; its absence means the content is partial and must be rebuilt.
    """

    dir_path: Path
    version: str

    def valid(self, recompute: bool = False) -> bool:
        return (self.dir_path / "valid").exists() and not recompute

    @staticmethod
    def build(cache: Cache, name: str, input_versions: Sequence[str]) -> "VersionedDirectory":
        dirs_dir = cache.mkdir("versioned_fixtures") / name
        dirs_dir.mkdir(parents=True, exist_ok=True)

        # namespace the version with the name so two artifacts with the
        # same input versions don't collide
        version = name + "." + _hash_data(list(input_versions))

        dir_path = dirs_dir / version

        dir_path.mkdir(parents=True, exist_ok=True)

        return VersionedDirectory(dir_path, version)


@dataclass
class VersionedUncachedData:
    """A versioned data object that is not cached to disk."""

    data: Any
    version: str

    @staticmethod
    def build(
        data: Any, generating_fun: Callable, input_versions: Sequence[str]
    ) -> "VersionedUncachedData":
        # the version includes the source of the generating function so that
        # editing the generator invalidates downstream artifacts
        version = generating_fun.__name__ + _hash_data(
            [list(input_versions) + [inspect.getsource(generating_fun)]]
        )
        return VersionedUncachedData(data, version)


@dataclass
class VersionedData:
    """A versioned data object cached to disk by pickling it."""

    file_path: Path
    version: str
    data: Any = None

    def save(self, data: Any) -> None:
        self.data = data

        with open(self.file_path, "wb") as f:
            pickle.dump(data, f)

    def load(self, recompute: bool = False) -> bool:
        """Load the cached data; return False if absent or ``recompute`` is set."""
        if not self.file_path.exists() or recompute:
            return False

        with open(self.file_path, "rb") as f:
            self.data = pickle.load(f)

        return True

    @staticmethod
    def build(cache: Cache, name: str, input_versions: Sequence[str]) -> "VersionedData":
        files_dir = cache.mkdir("versioned_fixtures") / name
        files_dir.mkdir(parents=True, exist_ok=True)

        # namespace the version with the name so two artifacts with the
        # same input versions don't collide
        version = name + "." + _hash_data(list(input_versions))

        file_path = files_dir / f"{version}.pkl"

        return VersionedData(file_path, version)


def get_or_generate_file(
    cache: Cache,
    name: str,
    suffix: str,
    input_versions: Sequence[str],
    generate_fn: Callable[[Path], None],
    recompute: bool = False,
) -> VersionedFile:
    """Return the versioned file for ``name``, generating it under a lock if stale.

    ``generate_fn`` receives the target path and must write the file there.
    """
    logger.debug(f"[generating versioned file] {name}")

    versioned_file = VersionedFile.build(cache, name, suffix, input_versions)

    with FileLock(str(versioned_file.file_path) + ".lock"):
        if not versioned_file.valid(recompute):
            logger.debug(f"[cache miss] {name} -> {versioned_file.file_path}")
            generate_fn(versioned_file.file_path)
        else:
            logger.debug(f"[cache hit] {name} -> {versioned_file.file_path}")

    return versioned_file


def get_or_generate_directory(
    cache: Cache,
    name: str,
    input_versions: Sequence[str],
    generate_fn: Callable[[Path], None],
    recompute: bool = False,
) -> VersionedDirectory:
    """Return the versioned directory for ``name``, generating it under a lock if stale.

    ``generate_fn`` receives the target directory and must populate it; a
    ``valid`` sentinel is written after ``generate_fn`` returns successfully.
    """
    logger.debug(f"[generating versioned directory] {name}")

    versioned_dir = VersionedDirectory.build(cache, name, input_versions)

    with FileLock(str(versioned_dir.dir_path) + ".lock"):
        if not versioned_dir.valid(recompute):
            logger.debug(f"[cache miss] {name} -> {versioned_dir.dir_path}")
            generate_fn(versioned_dir.dir_path)

            # mark the directory as valid
            with open(versioned_dir.dir_path / "valid", "w") as f:
                pass
        else:
            logger.debug(f"[cache hit] {name} -> {versioned_dir.dir_path}")

    return versioned_dir


def get_or_compute_data(
    cache: Cache,
    name: str,
    input_versions: Sequence[str],
    compute_fn: Callable[[], Any],
    recompute: bool = False,
) -> VersionedData:
    """Return the cached data for ``name``, computing and pickling it under a lock if stale."""
    logger.debug(f"[generating versioned data] {name}")

    versioned_data = VersionedData.build(cache, name, input_versions)

    with FileLock(str(versioned_data.file_path) + ".lock"):
        if not versioned_data.load(recompute):
            logger.debug(f"[cache miss] {name} -> {versioned_data.file_path}")
            versioned_data.save(compute_fn())
        else:
            logger.debug(f"[cache hit] {name} -> {versioned_data.file_path}")

    return versioned_data
