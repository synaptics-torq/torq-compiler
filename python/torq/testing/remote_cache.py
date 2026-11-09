"""
A store of cached fixture results shared across test sessions (see versioned_fixtures).

A session reads entries from the stores of --remote-cache-read, searched in order, and writes the
entries it computes to the store of --remote-cache-write (which it also reads first). A store is
a local directory or an S3 prefix (s3://bucket/prefix). Each entry is one archive,
<fixture name>/<version>.tar.gz, written once and never changed: its version is the hash of what
it depends on.

Entries are only written for results that were computed successfully. A directory with a
.no_remote_cache file (e.g. the results of a run on a board) is not written. A directory fetched
from a store gets a .from_remote_cache file, so that the times measured when it was computed, on
another machine, are not reported again. Any error of a store disables it for the rest of the
process, with a warning: the tests then compute what they need.
"""

import logging
import os
import shutil
import tarfile
import tempfile
import warnings
from pathlib import Path

logger = logging.getLogger("versioned_fixtures")

# files of a directory entry that are never stored: the compiler's IR dumps, which are large and
# not needed to reuse a compile, and the markers of the local cache
_EXCLUDED_DIR_ENTRIES = {"phases", "ir", "valid"}

NO_REMOTE_CACHE_MARKER = ".no_remote_cache"
FROM_REMOTE_CACHE_MARKER = ".from_remote_cache"


class LocalStore:
    def __init__(self, root):
        self.root = Path(root)
        self.name = str(root)

    def exists(self, key):
        return (self.root / key).exists()

    def get(self, key, dest):
        path = self.root / key
        if not path.exists():
            return False
        shutil.copyfile(path, dest)
        return True

    def put(self, key, src):
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, path)


class S3Store:
    def __init__(self, url):
        import boto3
        bucket, _, prefix = url[len("s3://"):].partition("/")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.name = url
        self.s3 = boto3.client("s3")

    def _key(self, key):
        return f"{self.prefix}/{key}" if self.prefix else key

    def _missing(self, error):
        # needs s3:ListBucket: without it, S3 answers 403 for a missing key, which disables the store
        return error.response.get("Error", {}).get("Code") in ("404", "NoSuchKey")

    def exists(self, key):
        import botocore
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except botocore.exceptions.ClientError as e:
            if self._missing(e):
                return False
            raise

    def get(self, key, dest):
        import botocore
        try:
            self.s3.download_file(self.bucket, self._key(key), str(dest))
            return True
        except botocore.exceptions.ClientError as e:
            if self._missing(e):
                return False
            raise

    def put(self, key, src):
        self.s3.upload_file(str(src), self.bucket, self._key(key))


def _open_store(url):
    if url.startswith("s3://"):
        return S3Store(url)
    return LocalStore(url[len("file://"):] if url.startswith("file://") else url)


def _try_open_store(url):
    try:
        return _open_store(url)
    except Exception as e:  # e.g. boto3 not installed
        warnings.warn(f"remote test cache {url} disabled: {e}")
        return None


class RemoteCache:
    def __init__(self, read_urls, write_url):
        self.write_store = _try_open_store(write_url) if write_url else None
        read_stores = [_try_open_store(u) for u in read_urls if u and u != write_url]
        self.read_stores = ([self.write_store] if self.write_store else []) + [s for s in read_stores if s]

    def _disable(self, store, error):
        warnings.warn(f"remote test cache {store.name} disabled: {error}")
        self.read_stores = [s for s in self.read_stores if s is not store]
        if self.write_store is store:
            self.write_store = None

    def fetch(self, fixture_name, version, dest, is_dir):
        """Puts the entry in dest (a file, or the contents of a directory); False if no store has it."""
        key = f"{fixture_name}/{version}.tar.gz"
        for store in list(self.read_stores):
            try:
                with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
                    archive = Path(tmp) / "entry.tar.gz"
                    if not store.get(key, archive):
                        continue
                    # created first: the archive of an empty directory has no members
                    extracted = Path(tmp) / "entry"
                    extracted.mkdir()
                    with tarfile.open(archive) as tar:
                        tar.extractall(extracted, filter="data")
                    if is_dir:
                        # what a compute that failed left there (e.g. debug/ of a compile that
                        # timed out) would stop the entries from being moved in
                        for item in dest.iterdir():
                            if item.is_dir() and not item.is_symlink():
                                shutil.rmtree(item)
                            else:
                                item.unlink()
                        for item in extracted.iterdir():
                            os.replace(item, dest / item.name)
                        (dest / FROM_REMOTE_CACHE_MARKER).touch()
                    else:
                        os.replace(extracted / "file", dest)
                logger.debug(f"[remote cache hit] {key} from {store.name}")
                return True
            except Exception as e:
                self._disable(store, e)
        return False

    def store(self, fixture_name, version, src, is_dir):
        """Writes the entry in src to the write store, unless it is there already."""
        store = self.write_store
        if store is None or (is_dir and (src / NO_REMOTE_CACHE_MARKER).exists()):
            return
        key = f"{fixture_name}/{version}.tar.gz"
        try:
            if store.exists(key):
                return
            with tempfile.TemporaryDirectory(dir=src.parent) as tmp:
                archive = Path(tmp) / "entry.tar.gz"
                with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
                    if is_dir:
                        for item in sorted(src.iterdir()):
                            if item.name not in _EXCLUDED_DIR_ENTRIES and not item.name.endswith(".lock"):
                                tar.add(item, arcname=item.name)
                    else:
                        tar.add(src, arcname="file")
                store.put(key, archive)
            logger.debug(f"[remote cache store] {key} to {store.name}")
        except Exception as e:
            self._disable(store, e)
