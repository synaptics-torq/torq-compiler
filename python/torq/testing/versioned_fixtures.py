import pytest

from filelock import FileLock

from dataclasses import dataclass
import hashlib
import json
from dataclasses import is_dataclass, asdict
from pathlib import Path
import pickle
from functools import wraps
import inspect
import hashlib
import importlib.metadata
import logging
import subprocess
import sys
import os
from collections import Counter, defaultdict

from .remote_cache import FROM_REMOTE_CACHE_MARKER, RemoteCache

"""
This module provides decorators that allow to create fixtures that cache their results.

In order to properly invalidate the cache the decorators compute a version string
based on the input parameters of the fixture. If any of the input parameters change
the version string will change and the fixture will be recomputed.

In order to understand if the input parameters have changed, the decorators check the
version of the input parameters which must be other versioned fixtures. In some cases
a fixture requires inputs that do not need or can't be cached to disk. The module
provide special decorators for these inputs that always unconditionally recompute the
fixture and create a version for it based on its hash or the inputs used to generate it.
"""


def pytest_addoption(parser):
    parser.addoption("--recompute-cache", action="store_true", default=False, help="Re-creates all cached artifacts instead of using existing cached values")
    parser.addoption("--debug-cache", action="store_true", default=False, help="Enables debug output for cache operations")
    parser.addoption("--remote-cache-read", default=os.environ.get("TORQ_REMOTE_CACHE_READ", ""),
                     help="Comma-separated stores (directories or s3://bucket/prefix) to read cached results from (default: $TORQ_REMOTE_CACHE_READ)")
    parser.addoption("--remote-cache-write", default=os.environ.get("TORQ_REMOTE_CACHE_WRITE", ""),
                     help="Store to write the computed results to, and to read first (default: $TORQ_REMOTE_CACHE_WRITE)")
    parser.addoption("--test-cache-summary", default=None,
                     help="Markdown file to append the cache use of the session to, per test file and per fixture (e.g. $GITHUB_STEP_SUMMARY)")

logger = logging.getLogger("versioned_fixtures")

# the store of results shared across sessions, if any (see remote_cache)
_remote = None


def pytest_configure(config):
    global _remote

    if config.getoption("--debug-cache"):
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.StreamHandler())

    read_urls = [u.strip() for u in config.getoption("--remote-cache-read").split(",") if u.strip()]
    write_url = config.getoption("--remote-cache-write").strip()
    if read_urls or write_url:
        _remote = RemoteCache(read_urls, write_url)


def local_only(fun):
    """
    Marks a fixture whose results are not shared across sessions (see remote_cache): they are
    cheap to compute again, and would only make the shared store bigger. Put it under the
    versioned fixture decorator.
    """
    fun.local_only = True
    return fun


def _fetch_remote(request, fun, version, dest, is_dir):
    """Puts the result of a fixture in dest from the shared store; False if it isn't there."""
    if _remote is None or getattr(fun, "local_only", False) or request.config.getoption("--recompute-cache"):
        return False
    return _remote.fetch(fun.__name__, version, dest, is_dir)


def _store_remote(fun, version, src, is_dir):
    if _remote is not None and not getattr(fun, "local_only", False):
        _remote.store(fun.__name__, version, src, is_dir)


# how the tests got the results of the fixtures, per test file: Counter of (fixture, use), use
# being "computed", "reused" (from the local cache) or "fetched" (from a store)
_cache_uses = defaultdict(Counter)


def _record_cache_use(request, fun, use, from_store=False):
    """
    from_store: the result was fetched from a store, by this test or (for a reused directory) by
    an earlier one, so its measured times are not reported
    """
    # the local_only results are never in a store: they would only lower the hit rates
    if not getattr(fun, "local_only", False):
        request.node.user_properties.append(("test_cache_use", (fun.__name__, use, from_store or use == "fetched")))


def pytest_runtest_logreport(report):
    # the teardown report has the properties recorded by all the fixtures of the test
    if report.when == "teardown":
        test_file = report.nodeid.split("::")[0]
        for name, value in report.user_properties:
            if name == "test_cache_use":
                _cache_uses[test_file][(value[0], value[1])] += 1


def _cache_use_summary():
    def row(name, uses):
        total = sum(uses.values())
        hits = uses["reused"] + uses["fetched"]
        return (f"| {name} | {total} | {uses['reused']} | {uses['fetched']} | {uses['computed']} | "
                f"{100 * hits / total:.0f}% |")

    header = ["| {} | Results | Reused | Fetched | Computed | Hit rate |", "|---|---:|---:|---:|---:|---:|"]
    lines = ["### Test cache", "", header[0].format("Test file"), header[1]]
    per_fixture = defaultdict(Counter)
    for test_file, uses in sorted(_cache_uses.items()):
        per_test_file = Counter()
        for (fixture, use), count in uses.items():
            per_test_file[use] += count
            per_fixture[fixture][use] += count
        lines.append(row(test_file, per_test_file))
    lines += ["", header[0].format("Fixture"), header[1]]
    lines += [row(fixture, uses) for fixture, uses in sorted(per_fixture.items())]
    return "\n".join(lines) + "\n"


def pytest_sessionfinish(session):
    summary_file = session.config.getoption("--test-cache-summary")
    # under xdist, only the controller has the reports of all the workers
    if summary_file and _cache_uses and not hasattr(session.config, "workerinput"):
        with open(summary_file, "a") as f:
            f.write(_cache_use_summary())


# What the cached results depend on beyond the inputs of the fixtures: the tools that make the
# models and references, and the code of the tests. It is part of every version, so that a
# cache shared across sessions is not reused by other code. Set for each test before its
# fixtures.
_salt = None

# salt part of the session, computed once per process
_global_salt = None

# salt part of each test module
_module_salts = {}


def _files_hash(root, files):
    return _hash_data([(str(Path(f).relative_to(root)), hashlib.sha256(Path(f).read_bytes()).hexdigest())
                       for f in files])


def _gitlink(root, path):
    """The commit of a submodule in the checkout, without needing the submodule itself."""
    try:
        out = subprocess.run(["git", "ls-tree", "HEAD", path], cwd=root, capture_output=True, text=True,
                             check=True).stdout.split()
        return out[2] if len(out) > 2 else "none"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _get_global_salt(root):
    """
    The tools the tests use to make models and references (the installed packages but our own
    wheels, which are versioned by their build time, and the iree and iree-turbine commits, which
    also cover the LLVM-CPU references and the ONNX import), and the code of the torq packages.
    """
    global _global_salt
    if _global_salt is None:
        packages = sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
                           if d.metadata['Name'] and not d.metadata['Name'].lower().startswith("torq")})
        _global_salt = _hash_data({
            "python": sys.version,
            "packages": packages,
            "iree": _gitlink(root, "third_party/iree"),
            "iree-turbine": _gitlink(root, "third_party/iree-turbine"),
            "torq": _files_hash(root, sorted((root / "python" / "torq").rglob("*.py"))),
        })
    return _global_salt


def _get_module_salt(root, module_path):
    """
    The code of a test module: its file, the helper modules next to it (not test_*.py) and the
    conftest.py files above it.
    """
    if module_path not in _module_salts:
        folder = module_path.parent
        files = [module_path] + sorted(p for p in folder.glob("*.py") if not p.name.startswith("test_"))
        files += [d / "conftest.py" for d in [folder, *folder.parents]
                  if (d / "conftest.py").exists() and d.is_relative_to(root)]
        _module_salts[module_path] = _files_hash(root, sorted(set(files)))
    return _module_salts[module_path]


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    global _salt
    root = Path(item.config.rootpath)
    module_path = Path(item.path)
    if module_path.is_relative_to(root):
        _salt = _hash_data([_get_global_salt(root), _get_module_salt(root, module_path)])
    else:
        _salt = _get_global_salt(root)


def _dataclass_dict_deep(obj):
    if isinstance(obj, dict):
        return {k: _dataclass_dict_deep(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_dataclass_dict_deep(v) for v in obj]
    elif is_dataclass(obj):
        return _dataclass_dict_deep(asdict(obj))
    else:
        return obj


def _hash_data(data) -> str:
    """
    Returns the hash of an object by serializing it to JSON using sorted keys.
    """    
    hash_data = json.dumps(_dataclass_dict_deep(data), sort_keys=True).encode('utf-8')
    hash_obj = hashlib.sha256()
    hash_obj.update(hash_data)
    return hash_obj.hexdigest()


def _hash_file(request, file_path):
    """
    Returns the hash of a file, caching the result based on the file's mtime.
    """

    cache = request.config.cache
    cached_hashes = cache.get("torq_file_hashes", {})

    file_stat = file_path.stat()

    # check if we have a cached hash for this file and if the mtime matches
    if str(file_path) in cached_hashes:
        cached_entry = cached_hashes[str(file_path)]
        if cached_entry['mtime'] == file_stat.st_mtime:
            logger.debug(f"[file content hash cache hit] {file_path} -> {cached_entry['hash']}")
            return cached_entry['hash']
        
    # compute the hash
    hash_obj = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            hash_obj.update(data)

    file_hash = hash_obj.hexdigest()

    # cache the hash
    cached_hashes[str(file_path)] = {
        'mtime': file_stat.st_mtime,
        'hash': file_hash
    }
    cache.set("torq_file_hashes", cached_hashes)

    logger.debug(f"[file content hash computed] {file_path} -> {file_hash}")

    return file_hash


# content hash of each file, by (path, mtime, size), computed once per process
_content_hashes = {}


def content_hash(path) -> str:
    """
    Returns the hash of the content of a file, or of the files of a directory and their paths
    relative to it ("missing" if path does not exist).

    Unlike _hash_file, the hashes are not stored in the pytest cache: this is meant for the outputs
    of fixtures, which are many and change in each session.
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file())
        return _hash_data([(str(p.relative_to(path)), content_hash(p)) for p in files])
    if not path.exists():
        return "missing"

    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _content_hashes:
        hash_obj = hashlib.sha256()
        with open(path, 'rb') as f:
            for data in iter(lambda: f.read(1 << 20), b""):
                hash_obj.update(data)
        _content_hashes[key] = hash_obj.hexdigest()
    return _content_hashes[key]


def _get_fixture_params(kwargs):
    input_params_versions = []

    kwargs_data = {}

    sorted_keys = sorted(kwargs.keys())

    for name in sorted_keys:

        value = kwargs[name]

        if name == 'request':
            kwargs_data['request'] = value
            continue

        # we will replace this with the real value
        if name == 'versioned_file' or name == 'versioned_dir':
            continue
                
        if isinstance(value, VersionedData):
            kwargs_data[name] = value.data
        elif isinstance(value, VersionedFile):
            kwargs_data[name] = value.file_path
        elif isinstance(value, VersionedUncachedData):
            kwargs_data[name] = value.data
        elif isinstance(value, VersionedDirectory):
            kwargs_data[name] = value.dir_path
        else:
            raise ValueError(f"Unsupported argument type {type(value)} for fixture parameter '{name}'")
        
        logger.debug("[param version] " + name + " -> " + value.version)
        input_params_versions.append(value.version)

    if _salt is not None:
        input_params_versions.append("salt." + _salt)

    return input_params_versions, kwargs_data


@pytest.fixture
def versioned_file():
    """
    This is a fake fixture that is used as placeholder to pass the path to the versioned file that
    will be generated by fixture decorated with the versioned_generated_file_fixture decorator.

    The actual parampeter passed to the wrapped function is a Path object pointing to the
    generated file and not this value.

    This fixture is required to make sure pytest sees the same parameters list on the original
    function being decorated and the wrapping function that takes care of the caching logic.
    """

    return None


@pytest.fixture
def versioned_dir():
    """
    This is a fake fixture that is used as placeholder to pass the path to
    the versioned directory that will be generated by fixture decorated with the versioned_generated_directory_fixture decorator.
    The actual parampeter passed to the wrapped function is a Path object pointing to the
    generated directory and not this value.
    This fixture is required to make sure pytest sees the same parameters list on the original
    function being decorated and the wrapping function that takes care of the caching logic.
    """
    
    return None


@dataclass
class VersionedFile:
    """
    Represents a versioned file generated by a fixture, the content of the file are not loaded
    to memory automatically.
    """


    """
    Path to this version of the fixture
    """
    file_path: Path

    """
    Version string of this fixture, any two versioned fixtures with the same version string
    are considered equivalent (even if they are of different types)
    """
    version: str

    def valid(self, request):
        return self.file_path.exists() and not request.config.getoption("--recompute-cache")
    
    @staticmethod
    def build(request, fixture_name: str, suffix: str, input_params) -> 'VersionedFile':
        cached_fixture_dir = Path(request.config.cache.makedir("versioned_fixtures"))
        files_dir = Path(cached_fixture_dir) / fixture_name        
        files_dir.mkdir(exist_ok=True)

        # we namespace the version with the dir_name to make sure two
        # fixtures with the same input_params don't collide
        version = fixture_name + "." + _hash_data(input_params)

        file_path = files_dir / f"{version}.{suffix}"

        return VersionedFile(file_path, version)


@dataclass
class VersionedDirectory:
    """
    Represents a versioned directory generated by a fixture.
    """


    """
    Path to this version of the fixture
    """
    dir_path: Path

    """
    Version string of this fixture, any two versioned fixtures with the same version string
    are considered equivalent (even if they are of different types)
    """
    version: str

    def valid(self, request):
        return (self.dir_path / "valid").exists() and not request.config.getoption("--recompute-cache")
    
    @staticmethod
    def build(request, fixture_name: str, input_params) -> 'VersionedDirectory':
        cached_fixture_dir = Path(request.config.cache.makedir("versioned_fixtures"))
        dirs_dir = Path(cached_fixture_dir) / fixture_name        
        dirs_dir.mkdir(exist_ok=True)

        # we namespace the version with the dir_name to make sure two
        # fixtures with the same input_params don't collide    
        version = fixture_name + "." + _hash_data(input_params)

        dir_path = dirs_dir / f"{version}"

        dir_path.mkdir(exist_ok=True)

        return VersionedDirectory(dir_path, version)


@dataclass
class VersionedUncachedData:
    """
    Represents a versioned data object that is not cached to disk
    """

    """
    Data that is being versioned
    """
    data: any

    """
    Version string of this fixture, any two versioned fixtures with the same version string
    are considered equivalent (even if they are of different types)
    """
    version: str

    @staticmethod
    def build(data, generating_fun, input_params) -> 'VersionedUncachedData':
        version = generating_fun.__name__ + _hash_data([input_params + [inspect.getsource(generating_fun)]])
        return VersionedUncachedData(data, version)


@dataclass
class VersionedData:
    """
    Represents a versioned data object that is cached to disk by pickling it
    """

    """
    Path to this version of the cached file 
    """
    file_path: Path

    """
    Version string of this fixture, any two versioned fixtures with the same version string
    are considered equivalent (even if they are of different types)
    """
    version: str

    """
    Actual data of the fixture that is loaded from disk
    """
    data: any = None
    
    def save(self, data):

        self.data = data

        with open(self.file_path, 'wb') as f:
            pickle.dump(data, f)
            
    def load(self, request):

        if not self.file_path.exists() or request.config.getoption("--recompute-cache"):
            return False

        with open(self.file_path, 'rb') as f:
            self.data = pickle.load(f)

        return True

    @staticmethod
    def build(request, fixture_name: str, input_params) -> 'VersionedData':
        cached_fixture_dir = Path(request.config.cache.makedir("versioned_fixtures"))
        files_dir = Path(cached_fixture_dir) / fixture_name        
        files_dir.mkdir(exist_ok=True)

        # we namespace the version with the dir_name to make sure two
        # fixtures with the same input_params don't collide        
        version = fixture_name + "." +_hash_data(input_params)
        
        file_path = files_dir / f"{version}.pkl"

        return VersionedData(file_path, version)


def versioned_generated_file_fixture(suffix):
    """
    Decorator to create a fixture that generates a versioned file cached on disk.

    The decorated fixture must accept a 'request' parameter and a 'versioned_file' parameter.

    The 'versioned_file' parameter is a placeholder that will be replaced by the actual
    path to the generated file.

    The version of the generated file is computed based on the versions of the input
    parameters of the fixture.

    The decorated fixture must generate the file at the path specified by the
    'versioned_file' parameter.

    Example:
    @versioned_generated_file_fixture("mlir")
    def my_generated_mlir_file(request, versioned_file, some_input_param):
        # generate the mlir file and save it to versioned_file
        with open(versioned_file, "w") as f:
            f.write("...")
        

    All input parameters of the fixture (except request and versioned_file) must be other versioned
    fixtures (i.e. decorated with versioned_cached_data_fixture, versioned_hashable_object_fixture,
    versioned_unhashable_object_fixture, or versioned_generated_file_fixture).

    The decorated function must not use the request parameter to read any configuration
    options, as this would break the caching mechanism.
    """

    if not isinstance(suffix, str):
        raise ValueError("versioned_generated_file_fixture requires a suffix argument of type str that will be used as file extension for the generated file")

    def decorator(fun):

        # we cannot easily request this fixture here because either
        # pytest gets confused with the location of the fixture
        # either it gets confused on the requested fixtures
        if "request" not in inspect.signature(fun).parameters:
            raise ValueError("versioned_generated_file_fixture can only be applied to fixtures that accept a 'request' parameter")

        @pytest.fixture
        @wraps(fun)
        def wrapper(**kwargs):

            logger.debug("[generating versioned file] " + fun.__name__)

            input_params_versions, kwargs_data = _get_fixture_params(kwargs)

            request = kwargs['request']
                    
            versioned_file = VersionedFile.build(request, fun.__name__, suffix, input_params_versions)

            kwargs_data['versioned_file'] = versioned_file.file_path

            with FileLock(str(versioned_file.file_path) + ".lock"):

                if not versioned_file.valid(request):
                    if _fetch_remote(request, fun, versioned_file.version, versioned_file.file_path, False):
                        _record_cache_use(request, fun, "fetched")
                    else:
                        logger.debug("[cache miss] " + fun.__name__ + f" -> {versioned_file.file_path}")
                        fun(**kwargs_data)
                        _store_remote(fun, versioned_file.version, versioned_file.file_path, False)
                        _record_cache_use(request, fun, "computed")
                else:
                    logger.debug("[cache hit] " + fun.__name__ + f" -> {versioned_file.file_path}")
                    _record_cache_use(request, fun, "reused")

            logger.debug("[generated file version] " + fun.__name__ + f" -> {versioned_file.file_path}")

            return versioned_file

        return wrapper
    
    return decorator


def versioned_generated_directory_fixture(fun):
    """
    Decorator to create a fixture that generates a versioned directory cached on disk.

    The decorated fixture must accept a 'request' parameter and a 'versioned_dir' parameter.

    The 'versioned_dir' parameter is a placeholder that will be replaced by the actual
    path to the generated directory.

    The version of the generated directory is computed based on the versions of the input
    parameters of the fixture.

    The decorated fixture must generate files in the directory specified by the
    'versioned_dir' parameter.

    Example:
    @versioned_generated_directory_fixture("mlir")
    def my_generated_directory(request, versioned_dir, some_input_param):
        # generate the mlir file and save it to versioned_file
        with open(versioned_dir / "myfile", "w") as f:
            f.write("...")
        

    All input parameters of the fixture (except request and versioned_dir) must be other versioned
    fixtures (i.e. decorated with versioned_cached_data_fixture, versioned_hashable_object_fixture,
    versioned_unhashable_object_fixture, or versioned_generated_directory_fixture).

    The decorated function must not use the request parameter to read any configuration
    options, as this would break the caching mechanism.
    """

    # we cannot easily request this fixture here because either
    # pytest gets confused with the location of the fixture
    # either it gets confused on the requested fixtures
    if "request" not in inspect.signature(fun).parameters:
        raise ValueError("versioned_generated_directory_fixture can only be applied to fixtures that accept a 'request' parameter")

    @pytest.fixture
    @wraps(fun)
    def wrapper(**kwargs):

        logger.debug("[generating versioned directory] " + fun.__name__)
        input_params_versions, kwargs_data = _get_fixture_params(kwargs)

        request = kwargs['request']

        versioned_dir = VersionedDirectory.build(request, fun.__name__, input_params_versions)

        kwargs_data['versioned_dir'] = versioned_dir.dir_path

        with FileLock(str(versioned_dir.dir_path) + ".lock"):

            if not versioned_dir.valid(request):
                if _fetch_remote(request, fun, versioned_dir.version, versioned_dir.dir_path, True):
                    _record_cache_use(request, fun, "fetched")
                else:
                    logger.debug("[cache miss] " + fun.__name__ + f" -> {versioned_dir.dir_path}")

                    # left by an earlier fetch of the same version (e.g. with --recompute-cache)
                    (versioned_dir.dir_path / FROM_REMOTE_CACHE_MARKER).unlink(missing_ok=True)

                    fun(**kwargs_data)

                    _store_remote(fun, versioned_dir.version, versioned_dir.dir_path, True)
                    _record_cache_use(request, fun, "computed")

                # mark the directory as valid
                with open(versioned_dir.dir_path / "valid", "w") as f:
                    pass
            else:
                logger.debug("[cache hit] " + fun.__name__ + f" -> {versioned_dir.dir_path}")
                _record_cache_use(request, fun, "reused",
                                  from_store=(versioned_dir.dir_path / FROM_REMOTE_CACHE_MARKER).exists())

        logger.debug("[generated dir version] " + fun.__name__ + f" -> {versioned_dir.dir_path}")

        return versioned_dir

    return wrapper
    

def versioned_static_file_fixture(fun):
    """
    Decorator to create a fixture that represents a static versioned file.

    The decorated fixture must return the path to a static file on disk (either
    a string or a Path object).

    The version of the file is computed based on the hash of the file content.

    The version is cached based on the file's mtime to avoid recomputing the hash
    on every invocation.

    The fixture must accept a 'request' parameter.

    Example:
    @versioned_static_file_fixture
    def my_static_mlir_file(request):
        return "/path/to/static/file.mlir"

    """

    @pytest.fixture
    @wraps(fun)
    def wrapper(**kwargs):

        file_path = Path(fun(**kwargs))

        version = fun.__name__ + _hash_file(kwargs['request'], file_path)

        logger.debug("[static file] " + fun.__name__ + f" -> {file_path}")
        logger.debug("[static file version] " + fun.__name__ + f" -> {version}")

        return VersionedFile(file_path, version)

    return wrapper


def versioned_unhashable_object_fixture(fun):
    """
    Decorator to create a fixture that represents a versioned object that is not
    cached to disk.

    The version of the object is computed based on the input parameters of the fixture
    and the source code of the generating function.

    Example:
    @versioned_unhashable_object_fixture
    def my_unhashable_object(some_input_param):
        return SomeUnhashableObject(...)
    """


    @pytest.fixture
    @wraps(fun)
    def wrapper(**kwargs):        

        logger.debug("[uncached unhashable] " + fun.__name__)

        input_params_versions, kwargs_data = _get_fixture_params(kwargs)

        logger.debug("[unhashable object version] " + fun.__name__ + " -> " + str(input_params_versions))

        return VersionedUncachedData.build(fun(**kwargs_data), fun, input_params_versions)

    return wrapper


def versioned_hashable_object_fixture(fun):
    """
    Decorator to create a fixture that represents a versioned object that is not
    cached to disk.

    The version of the object is computed based on the hash of the object itself.

    Example:
    @versioned_hashable_object_fixture
    def my_hashable_object(some_input_param):
        return SomeHashableObject(...)
    """
        
    @pytest.fixture
    @wraps(fun)
    def wrapper(**kwargs):
        logger.debug("[uncached hashable] " + fun.__name__)        
        data = fun(**kwargs)
        version = fun.__name__ + _hash_data(data)
        logger.debug("[hashable version] " + fun.__name__ + " -> " + version)
        return VersionedUncachedData(data, version)

    return wrapper


def versioned_cached_data_fixture(fun):
    """
    Decorator to create a fixture that caches its result to disk.

    The decorated fixture must accept a 'request' parameter.

    The version of the cached data is computed based on the versions of the input
    parameters of the fixture.

    Example:
    @versioned_cached_data_fixture
    def my_cached_data(request, some_input_param):
        # compute the data
        return data

    All input parameters of the fixture (except request) must be other versioned
    fixtures.

    The decorated function must not use the request parameter to read any configuration
    options, as this would break the caching mechanism.
    """

    if "request" not in inspect.signature(fun).parameters:
        raise ValueError("versioned_cached_data_fixture can only be applied to fixtures that accept a 'request' parameter")

    @pytest.fixture
    @wraps(fun)
    def wrapper(**kwargs):

        logger.debug("[generating versioned data] " + fun.__name__)
        
        input_params_versions, kwargs_data = _get_fixture_params(kwargs)

        request = kwargs['request']

        versioned_data = VersionedData.build(request, fun.__name__, input_params_versions)

        with FileLock(str(versioned_data.file_path) + ".lock"):

            if versioned_data.load(request):
                logger.debug("[cache hit] " + fun.__name__ + f" -> {versioned_data.file_path}")
                _record_cache_use(request, fun, "reused")
            elif (_fetch_remote(request, fun, versioned_data.version, versioned_data.file_path, False)
                  and versioned_data.load(request)):
                _record_cache_use(request, fun, "fetched")
            else:
                logger.debug("[cache miss] " + fun.__name__ + f" -> {versioned_data.file_path}")
                versioned_data.save(fun(**kwargs_data))
                _store_remote(fun, versioned_data.version, versioned_data.file_path, False)
                _record_cache_use(request, fun, "computed")

        logger.debug("[data version] " + fun.__name__ + " -> " + versioned_data.version)

        return versioned_data

    return wrapper