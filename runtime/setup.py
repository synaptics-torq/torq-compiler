# Copyright 2025 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Torq Runtime Python Package Setup.

This packages the Torq runtime alongside the IREE runtime Python bindings
into a single wheel with both namespaces:

    from torq.runtime import ...   # Torq-specific
    from iree.runtime import ...   # Existing IREE runtime

Cross-compile aarch64 build:

    TORQ_RUNTIME_CMAKE_BUILD_DIR=../iree-build-soc-python pip wheel \\
        --no-build-isolation --no-deps -w dist runtime/

Native host build:

    pip wheel --no-build-isolation --no-deps -w dist runtime/

Environment variables:
    TORQ_RUNTIME_CMAKE_BUILD_DIR  Path to iree-build-soc build directory.
                                  When set, cmake configure+build are skipped
                                  and only cmake --install is run.
    TORQ_WHEEL_PLAT               Force the wheel platform tag (e.g.
                                  manylinux2014_aarch64).  When unset, the tag
                                  is auto-detected from the ELF headers of the
                                  built .so files.
"""

import glob
import os
import shutil
import struct
import subprocess
import sys
import sysconfig
from datetime import datetime, timezone

from distutils.command.build import build as _build
from setuptools import Extension, find_namespace_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SETUPPY_DIR = os.path.realpath(os.path.dirname(__file__))
TORQ_SOURCE_DIR = os.path.join(SETUPPY_DIR, "..")
IREE_SOURCE_DIR = os.path.join(TORQ_SOURCE_DIR, "third_party", "iree")

# Where cmake --install stages the built artifacts (relative to setup.py).
CMAKE_INSTALL_DIR_REL = os.path.join("build", "install")
CMAKE_INSTALL_DIR_ABS = os.path.join(SETUPPY_DIR, CMAKE_INSTALL_DIR_REL)

# CMake build directory: either pre-existing or created from scratch.
PREBUILT_DIR = os.getenv("TORQ_RUNTIME_CMAKE_BUILD_DIR")
if PREBUILT_DIR:
    # Resolve relative paths against the project root (one level up from runtime/)
    if not os.path.isabs(PREBUILT_DIR):
        PREBUILT_DIR = os.path.normpath(os.path.join(TORQ_SOURCE_DIR, PREBUILT_DIR))
    CMAKE_BUILD_DIR = os.path.realpath(PREBUILT_DIR)
else:
    CMAKE_BUILD_DIR = os.path.join(SETUPPY_DIR, "build", "cmake")

TORQ_WHEEL_VERSION = os.getenv("TORQ_WHEEL_VERSION")
if TORQ_WHEEL_VERSION:
    # Release build: use explicit version
    VERSION = TORQ_WHEEL_VERSION
else:
    # Debug build (default)    
    VERSION = f"0.dev"

# ---------------------------------------------------------------------------
# Paths to pure-Python source packages in the source trees
# ---------------------------------------------------------------------------                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   

IREE_PYTHON_DIR = os.path.join(
    IREE_SOURCE_DIR, "runtime", "bindings", "python"
)
TORQ_PYTHON_DIR = os.path.join(SETUPPY_DIR, "bindings", "python")

# ---------------------------------------------------------------------------
# CMake build helpers
# ---------------------------------------------------------------------------


def cmake_configure_and_build():
    """Run cmake configure + build for a from-scratch native build."""
    subprocess.check_call(["cmake", "--version"])
    cfg = os.getenv("CMAKE_BUILD_TYPE", "Release")

    os.makedirs(CMAKE_BUILD_DIR, exist_ok=True)

    cmake_cache = os.path.join(CMAKE_BUILD_DIR, "CMakeCache.txt")
    if not os.path.exists(cmake_cache):
        cmake_args = [
            "-GNinja",
            "-DIREE_BUILD_PYTHON_BINDINGS=ON",
            "-DIREE_BUILD_COMPILER=OFF",
            "-DIREE_BUILD_SAMPLES=OFF",
            "-DIREE_BUILD_TESTS=OFF",
            "-DIREE_HAL_DRIVER_LOCAL_SYNC=ON",
            "-DIREE_HAL_DRIVER_LOCAL_TASK=ON",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]
        subprocess.check_call(
            ["cmake", TORQ_SOURCE_DIR] + cmake_args, cwd=CMAKE_BUILD_DIR
        )

    targets = [
        "iree_runtime_bindings_python_runtime",
        "torq_runtime_python_bindings",
    ]
    for target in targets:
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", target], cwd=CMAKE_BUILD_DIR
        )


def cmake_install():
    """Run cmake --install for both IREE and Torq Python components."""
    os.makedirs(CMAKE_INSTALL_DIR_ABS, exist_ok=True)

    cmake_install_script = os.path.join(CMAKE_BUILD_DIR, "cmake_install.cmake")
    for component in ["IreePythonPackage-runtime", "TorqPythonPackage-runtime"]:
        install_args = [
            "cmake",
            f"-DCMAKE_INSTALL_PREFIX={CMAKE_INSTALL_DIR_ABS}/",
            f"-DCMAKE_INSTALL_COMPONENT={component}",
            "-P",
            cmake_install_script,
        ]
        subprocess.check_call(install_args)


# ---------------------------------------------------------------------------
# Setuptools command overrides
# ---------------------------------------------------------------------------


class CMakeBuildPy(_build_py):
    def run(self):
        if not PREBUILT_DIR:
            cmake_configure_and_build()
        cmake_install()

        # Copy the cmake-installed _runtime_libs into the setuptools build tree
        # so the native .so and CLI tools end up in the wheel.
        target_dir = os.path.join(
            os.path.abspath(self.build_lib), "iree", "_runtime_libs"
        )
        source_dir = os.path.join(
            CMAKE_INSTALL_DIR_ABS,
            "python_packages",
            "iree_runtime",
            "iree",
            "_runtime_libs",
        )
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir, symlinks=False)

        # Copy the torq-run-module binary into torq/_runtime_libs.
        torq_libs_dir = os.path.join(
            os.path.abspath(self.build_lib), "torq", "_runtime_libs"
        )
        os.makedirs(torq_libs_dir, exist_ok=True)
        torq_rm_src = os.path.join(
            CMAKE_BUILD_DIR, "runtime", "tools", "torq-run-module"
        )
        if os.path.isfile(torq_rm_src):
            shutil.copy2(torq_rm_src, torq_libs_dir)

        # Let setuptools handle all the pure-Python packages.
        super().run()


class CustomBuild(_build):
    def run(self):
        self.run_command("build_py")
        self.run_command("build_ext")
        self.run_command("build_scripts")


class CMakeExtension(Extension):
    """A stub extension that prevents setuptools from trying to build it."""
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class NoopBuildExtension(_build_ext):
    """Prevents setuptools from trying to compile the CMake extension.

    The real native .so is provided by the cmake install step and included
    via package_data. The ext_modules declaration exists only so that
    setuptools produces a platform-specific wheel tag.
    """
    def build_extension(self, ext):
        pass


# ---------------------------------------------------------------------------
# Cross-compilation: wheel platform tag detection
# ---------------------------------------------------------------------------

# ELF e_machine → platform arch component
_ELF_MACHINE_TO_ARCH = {
    0x03: "i686",
    0x28: "armv7l",
    0x3E: "x86_64",
    0xB7: "aarch64",
}

WHEEL_PLAT = os.getenv("TORQ_WHEEL_PLAT")


def _detect_platform_from_so(search_dir):
    """Detect the target platform by reading ELF headers of built .so files."""
    so_files = glob.glob(os.path.join(search_dir, "*.so"))
    if not so_files:
        return None
    with open(so_files[0], "rb") as f:
        header = f.read(20)
    if len(header) < 20 or header[:4] != b"\x7fELF":
        return None
    ei_data = header[5]  # 1 = little-endian, 2 = big-endian
    byte_order = "<" if ei_data == 1 else ">"
    e_machine = struct.unpack(byte_order + "H", header[18:20])[0]
    arch = _ELF_MACHINE_TO_ARCH.get(e_machine)
    if arch is None:
        return None
    return f"linux_{arch}"


_RUNTIME_LIBS_DIR = os.path.join(
    CMAKE_INSTALL_DIR_ABS,
    "python_packages", "iree_runtime", "iree", "_runtime_libs",
)


class PlatOverrideBdistWheel(_bdist_wheel):
    """Use the correct platform tag when cross-compiling.

    Priority: TORQ_WHEEL_PLAT env var > ELF header detection > default.
    """
    def get_tag(self):
        impl, abi, plat = super().get_tag()
        if WHEEL_PLAT:
            plat = WHEEL_PLAT
        else:
            detected = _detect_platform_from_so(_RUNTIME_LIBS_DIR)
            if detected and detected != plat:
                plat = detected
        return impl, abi, plat


# ---------------------------------------------------------------------------
# Ensure built-package directories exist before setup() introspects them
# ---------------------------------------------------------------------------


def populate_built_package(abs_dir):
    os.makedirs(abs_dir, exist_ok=True)
    init_py = os.path.join(abs_dir, "__init__.py")
    if not os.path.exists(init_py):
        with open(init_py, "wt") as f:
            pass


populate_built_package(
    os.path.join(
        CMAKE_INSTALL_DIR_ABS,
        "python_packages",
        "iree_runtime",
        "iree",
        "_runtime_libs",
    )
)

# ---------------------------------------------------------------------------
# Package discovery
# ---------------------------------------------------------------------------

packages = (
    find_namespace_packages(
        where=IREE_PYTHON_DIR,
        include=["iree.runtime", "iree.runtime.*",
                 "iree._runtime", "iree._runtime.*"],
    )
    + find_namespace_packages(
        where=TORQ_PYTHON_DIR,
        include=["torq", "torq.*"],
    )
    + ["iree._runtime_libs"]
    + ["torq._runtime_libs"]
)

# ---------------------------------------------------------------------------
# setup()
# ---------------------------------------------------------------------------

setup(
    name="torq-runtime",
    version=VERSION,
    author="synaptics-astra",
    author_email="zSvcastrateam@synaptics.com",
    description="Torq runtime python bindings",
    long_description=open(
        os.path.join(IREE_PYTHON_DIR, "iree", "runtime", "README.md"), "rt"
    ).read(),
    long_description_content_type="text/markdown",
    license="Apache-2.0 WITH LLVM-exception",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
    ],
    url="https://github.com/synaptics-torq/torq-compiler",
    python_requires=">=3.12",
    ext_modules=[
        CMakeExtension("iree._runtime_libs._runtime"),
    ],
    cmdclass={
        "build": CustomBuild,
        "bdist_wheel": PlatOverrideBdistWheel,
        "build_ext": NoopBuildExtension,
        "build_py": CMakeBuildPy,
    },
    zip_safe=False,
    package_dir={
        # Pure Python from IREE source tree
        "iree.runtime": os.path.join(IREE_PYTHON_DIR, "iree", "runtime"),
        "iree._runtime": os.path.join(IREE_PYTHON_DIR, "iree", "_runtime"),
        # Built artifacts from cmake install
        "iree._runtime_libs": os.path.join(
            CMAKE_INSTALL_DIR_REL,
            "python_packages",
            "iree_runtime",
            "iree",
            "_runtime_libs",
        ),
        # Torq pure Python from source tree
        "torq": os.path.join(TORQ_PYTHON_DIR, "torq"),
        "torq.runtime": os.path.join(TORQ_PYTHON_DIR, "torq", "runtime"),
        # Torq native binaries
        "torq._runtime_libs": os.path.join(TORQ_PYTHON_DIR, "torq", "_runtime_libs"),
    },
    packages=packages,
    package_data={
        "iree._runtime_libs": [
            f"*{sysconfig.get_config_var('EXT_SUFFIX')}",
            "iree-run-module*",
            "iree-benchmark-executable*",
            "iree-benchmark-module*",
            "iree-create-parameters*",
            "iree-convert-parameters*",
            "iree-dump-module*",
            "iree-dump-parameters*",
            "iree-cpuinfo*",
        ],
        "torq._runtime_libs": [
            "torq-run-module*",
        ],
        "iree.runtime": ["*.pyi"],
    },
    entry_points={
        "console_scripts": [
            "iree-run-module = iree._runtime.scripts.iree_run_module.__main__:main",
            "iree-benchmark-executable = iree._runtime.scripts.iree_benchmark_executable.__main__:main",
            "iree-benchmark-module = iree._runtime.scripts.iree_benchmark_module.__main__:main",
            "iree-create-parameters = iree._runtime.scripts.iree_create_parameters.__main__:main",
            "iree-convert-parameters = iree._runtime.scripts.iree_convert_parameters.__main__:main",
            "iree-dump-module = iree._runtime.scripts.iree_dump_module.__main__:main",
            "iree-dump-parameters = iree._runtime.scripts.iree_dump_parameters.__main__:main",
            "iree-cpuinfo = iree._runtime.scripts.iree_cpuinfo.__main__:main",
            "torq-run-module = torq.runtime.scripts.torq_run_module.__main__:main",
        ],
    },
    install_requires=[
        "numpy<2.0.0",
        "PyYAML>=5.4.1",
        "ml-dtypes>=0.4.0",
    ],
)
