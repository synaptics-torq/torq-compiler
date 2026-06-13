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

import os
import shutil
import sys
import sysconfig

from setuptools import find_namespace_packages, setup
from setuptools.command.build_py import build_py as _build_py
from torq.build_tools.setup_helpers import (
    CMakeExtension,
    CustomBuild,
    NoopBuildExtension,
    cmake_build_targets,
    cmake_configure_if_needed,
    cmake_install_script as run_cmake_install_script,
    make_platform_bdist_wheel,
    populate_built_package,
    resolve_cmake_build_dir,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SETUPPY_DIR = os.path.realpath(os.path.dirname(__file__))
TORQ_SOURCE_DIR = os.path.join(SETUPPY_DIR, "..")
IREE_SOURCE_DIR = os.path.join(TORQ_SOURCE_DIR, "third_party", "iree")

# Where cmake --install stages the built artifacts (relative to setup.py).
CMAKE_INSTALL_DIR_REL = os.path.join("build", "install")
CMAKE_INSTALL_DIR_ABS = os.path.join(SETUPPY_DIR, CMAKE_INSTALL_DIR_REL)

PREBUILT_DIR, CMAKE_BUILD_DIR = resolve_cmake_build_dir(
    SETUPPY_DIR, TORQ_SOURCE_DIR, "TORQ_RUNTIME_CMAKE_BUILD_DIR"
)

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
    cfg = os.getenv("CMAKE_BUILD_TYPE", "Release")
    cmake_args = [
        "-GNinja",
        "-DTORQ_MPACT_SIMULATOR_LIB=OFF", # TODO: we need to enable this and package the MPACT lib
        "-DIREE_BUILD_PYTHON_BINDINGS=ON",
        "-DIREE_BUILD_COMPILER=OFF",
        "-DIREE_BUILD_SAMPLES=OFF",
        "-DIREE_BUILD_TESTS=OFF",
        "-DIREE_HAL_DRIVER_LOCAL_SYNC=ON",
        "-DIREE_HAL_DRIVER_LOCAL_TASK=ON",
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-DCMAKE_BUILD_TYPE={cfg}",
    ]
    cmake_configure_if_needed(TORQ_SOURCE_DIR, CMAKE_BUILD_DIR, cmake_args)
    cmake_build_targets(
        CMAKE_BUILD_DIR,
        [
            "iree_runtime_bindings_python_runtime",
            "torq_runtime_python_bindings",
        ],
    )


def cmake_install():
    """Run cmake --install for both IREE and Torq Python components."""
    if os.path.exists(CMAKE_INSTALL_DIR_ABS):
        shutil.rmtree(CMAKE_INSTALL_DIR_ABS)
    os.makedirs(CMAKE_INSTALL_DIR_ABS, exist_ok=True)

    cmake_install_script = os.path.join(CMAKE_BUILD_DIR, "cmake_install.cmake")
    for component in ["IreePythonPackage-runtime", "TorqPythonPackage-runtime"]:
        run_cmake_install_script(
            cmake_install_script,
            CMAKE_INSTALL_DIR_ABS,
            component=component,
        )


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


# ---------------------------------------------------------------------------
# Cross-compilation: wheel platform tag detection
# ---------------------------------------------------------------------------

# minimum glibc version for the manylinux platform tag
# we should be using GLIBC 2.39 but there are currently
# no stable x86 and aarch64 manylinux releases.
_MANYLINUX_GLIBC = "2_28"

_RUNTIME_LIBS_DIR = os.path.join(
    CMAKE_INSTALL_DIR_ABS,
    "python_packages",
    "iree_runtime",
    "iree",
    "_runtime_libs",
)


PlatOverrideBdistWheel = make_platform_bdist_wheel(
    _RUNTIME_LIBS_DIR,
    glibc_version=_MANYLINUX_GLIBC,
    recursive=False,
)


# ---------------------------------------------------------------------------
# Ensure built-package directories exist before setup() introspects them
# ---------------------------------------------------------------------------

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
        "numpy>2.0.0b1",
        "PyYAML>=5.4.1",
        "ml-dtypes>=0.4.0",
    ],
)
