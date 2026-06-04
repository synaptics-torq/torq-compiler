# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Torq Compiler Python Package Setup.

This packages the Torq compiler alongside the IREE compiler Python bindings
into a single wheel with both namespaces:

    from torq.compiler import ...   # Torq-specific
    from iree.compiler import ...   # Existing IREE compiler

Native host build:

    pip wheel --no-build-isolation --no-deps -w dist compiler/

Pre-built (skip cmake configure+build, just install+package):

    TORQ_COMPILER_CMAKE_BUILD_DIR=../iree-build pip wheel \\
        --no-build-isolation --no-deps -w dist compiler/

Environment variables:
    TORQ_COMPILER_CMAKE_BUILD_DIR  Path to iree-build directory.
                                   When set, cmake configure+build are skipped
                                   and only cmake --install is run.
    TORQ_WHEEL_PLAT                Force the wheel platform tag (e.g.
                                   manylinux_2_28_x86_64).
    TORQ_WHEEL_VERSION             Wheel version string.
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
    strip_all_in_dir,
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
    SETUPPY_DIR, TORQ_SOURCE_DIR, "TORQ_COMPILER_CMAKE_BUILD_DIR"
)

TORQ_WHEEL_VERSION = os.getenv("TORQ_WHEEL_VERSION")
if TORQ_WHEEL_VERSION:
    VERSION = TORQ_WHEEL_VERSION
else:
    VERSION = "0.dev"

# ---------------------------------------------------------------------------
# Paths to pure-Python source packages in the source trees
# ---------------------------------------------------------------------------

IREE_COMPILER_PYTHON_DIR = os.path.join(
    IREE_SOURCE_DIR, "compiler", "bindings", "python"
)
TORQ_COMPILER_PYTHON_DIR = os.path.join(SETUPPY_DIR, "bindings", "python")

# iree.tools.tf wrappers live in the IREE submodule (for SavedModel import)
IREE_TF_INTEGRATIONS_DIR = os.path.join(
    IREE_SOURCE_DIR, "integrations", "tensorflow", "python_projects"
)
IREE_TF_PYTHON_DIR = os.path.join(IREE_TF_INTEGRATIONS_DIR, "iree_tf")

# ---------------------------------------------------------------------------
# CMake build helpers
# ---------------------------------------------------------------------------


def cmake_configure_and_build():
    """Run cmake configure + build for a from-scratch native build."""
    cfg = os.getenv("CMAKE_BUILD_TYPE", "Release")
    cmake_args = [
        "-GNinja",
        "-DIREE_BUILD_PYTHON_BINDINGS=ON",
        "-DIREE_BUILD_SAMPLES=OFF",
        "-DIREE_BUILD_TESTS=OFF",
        "-DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON",
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-DCMAKE_BUILD_TYPE={cfg}",
    ]
    cmake_configure_if_needed(TORQ_SOURCE_DIR, CMAKE_BUILD_DIR, cmake_args)
    cmake_build_targets(
        CMAKE_BUILD_DIR,
        [
            "torq-compile",
            "torq_compiler_python_bindings",
        ],
    )


def cmake_install():
    """Run cmake --install to stage both IREE and Torq compiler packages."""
    if os.path.exists(CMAKE_INSTALL_DIR_ABS):
        shutil.rmtree(CMAKE_INSTALL_DIR_ABS)
    os.makedirs(CMAKE_INSTALL_DIR_ABS, exist_ok=True)

    cmake_install_script = os.path.join(
        CMAKE_BUILD_DIR, "third_party", "iree", "compiler",
        "bindings", "python", "cmake_install.cmake"
    )

    # Install the IREE compiler Python modules (no component = installs all)
    run_cmake_install_script(
        cmake_install_script,
        CMAKE_INSTALL_DIR_ABS,
        strip=True,
    )

    # Install the Torq compiler Python package
    torq_install_script = os.path.join(
        CMAKE_BUILD_DIR, "third_party", "iree", "compiler", "plugins",
        "target", "TORQ", "bindings", "python", "cmake_install.cmake"
    )
    run_cmake_install_script(
        torq_install_script,
        CMAKE_INSTALL_DIR_ABS,
        component="TorqPythonPackage-compiler",
        strip=True,
    )


# ---------------------------------------------------------------------------
# Setuptools command overrides
# ---------------------------------------------------------------------------


class CMakeBuildPy(_build_py):
    def run(self):
        if not PREBUILT_DIR:
            cmake_configure_and_build()
        cmake_install()

        # Copy the entire cmake install tree into the setuptools build tree.
        # This is the approach used by IREE's compiler setup.py.
        target_dir = os.path.abspath(self.build_lib)
        source_dir = os.path.join(
            CMAKE_INSTALL_DIR_ABS, "python_packages", "iree_compiler"
        )
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir, symlinks=False)

        # Strip all ELF binaries (.so files and executables) in the tree.
        strip_all_in_dir(target_dir)

        # Also place a copy of torq-compile in torq/_compiler_libs/ for
        # discovery by binaries.py as a fallback.
        mlir_libs_dir = os.path.join(target_dir, "iree", "compiler", "_mlir_libs")
        torq_compile_src = os.path.join(mlir_libs_dir, "torq-compile")
        if os.path.isfile(torq_compile_src):
            torq_libs_dir = os.path.join(target_dir, "torq", "_compiler_libs")
            os.makedirs(torq_libs_dir, exist_ok=True)
            shutil.copy2(torq_compile_src, os.path.join(torq_libs_dir, "torq-compile"))

        # Copy iree.tools.tf wrappers into the build tree (for SavedModel import).
        iree_tf_src = os.path.join(IREE_TF_PYTHON_DIR, "iree", "tools", "tf")
        iree_tools_dst = os.path.join(target_dir, "iree", "tools")
        os.makedirs(iree_tools_dst, exist_ok=True)
        if os.path.isdir(iree_tf_src):
            shutil.copytree(iree_tf_src, os.path.join(iree_tools_dst, "tf"))


_MANYLINUX_GLIBC = "2_28"

PlatOverrideBdistWheel = make_platform_bdist_wheel(
    CMAKE_INSTALL_DIR_ABS,
    glibc_version=_MANYLINUX_GLIBC,
)


# ---------------------------------------------------------------------------
# Ensure built-package directories exist before setup() introspects them
# ---------------------------------------------------------------------------

populate_built_package(
    os.path.join(
        CMAKE_INSTALL_DIR_ABS,
        "python_packages",
        "iree_compiler",
        "iree",
        "compiler",
        "_mlir_libs",
    )
)

# ---------------------------------------------------------------------------
# Package discovery
# ---------------------------------------------------------------------------

_INSTALL_PACKAGES_DIR = os.path.join(
    CMAKE_INSTALL_DIR_ABS, "python_packages", "iree_compiler"
)

iree_packages = find_namespace_packages(
    where=_INSTALL_PACKAGES_DIR,
    include=[
        "iree.*",
    ],
)

torq_packages = find_namespace_packages(
    where=TORQ_COMPILER_PYTHON_DIR,
    include=[
        "torq",
        "torq.*",
    ],
)

# iree.tools.tf wrappers
iree_tools_packages = find_namespace_packages(
    where=IREE_TF_PYTHON_DIR,
    include=["iree.tools.tf*"],
)

packages = iree_packages + torq_packages + iree_tools_packages

# ---------------------------------------------------------------------------
# setup()
# ---------------------------------------------------------------------------

setup(
    name="torq-compiler",
    version=VERSION,
    author="synaptics-astra",
    author_email="zSvcastrateam@synaptics.com",
    description="Torq compiler python bindings",
    long_description="Torq compiler with IREE compiler Python bindings.",
    long_description_content_type="text/plain",
    license="Apache-2.0 WITH LLVM-exception",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
    ],
    url="https://github.com/synaptics-torq/torq-compiler",
    python_requires=">=3.12",
    ext_modules=[
        CMakeExtension("iree.compiler._mlir_libs._mlir"),
    ],
    cmdclass={
        "build": CustomBuild,
        "bdist_wheel": PlatOverrideBdistWheel,
        "build_ext": NoopBuildExtension,
        "build_py": CMakeBuildPy,
    },
    zip_safe=False,
    package_dir={
        # All IREE and Torq packages are found relative to this root.
        # The cmake install puts everything under python_packages/iree_compiler/
        "": os.path.join(CMAKE_INSTALL_DIR_REL, "python_packages", "iree_compiler"),
        # Torq sources are discovered directly so metadata generation does not
        # depend on stale cmake install contents.
        "torq": os.path.join("bindings", "python", "torq"),
        # iree.tools.tf from the IREE submodule
        "iree.tools.tf": os.path.join(IREE_TF_PYTHON_DIR, "iree", "tools", "tf"),
    },
    packages=packages,
    package_data={
        "iree.compiler._mlir_libs": [
            f"*{sysconfig.get_config_var('EXT_SUFFIX')}",
            "*.so*",
            "iree-compile*",
            "iree-opt*",
            "iree-link*",
            "iree-lld*",
            "torq-compile*",
        ],
        "torq._compiler_libs": [
            "torq-compile*",
        ],
    },
    entry_points={
        "console_scripts": [
            "torq-compile = torq.compiler.tools.binaries:main",
            "torq-gen-config = torq.gen_config.cli:main",
            "iree-compile = iree.compiler.tools.scripts.iree_compile.__main__:main",
            "iree-opt = iree.compiler.tools.scripts.iree_opt.__main__:main",
            "iree-import-tf = iree.tools.tf.scripts.iree_import_tf.__main__:main [tf]",
        ],
    },
    # IMPORTANT: must be synced with third_party/iree/compiler/setup.py
    install_requires=[
        "numpy",
        "sympy",
    ],
    # IMPORTANT: dependencies must be synced with ./requirements.txt
    extras_require={
        "onnx": [
            "onnx==1.19.1",
        ],
        "tflite": [
            "tosa-converter-for-tflite==2026.2.0",
        ],
        "tf": [
            "tensorflow==2.18.1",
        ],
    },
)
