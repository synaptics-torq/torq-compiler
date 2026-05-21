# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared setup.py helpers for Torq Python packages."""

import glob
import os
import shutil
import struct
import subprocess
import sys

from distutils.command.build import build as _build
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


_ELF_MACHINE_TO_ARCH = {
    0x03: "i686",
    0x28: "armv7l",
    0x3E: "x86_64",
    0xB7: "aarch64",
}


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
    """Prevents setuptools from trying to compile the CMake extension."""

    def build_extension(self, ext):
        pass


def resolve_cmake_build_dir(setuppy_dir, source_dir, env_var):
    """Resolve the optional prebuilt CMake directory for a setup.py."""
    prebuilt_dir = os.getenv(env_var)
    if prebuilt_dir:
        if not os.path.isabs(prebuilt_dir):
            prebuilt_dir = os.path.normpath(os.path.join(source_dir, prebuilt_dir))
        return os.path.realpath(prebuilt_dir), os.path.realpath(prebuilt_dir)
    return None, os.path.join(setuppy_dir, "build", "cmake")


def cmake_configure_if_needed(source_dir, build_dir, cmake_args):
    """Run cmake configure when the build directory has no cache yet."""
    subprocess.check_call(["cmake", "--version"])
    os.makedirs(build_dir, exist_ok=True)

    cmake_cache = os.path.join(build_dir, "CMakeCache.txt")
    if not os.path.exists(cmake_cache):
        subprocess.check_call(["cmake", source_dir] + list(cmake_args), cwd=build_dir)


def cmake_build_targets(build_dir, targets):
    """Build a sequence of CMake targets."""
    for target in targets:
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", target], cwd=build_dir
        )


def cmake_install_script(install_script, install_prefix, component=None, strip=False):
    """Run a generated cmake_install.cmake script."""
    install_args = [
        "cmake",
        f"-DCMAKE_INSTALL_PREFIX={install_prefix}/",
    ]
    if component:
        install_args.append(f"-DCMAKE_INSTALL_COMPONENT={component}")
    if strip:
        install_args.append("-DCMAKE_INSTALL_DO_STRIP=ON")
    install_args.extend(["-P", install_script])
    subprocess.check_call(install_args)


def strip_binary(path):
    """Strip debug symbols from an ELF binary using the system strip tool."""
    if not os.path.isfile(path):
        return
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic != b"\x7fELF":
        return
    strip_cmd = shutil.which("strip")
    if strip_cmd:
        print(f"Stripping: {path}", file=sys.stderr)
        subprocess.check_call([strip_cmd, "--strip-unneeded", path])


def strip_all_in_dir(directory):
    """Walk a directory and strip all ELF binaries (.so files and executables)."""
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            fpath = os.path.join(root, fname)
            if fname.endswith(".so") or os.access(fpath, os.X_OK):
                strip_binary(fpath)


def linux_to_manylinux(plat, glibc_version="2_28"):
    """Convert a linux_* platform tag to a PEP-compatible manylinux tag."""
    if plat.startswith("linux_"):
        arch = plat[len("linux_"):]
        return f"manylinux_{glibc_version}_{arch}"
    return plat


def detect_platform_from_so(search_dir, glibc_version="2_28", recursive=True):
    """Detect the target platform by reading ELF headers of built .so files."""
    pattern = "**/*.so" if recursive else "*.so"
    so_files = glob.glob(os.path.join(search_dir, pattern), recursive=recursive)
    if not so_files:
        return None
    with open(so_files[0], "rb") as f:
        header = f.read(20)
    if len(header) < 20 or header[:4] != b"\x7fELF":
        return None
    ei_data = header[5]
    byte_order = "<" if ei_data == 1 else ">"
    e_machine = struct.unpack(byte_order + "H", header[18:20])[0]
    arch = _ELF_MACHINE_TO_ARCH.get(e_machine)
    if arch is None:
        return None
    return f"manylinux_{glibc_version}_{arch}"


def make_platform_bdist_wheel(
    search_dir,
    glibc_version,
    wheel_plat_env="TORQ_WHEEL_PLAT",
    recursive=True,
):
    """Create a bdist_wheel class with Torq platform tag override behavior."""

    class PlatOverrideBdistWheel(_bdist_wheel):
        """Use the correct platform tag for the wheel."""

        def get_tag(self):
            impl, abi, plat = super().get_tag()
            wheel_plat = os.getenv(wheel_plat_env)
            if wheel_plat:
                plat = wheel_plat
            else:
                detected = detect_platform_from_so(
                    search_dir, glibc_version=glibc_version, recursive=recursive
                )
                if detected:
                    plat = detected
                else:
                    plat = linux_to_manylinux(plat, glibc_version=glibc_version)
            return impl, abi, plat

    return PlatOverrideBdistWheel


def populate_built_package(abs_dir):
    """Ensure a built package directory exists before setup() introspects it."""
    os.makedirs(abs_dir, exist_ok=True)
    init_py = os.path.join(abs_dir, "__init__.py")
    if not os.path.exists(init_py):
        with open(init_py, "wt") as f:
            pass
