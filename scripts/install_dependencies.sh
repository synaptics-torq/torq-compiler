#!/bin/bash

# This script installs all system dependencies for Ubuntu 24.04

# abort on error
set -e

apt-get update

# vim, ccache, gdb, ping: for development convenience
# file: is used by the toolchain extraction script
# adb can be useful to connect to the astra board from inside the docker
# graphviz imagemagick : these are for building the documentation
apt-get install -y cmake ninja-build clang lld llvm wget curl git git-lfs clang-format-18 \
                         python3-pip python3-dev python3-venv python3-setuptools \
                         qemu-system-misc \
                         libc6-dev-armhf-cross libgcc-s1-armhf-cross libgcc-13-dev-armhf-cross libstdc++-13-dev-armhf-cross binutils-arm-linux-gnueabihf \
                         libstdc++-13-dev-arm64-cross \
                         gcc-aarch64-linux-gnu flex bison bc \
                         file ccache vim adb gdb iputils-ping \
                         graphviz imagemagick
