#!/bin/bash

# This script installs all system dependencies for Ubuntu 24.04

# abort on error
set -e

apt-get update

apt-get install -y cmake ninja-build clang lld wget curl git git-lfs clang-format-18 \
                         python3-pip python3-dev python3-venv python3-setuptools \
                         binutils-arm-linux-gnueabihf \
                         gcc-aarch64-linux-gnu flex bison bc \
                         file ccache iputils-ping
