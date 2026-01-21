#!/bin/bash

# This script installs all system dependencies for Ubuntu 24.04

# abort on error
set -e

apt-get update
apt-get install -y cmake ninja-build clang lld llvm wget curl git git-lfs clang-format-18 \
                         python3-pip python3-dev python3-venv python3-setuptools \
                         qemu-system-misc \
                         libc6-dev-armhf-cross libgcc-s1-armhf-cross libgcc-13-dev-armhf-cross libstdc++-13-dev-armhf-cross binutils-arm-linux-gnueabihf \
                         libstdc++-13-dev-arm64-cross \
                         gcc-aarch64-linux-gnu flex bison bc \
                         graphviz imagemagick # these are for building the documentation

# Install Synaptics Astra Toolchain if not present
TOOLCHAIN_DIR="/opt/synaptics/astra/toolchain"
FILE="sl2619_scarthgap-poky-glibc-x86_64-astra-media-cortexa55-sl2619-toolchain-5.0.9.sh"
HASH="ba0cfb28c890a62db786ce8559347e91"
URL="https://github.com/synaptics-astra/sdk/releases/download/scarthgap_6.12_v2.1.0"

if [ ! -d "$TOOLCHAIN_DIR" ]; then
    echo "Downloading Astra toolchain..."
    wget -c "$URL/$FILE.000" "$URL/$FILE.001"

    echo "Combining and installing..."
    cat "${FILE}."* > "$FILE" && rm -f "${FILE}."*

    if echo "$HASH $FILE" | md5sum -c -; then
        chmod +x "$FILE"
        ./"$FILE" -y -d "$TOOLCHAIN_DIR" && rm -f "$FILE"
        echo "Toolchain installed to $TOOLCHAIN_DIR"
    else
        echo "MD5 Verification Failed!"
        rm -f "$FILE"
        exit 1
    fi
else
    echo "Toolchain already installed at $TOOLCHAIN_DIR"
fi
