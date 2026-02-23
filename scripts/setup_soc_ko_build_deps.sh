#!/bin/bash

# This script checkout syna-kernel, prepare for module build, package required artifacts to build kernel module for npu

set -e

BASE_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")/..
THIRDPARTY_PREBUILTS_SOC_KERNEL=${BASE_DIR}/third_party/runtimes-prebuilt/soc-kernel/
SYNA_KERNEL_CHECKOUT=${THIRDPARTY_PREBUILTS_SOC_KERNEL}/main
SYNA_KERNEL_MODULE_BUILD_PKG=syna-kernel-artifacts.tgz
SYNA_GERRIT_USER=$(id -n -u)

echo "Configuring to build syna-kernel for ${SYNA_GERRIT_USER}"

if [ -f "${THIRDPARTY_PREBUILTS_SOC_KERNEL}/${SYNA_KERNEL_MODULE_BUILD_PKG}" ]; then
    echo "kernel artifacts already packed. To rebuild, remove ${THIRDPARTY_PREBUILTS_SOC_KERNEL}/${SYNA_KERNEL_MODULE_BUILD_PKG}"
else
    mkdir -p ${THIRDPARTY_PREBUILTS_SOC_KERNEL}
    # checkout 6.12.62 kernel from synaptics git
    # kernel commit: f10b8894fa493a1e66b36fb59e2b976798b72f2a (synced from vssdk dev_branch/linux/v6_12/master 2026-02-23)
    # driver commit: 02b2dc92cf2083ab12b1783c8503f20679808bc2 (synced from vssdk dev_branch/master 2026-02-23)
    cd ${THIRDPARTY_PREBUILTS_SOC_KERNEL}
    git clone --depth=1 --branch dev_branch/linux/v6_12/master ssh://${SYNA_GERRIT_USER}@sc-debu-git.synaptics.com:29420/astra/linux/main
    cd main/
    # checkout specific commit to ensure repeatability
    git fetch --depth=1 origin f10b8894fa493a1e66b36fb59e2b976798b72f2a
    git checkout f10b8894fa493a1e66b36fb59e2b976798b72f2a

    cd drivers/
    git clone --depth=1 --branch dev_branch/master ssh://${SYNA_GERRIT_USER}@sc-debu-git.synaptics.com:29420/debu/common/linux-driver/synaptics
    cd synaptics/
    git fetch --depth=1 origin 02b2dc92cf2083ab12b1783c8503f20679808bc2
    git checkout 02b2dc92cf2083ab12b1783c8503f20679808bc2

    # configure kernel to build the kernel module against it and pack the required headers/Makefiles/scripts
    cd ${SYNA_KERNEL_CHECKOUT}
    make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- sl261x_defconfig
    export LOCALVERSION=""
    make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- modules_prepare
    tar -cvzf ${THIRDPARTY_PREBUILTS_SOC_KERNEL}/${SYNA_KERNEL_MODULE_BUILD_PKG} .config include/ arch/arm64/include/ scripts/ Makefile arch/arm64/Makefile
    rm -rf ${SYNA_KERNEL_CHECKOUT}
fi
