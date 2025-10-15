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
    # checkout 6.12 kernel from synaptics git
    cd ${THIRDPARTY_PREBUILTS_SOC_KERNEL}
    git clone --depth=1 --branch wip_branch/vssdk/sl2619_bringup/202509161205 ssh://${SYNA_GERRIT_USER}@gerrit-ind.synaptics.com:29420/astra/linux/main
    cd main/
    # checkout specific commit to ensure repeatability
    git fetch --depth=1 origin 8674817448091cacab6dabb73fb367f2ab04b4f5
    git checkout 8674817448091cacab6dabb73fb367f2ab04b4f5
    # cherrypick NPU + IOMMU defconfig changes, remove and update above commit to that once merged
    git fetch --depth=1 origin d41b8de18eb8f25447217ca15871b4107addce45
    git cherry-pick d41b8de18eb8f25447217ca15871b4107addce45 --strategy-option theirs --no-commit

    cd drivers/
    git clone --depth=1 --branch wip_branch/vssdk/sl2619_bringup/202509161205 ssh://${SYNA_GERRIT_USER}@gerrit-ind.synaptics.com:29420/debu/common/linux-driver/synaptics
    cd synaptics/
    git fetch --depth=1 origin e00c0864e73e42c18d7fdb8af6df2bf627bdcb03
    git checkout e00c0864e73e42c18d7fdb8af6df2bf627bdcb03
    git fetch --depth=1 origin 052a607687a0a2a8b577f38288cdf717d32749b2
    git cherry-pick 052a607687a0a2a8b577f38288cdf717d32749b2 --strategy-option theirs --no-commit

    # configure kernel to build the kernel module against it and pack the required headers/Makefiles/scripts
    cd ${SYNA_KERNEL_CHECKOUT}
    make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- sl261x_defconfig
    export LOCALVERSION=""
    make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- modules_prepare
    tar -cvzf ${THIRDPARTY_PREBUILTS_SOC_KERNEL}/${SYNA_KERNEL_MODULE_BUILD_PKG} .config include/ arch/arm64/include/ scripts/ Makefile arch/arm64/Makefile
    rm -rf ${SYNA_KERNEL_CHECKOUT}
fi
