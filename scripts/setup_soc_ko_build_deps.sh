#!/bin/bash

# This script checks out the Synaptics kernel and driver, prepares for module build, and packages required artifacts to build the NPU kernel module.
# By default, it uses public GitHub repositories. Pass --use-gerrit to use Synaptics Gerrit instead.

set -e

BASE_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")/..
THIRDPARTY_PREBUILTS_SOC_KERNEL=${BASE_DIR}/third_party/runtimes-prebuilt/soc-kernel/
SYNA_KERNEL_MODULE_BUILD_PKG=syna-kernel-artifacts.tgz

# Parse options
USE_GERRIT=0
PACKAGE_PATH="${THIRDPARTY_PREBUILTS_SOC_KERNEL}/${SYNA_KERNEL_MODULE_BUILD_PKG}"
while [[ $# -gt 0 ]]; do
  case $1 in
    --use-gerrit)
      USE_GERRIT=1
      shift
      ;;
    --package-path)
      PACKAGE_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ $USE_GERRIT -eq 1 ]]; then
  SYNA_GERRIT_USER=$(id -n -u)
  LINUX_REPO_URL="ssh://${SYNA_GERRIT_USER}@sc-debu-git.synaptics.com:29420/astra/linux/main"
  LINUX_DRIVER_REPO_URL="ssh://${SYNA_GERRIT_USER}@sc-debu-git.synaptics.com:29420/debu/common/linux-driver/synaptics"
  KERNEL_BRANCH="dev_branch/linux/v6_12/master"
  DRIVER_BRANCH="dev_branch/master"
else
  LINUX_REPO_URL="https://github.com/synaptics-astra/linux_6_12-main"
  LINUX_DRIVER_REPO_URL="https://github.com/synaptics-astra/linux_6_12-drivers-synaptics"
  KERNEL_BRANCH="scarthgap_6.12_v2.3.0"
  DRIVER_BRANCH="scarthgap_6.12_v2.3.0"
fi

SYNA_KERNEL_CHECKOUT=${THIRDPARTY_PREBUILTS_SOC_KERNEL}/main

echo "Configuring to build syna-kernel (source: $([[ $USE_GERRIT -eq 1 ]] && echo 'Gerrit' || echo 'GitHub'))"

if [ -f "${PACKAGE_PATH}" ]; then
    echo "kernel artifacts already packed. To rebuild, remove ${PACKAGE_PATH}"
else
    mkdir -p ${THIRDPARTY_PREBUILTS_SOC_KERNEL}
    cd ${THIRDPARTY_PREBUILTS_SOC_KERNEL}

    # Remove existing main directory if present to avoid git clone errors
    if [ -d "main" ]; then
      rm -rf main
    fi
    git clone --depth=1 --branch ${KERNEL_BRANCH} ${LINUX_REPO_URL} main
    cd main/drivers/
    git clone --depth=1 --branch ${DRIVER_BRANCH} ${LINUX_DRIVER_REPO_URL} synaptics

    # configure kernel to build the kernel module against it and pack the required headers/Makefiles/scripts
    cd ${SYNA_KERNEL_CHECKOUT}
    make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- sl261x_defconfig
    export LOCALVERSION=""
    make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- modules_prepare
    tar -cvzf ${PACKAGE_PATH} .config include/ arch/arm64/include/ scripts/ Makefile arch/arm64/Makefile
    rm -rf ${SYNA_KERNEL_CHECKOUT}
fi
