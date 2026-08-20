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
    --help)
      echo "Usage: $0 [--use-gerrit] [--package-path <path>]"
      echo "  --use-gerrit: Use Synaptics Gerrit for source checkout instead of public GitHub repositories."
      echo "  --package-path <path>: Specify the path to save the packaged kernel artifacts (default: ${PACKAGE_PATH})."
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# we reference commit ids so that we can ensure we don't need to change the reference
# when we switch from pulling from private gerrit to public github.
KERNEL_COMMIT=822c418972646f73b3771a92fb78ab09a343897b
DRIVER_COMMIT=bb8b58e4b342f23efc6bc4a3fe931388ef811efe

if [[ -n ${LINUX_KERNEL_SSH_KEY:-} ]]; then
  echo "Using provided SSH key for Linux kernel checkout."
  LINUX_KERNEL_SSH_KEY_FILE=$(mktemp)
  trap 'rm -f "${LINUX_KERNEL_SSH_KEY_FILE}"' EXIT
  chmod 600 "${LINUX_KERNEL_SSH_KEY_FILE}"
  printf '%s\n' "${LINUX_KERNEL_SSH_KEY}" > "${LINUX_KERNEL_SSH_KEY_FILE}"
  export GIT_SSH_COMMAND="ssh ${LINUX_KERNEL_SSH_OPTIONS} -i ${LINUX_KERNEL_SSH_KEY_FILE} -o IdentitiesOnly=yes"

  if [[ -n ${LINUX_KERNEL_SSH_HOST_KEY:-} ]]; then
    echo "Using provided SSH host key for Linux kernel checkout."
    LINUX_KERNEL_SSH_HOST_KEY_FILE=$(mktemp)
    chmod 600 "${LINUX_KERNEL_SSH_HOST_KEY_FILE}"
    printf '%s\n' "${LINUX_KERNEL_SSH_HOST_KEY}" > "${LINUX_KERNEL_SSH_HOST_KEY_FILE}"
    export GIT_SSH_COMMAND="${GIT_SSH_COMMAND} -o UserKnownHostsFile=${LINUX_KERNEL_SSH_HOST_KEY_FILE} -o StrictHostKeyChecking=yes"
  fi
fi

if [[ $USE_GERRIT -eq 1 ]]; then
  SYNA_GERRIT_USER=$(id -n -u)
  LINUX_REPO_URL="ssh://${SYNA_GERRIT_USER}@sc-debu-git.synaptics.com:29420/astra/linux/main"
  LINUX_DRIVER_REPO_URL="ssh://${SYNA_GERRIT_USER}@sc-debu-git.synaptics.com:29420/debu/common/linux-driver/synaptics"  
else
  LINUX_REPO_URL="${LINUX_REPO_URL:-https://github.com/synaptics-astra/linux_6_12-main}"
  LINUX_DRIVER_REPO_URL="${LINUX_DRIVER_REPO_URL:-https://github.com/synaptics-astra/linux_6_12-drivers-synaptics}"
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

    git init main

    cd main
    git remote add origin "${LINUX_REPO_URL}"
    git fetch --depth=1 origin "${KERNEL_COMMIT}"
    git checkout FETCH_HEAD
    
    cd drivers

    git init synaptics

    cd synaptics
    git remote add origin "${LINUX_DRIVER_REPO_URL}"
    git fetch --depth=1 origin "${DRIVER_COMMIT}"
    git checkout FETCH_HEAD

    # configure kernel to build the kernel module against it and pack the required headers/Makefiles/scripts
    cd ${SYNA_KERNEL_CHECKOUT}
    make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- sl261x_defconfig
    export LOCALVERSION=""
    make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- modules_prepare
    tar -cvzf ${PACKAGE_PATH} .config include/ arch/arm64/include/ scripts/ Makefile arch/arm64/Makefile
    rm -rf ${SYNA_KERNEL_CHECKOUT}
fi
