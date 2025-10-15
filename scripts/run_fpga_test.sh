#!/bin/bash

# This script uncompresses a test case created by test_fpga.py
# in a temporary directory and runs the model using iree-run-module
# on the FPGA
# 
# The script uses a ubuntu 24.04 rootfs to run the iree-run-module
# as the AWS FPGA server is running an older distribution without
# a compatible libc and libc++.

set -e
set -x 

# get first argument and provide error if not found
if [ -z "$1" || -z "$2" || -z "$3" ]; then
    echo "Please provide the path to the test package: $0 <path-to-test-package> <output-path> <number-of-outputs>"
    exit 1
fi

INPUT_PACKAGE=$(readlink -f "$1")
OUT_FILE=$(readlink -f "$2")
NUMBER_OF_OUTPUTS=$3

shift 3

# create a temporary directory with mktemp
OUT_DIR=$(mktemp -d /tmp/torq_test.XXXXXXX)

# ensure we clean up the directory when the script exits
trap 'rm -rf "$OUT_DIR"' EXIT

# unzip the test package in the temporary directory
unzip -q ${INPUT_PACKAGE} -d "$OUT_DIR"

# make sure no iree-runtime-module is running
killall iree-run-module || true

# reprogram the FPGA to have a known state
/home/share/bin/load_fpga /home/share/afi/synpu-fpga-250418-150905

# run the test
pushd "$OUT_DIR"

INPUTS=$(ls input_*.npy | xargs -n1 -I{} echo "--input=@{} ")

OUTPUTS=""
for i in $(seq 0 $((${NUMBER_OF_OUTPUTS} - 1))); do
    OUTPUTS="${OUTPUTS} --output=@output_fpga${i}.bin "
done

# this is used to get access to the Ubuntu-24.04 libc and libc++ libraries
UBUNTU_ROOTFS=/home/zSvcsynpu/ubuntu-24.04.5-rootfs
UBUNTU_ROOTFS_LD_LIBRARY_PATH="${UBUNTU_ROOTFS}/lib/x86_64-linux-gnu/:${UBUNTU_ROOTFS}/usr/lib/x86_64-linux-gnu/:${UBUNTU_ROOTFS}/lib:${UBUNTU_ROOTFS}/usr/lib"
UBUNTU_ROOTFS_LD_LINUX="${UBUNTU_ROOTFS}/lib64/ld-linux-x86-64.so.2"

# allow next command to fail, we don't want the whole script to fail
# so that we can still collect the output files (traces)
set +e

LD_LIBRARY_PATH=${UBUNTU_ROOTFS_LD_LIBRARY_PATH} ${UBUNTU_ROOTFS_LD_LINUX} \
    ~/iree-run-module --device=torq \
                      --torq_hw_type=aws_fpga \
                      --module=model.vmfb \
                      --function=main \
                      ${INPUTS} \
                      ${OUTPUTS} \
                      --torq_profile=log_fpga.csv \
                      --torq_profile_host=log_fpga_host.csv  "$@"

 
# store the return code of the previous command
RET_CODE=$?

# re-enable automatic exit on error
set -e

popd

# zip the output to the output path
(cd "$OUT_DIR" && rm -rf "${OUT_FILE}" && zip -r "${OUT_FILE}" buffers-fpga output_fpga*.bin log_fpga.csv log_fpga_host.csv)

# return the iree-run-module return code
exit $RET_CODE