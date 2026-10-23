#!/bin/bash

# usage: setup_mpact.sh [ARCHIVE]
#
# Without ARCHIVE, downloads the mpact library for this platform from the snapshot release.
# ARCHIVE is either an mpact artifact of a CI run (mpact.zip or mpact-macos.zip) or a
# coralnpu-mpact-simulator-*.tar.gz.

set -e

BASE_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")/..

# same directory as the one used in runtime/torq_hw/CMakeLists.txt
PLATFORM=$(uname -m)-$(uname -s)
MPACT_DIR=${BASE_DIR}/third_party/coralnpu-mpact/${PLATFORM}

# the tarballs are named after the runner that built them (see build-torq.yml)
case "$(uname -s)" in
    Darwin) RUNNER=macos ;;
    *) RUNNER=ubuntu ;;
esac

ARCHIVE=$1

mkdir -p ${MPACT_DIR}

if [[ -z "${ARCHIVE}" ]]; then
    echo "Downloading MPACT artifacts to ${MPACT_DIR}..."

    ${BASE_DIR}/scripts/download_artifact.py --output-dir ${MPACT_DIR} --glob "coralnpu-mpact-simulator-${RUNNER}-*.tar.gz"
elif [[ "${ARCHIVE}" == *.zip ]]; then
    unzip -o -j ${ARCHIVE} "coralnpu-mpact-simulator-*.tar.gz" -d ${MPACT_DIR}
else
    cp ${ARCHIVE} ${MPACT_DIR}/
fi

echo "Extracting MPACT artifacts to ${MPACT_DIR}..."

tar -xzf ${MPACT_DIR}/coralnpu-mpact-simulator*.tar.gz --strip-components=1 -C ${MPACT_DIR}

rm ${MPACT_DIR}/coralnpu-mpact-simulator*.tar.gz

# a browser quarantines the files it downloads (unzip and tar pass it on to the extracted
# files) and Gatekeeper then refuses to load the library, which is not notarized
if [[ "$(uname -s)" == "Darwin" ]]; then
    for f in ${MPACT_DIR}/*; do
        if xattr -p com.apple.quarantine "$f" > /dev/null 2>&1; then
            chmod u+w "$f"
            xattr -d com.apple.quarantine "$f"
        fi
    done
fi
