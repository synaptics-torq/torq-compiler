#!/bin/bash

DOC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run --rm -v ${DOC_DIR}:${DOC_DIR} -w ${DOC_DIR} ghcr.io/syna-astra-dev/synaptics-sphinx-theme/builder:latest
