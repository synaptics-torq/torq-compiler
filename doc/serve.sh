#!/bin/bash


BASE_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

cd ${BASE_DIR}

sphinx-autobuild . _build/html