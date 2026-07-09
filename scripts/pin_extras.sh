#!/bin/bash

BASE_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")/..

echo $(cd $BASE_DIR/extras && git rev-parse HEAD) > $BASE_DIR/scripts/extras.gitlink
