#!/bin/bash

set -e

IMAGE_NAME="ghcr.io/syna-astra-dev/iree-synaptics-synpu/builder:latest"

# create a temporary file with the passwd file of the container

PATCHED_CONFIG=$(mktemp -d)

# on exit cleanup the temporary file
trap 'rm -rf ${PATCHED_CONFIG}' EXIT

# create a patched passwd file (this handles cases where ubuntu and the local user have different ids and the local
# user is not in /etc/passwd, like when using LDAP) 
docker run -it --rm -v ${PATCHED_CONFIG}:/config ${IMAGE_NAME} bash -c "
    sed 's|^ubuntu:|$(id -n -u):x:$(id -u):$(id -g):,,,:$HOME:/bin/bash|' < /etc/passwd > /config/passwd ;
    sed 's|^ubuntu:\(.*\)|$(id -n -u):\1|' < /etc/shadow > /config/shadow ;
    sed 's/ubuntu:x:1000/$(id -n -g):x:$(id -g)/' < /etc/group > /config/group"

# run the container with the patched passwd file
docker run -it --rm -v $(pwd):$(pwd) \
           -v ${HOME}/.ssh:${HOME}/.ssh \
           -v ${PATCHED_CONFIG}/passwd:/etc/passwd \
           -v ${PATCHED_CONFIG}/group:/etc/group \
           -u $(id -u):$(id -g) -w $(pwd) ${IMAGE_NAME}
