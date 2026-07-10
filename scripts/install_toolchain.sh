#!/bin/bash

set -e

function usage() {
    echo "usage: $0 PATH/TO/TOOLCHAIN_DIR"
}

if [[ -z "$1" ]]; then
    usage
    exit 1
fi

# Install Synaptics Astra Toolchain if not present
TOOLCHAIN_DIR="$1"
FILE="sl2619_scarthgap-poky-glibc-x86_64-astra-media-cortexa55-sl2619-toolchain-5.0.9.sh"
HASH="ba0cfb28c890a62db786ce8559347e91"
URL="https://github.com/synaptics-astra/sdk/releases/download/scarthgap_6.12_v2.1.0"

if [ ! -d "$TOOLCHAIN_DIR" ]; then
    echo "Downloading Astra toolchain... at $1"
    wget -nv -c "$URL/$FILE.000" "$URL/$FILE.001"

    echo "Combining and installing..."
    cat "${FILE}."* > "$FILE" && rm -f "${FILE}."*

    if echo "$HASH $FILE" | md5sum -c -; then
        chmod +x "$FILE"
        ./"$FILE" -y -d "$TOOLCHAIN_DIR"
        rm -f "$FILE"
        echo "Toolchain installed to $TOOLCHAIN_DIR"
    else
        echo "MD5 Verification Failed!"
        rm -f "$FILE"
        exit 1
    fi
else
    echo "Toolchain already installed at $TOOLCHAIN_DIR"
fi
