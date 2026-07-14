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
TOOLCHAIN_VERSION="scarthgap_6.12_v2.4.0"
TOOLCHAIN_VERSION_FILE="${TOOLCHAIN_DIR}/.astra_toolchain_release"
FILE="sl2619_scarthgap-poky-glibc-x86_64-astra-media-cortexa55-sl2619-toolchain-5.0.9.sh"
HASH="4429ec1c5eb092e1a53e943f133e897b"
URL="https://github.com/synaptics-astra/sdk/releases/download/${TOOLCHAIN_VERSION}"

if [ -d "$TOOLCHAIN_DIR" ] && { [ ! -f "$TOOLCHAIN_VERSION_FILE" ] || [ "$(cat "$TOOLCHAIN_VERSION_FILE")" != "$TOOLCHAIN_VERSION" ]; }; then
    if [ -t 0 ]; then
        read -r -p "Existing Astra toolchain at $TOOLCHAIN_DIR does not match $TOOLCHAIN_VERSION. Reinstall? [y/N]: " REINSTALL_CONFIRM
        case "$REINSTALL_CONFIRM" in
            [yY]|[yY][eE][sS])
                echo "Removing existing Astra toolchain from $TOOLCHAIN_DIR to install $TOOLCHAIN_VERSION..."
                rm -rf "$TOOLCHAIN_DIR"
                ;;
            *)
                echo "Keeping existing toolchain at $TOOLCHAIN_DIR"
                exit 0
                ;;
        esac
    else
        echo "Non-interactive shell detected. Removing existing Astra toolchain from $TOOLCHAIN_DIR to install $TOOLCHAIN_VERSION..."
        rm -rf "$TOOLCHAIN_DIR"
    fi
fi

if [ ! -d "$TOOLCHAIN_DIR" ]; then
    echo "Downloading Astra toolchain... at $1"
    wget -nv -c "$URL/$FILE.000" "$URL/$FILE.001"

    echo "Combining and installing..."
    cat "${FILE}."* > "$FILE" && rm -f "${FILE}."*

    if echo "$HASH $FILE" | md5sum -c -; then
        chmod +x "$FILE"
        ./"$FILE" -y -d "$TOOLCHAIN_DIR"
        printf '%s\n' "$TOOLCHAIN_VERSION" > "$TOOLCHAIN_VERSION_FILE"
        rm -f "$FILE"
        echo "Toolchain installed to $TOOLCHAIN_DIR"
    else
        echo "MD5 Verification Failed!"
        rm -f "$FILE"
        exit 1
    fi
else
    echo "Toolchain $TOOLCHAIN_VERSION already installed at $TOOLCHAIN_DIR"
fi
