#!/usr/bin/env python3
"""Set the hostname on a remote board via SSH.

Usage:
    python3 scripts/set_board_hostname.py 10.46.130.17 sl2619-dev-board-001
    python3 scripts/set_board_hostname.py 10.46.130.17 sl2619-dev-board-001 --port 22
"""

import argparse
import subprocess
import sys


def run_ssh(ip: str, port: int, cmd: str, timeout: int = 10) -> str:
    """Run a command on the remote board and return its output."""
    return subprocess.check_output(
        [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={timeout}",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-p", str(port),
            f"root@{ip}",
            cmd,
        ],
        stderr=subprocess.STDOUT,
        timeout=timeout + 5,
        text=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Set the hostname on a remote board.")
    parser.add_argument("ip", help="Board IP address (e.g. 10.46.130.17)")
    parser.add_argument("hostname", help="New hostname to set (e.g. sl2619-dev-board-001)")
    parser.add_argument("--port", type=int, default=22, help="SSH port (default: 22)")
    args = parser.parse_args()

    ip = args.ip
    new_hostname = args.hostname
    port = args.port

    print(f"Connecting to {ip}:{port} ...")

    old_hostname = run_ssh(ip, port, "hostname")
    print(f"Current hostname: {old_hostname}")

    if old_hostname == new_hostname:
        print("Hostname already set — nothing to do.")
        return

    # Set the transient hostname (takes effect immediately).
    run_ssh(ip, port, f"hostname {new_hostname}")

    # Persist across reboots by writing /etc/hostname.
    run_ssh(ip, port, f"echo {new_hostname} > /etc/hostname")

    # Update /etc/hosts so 127.0.0.1 resolves the new name.
    run_ssh(ip, port,
            f"sed -i 's/127\\.0\\.0\\.1.*/127.0.0.1\\tlocalhost {new_hostname}/' /etc/hosts")

    # Verify.
    verify = run_ssh(ip, port, "hostname")
    if verify == new_hostname:
        print(f"Hostname set to: {verify}")
    else:
        print(f"ERROR: Expected '{new_hostname}' but got '{verify}'", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
