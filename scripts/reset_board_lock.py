#!/usr/bin/env python3
"""Remove a stale session lock from a remote Astra board.

Usage:
    python3 scripts/reset_board_lock.py root@10.46.130.17
    python3 scripts/reset_board_lock.py root@10.46.130.17 --port 2222

The lock is a directory at /tmp/torq_board_test.lock on the board.
This script reads who created the lock (if available), removes it,
and confirms the result.
"""
import argparse
import subprocess
import sys

BOARD_LOCK_DIR = "/tmp/torq_board_test.lock"
BOARD_LOCK_INFO = f"{BOARD_LOCK_DIR}/info"


def _ssh(board_addr: str, port: int, cmd: str, timeout: int = 10) -> str | None:
    """Run a command on the board via SSH.  Returns stdout or None on failure."""
    try:
        result = subprocess.run(
            [
                "ssh",
                "-o", "BatchMode=yes",
                "-o", f"ConnectTimeout={timeout}",
                "-o", "StrictHostKeyChecking=no",
                "-p", str(port),
                board_addr,
                cmd,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove a stale session lock from a remote Astra board."
    )
    parser.add_argument(
        "board_addr",
        help="SSH address of the board (e.g. root@10.46.130.17)",
    )
    parser.add_argument(
        "--port", type=int, default=22,
        help="SSH port (default: 22)",
    )
    args = parser.parse_args()

    # Check if lock exists by reading lock info.
    info = _ssh(args.board_addr, args.port, f"cat {BOARD_LOCK_INFO}")
    if info is None:
        # Maybe the directory exists but info file is missing, or no lock at all.
        check = _ssh(args.board_addr, args.port, f"test -d {BOARD_LOCK_DIR} && echo locked")
        if check and "locked" in check:
            print(f"Lock directory exists on {args.board_addr} but has no info file.")
        else:
            print(f"No lock found on {args.board_addr}. Nothing to do.")
            return
    else:
        print(f"Current lock on {args.board_addr}:")
        for line in info.splitlines():
            print(f"  {line}")

    # Remove the lock.
    result = _ssh(args.board_addr, args.port, f"rm -rf {BOARD_LOCK_DIR}")
    if result is not None:
        print(f"Lock removed from {args.board_addr}.")
    else:
        print(f"ERROR: Failed to remove lock from {args.board_addr}.", file=sys.stderr)
        print("  The board may be unreachable.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
