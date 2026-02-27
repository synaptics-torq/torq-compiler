#!/usr/bin/env python3
"""Scan the 10.46.130.0/24 subnet for reachable boards via SSH.

For each host that responds, prints its IP and hostname.

Usage:
    python3 scripts/scan_boards.py [--subnet 10.46.130] [--port 22] [--timeout 2] [--workers 50]
"""

import argparse
import concurrent.futures
import re
import subprocess

# Only boards whose hostname matches this pattern are shown.
BOARD_HOSTNAME_RE = re.compile(r"^sl2619-dev-board-\d+$")


def probe(ip: str, port: int, timeout: int) -> tuple[str, str | None]:
    """Try to SSH into *ip* and return (ip, hostname) or (ip, None)."""
    try:
        out = subprocess.check_output(
            [
                "ssh",
                "-o", "BatchMode=yes",
                "-o", f"ConnectTimeout={timeout}",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR",
                "-p", str(port),
                f"root@{ip}",
                "hostname",
            ],
            stderr=subprocess.DEVNULL,
            timeout=timeout + 2,
            text=True,
        )
        return ip, out.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return ip, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan a /24 subnet for SSH-reachable boards.")
    parser.add_argument("--subnet", default="10.46.130", help="First three octets (default: 10.46.130)")
    parser.add_argument("--port", type=int, default=22, help="SSH port (default: 22)")
    parser.add_argument("--timeout", type=int, default=2, help="Per-host SSH timeout in seconds (default: 2)")
    parser.add_argument("--workers", type=int, default=50, help="Parallel workers (default: 50)")
    parser.add_argument("--find", default=None, metavar="HOSTNAME",
                        help="Stop as soon as this hostname is found and print its details")
    args = parser.parse_args()

    ips = [f"{args.subnet}.{i}" for i in range(1, 255)]

    if args.find:
        print(f"Searching for hostname '{args.find}' in {args.subnet}.1-254 ...")
    else:
        print(f"Scanning {args.subnet}.1-254 on port {args.port} (timeout {args.timeout}s) ...")
        print(f"{'IP':<20s} {'HOSTNAME'}")
        print("-" * 40)

    results: list[tuple[str, str]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(probe, ip, args.port, args.timeout): ip for ip in ips}
        for future in concurrent.futures.as_completed(futures):
            ip, hostname = future.result()
            if hostname is None:
                continue
            if not BOARD_HOSTNAME_RE.match(hostname):
                continue
            if args.find:
                if hostname == args.find:
                    print(f"{ip:<20s} {hostname}")
                    # Cancel remaining futures and exit early.
                    for f in futures:
                        f.cancel()
                    return
            else:
                results.append((ip, hostname))

    if args.find:
        print(f"Hostname '{args.find}' not found.")
        raise SystemExit(1)

    # Sort by last octet for readable output.
    results.sort(key=lambda r: int(r[0].rsplit(".", 1)[1]))

    for ip, hostname in results:
        print(f"{ip:<20s} {hostname}")

    print("-" * 40)
    print(f"{len(results)} board(s) reachable.")


if __name__ == "__main__":
    main()
