#!/usr/bin/env python3
"""Kill stale Stage-1 training jobs launched by older background scripts."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time


OLD_PATTERNS = (
    "reset_policy_noise_std=0.055",
    "stage1a_recoverywindow",
    "stage1a_midguard_recovery",
)


def iter_user_processes(user: str) -> list[tuple[int, str]]:
    output = subprocess.check_output(
        ["ps", "-u", user, "-o", "pid=,args="],
        text=True,
        errors="ignore",
    )
    processes: list[tuple[int, str]] = []
    for line in output.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        processes.append((pid, parts[1]))
    return processes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=3600.0)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--user", type=str, default=os.environ.get("USER", "zxl"))
    args = parser.parse_args()

    deadline = time.time() + max(1.0, args.seconds)
    own_pid = os.getpid()
    print(f"[killer] watching stale Stage-1 jobs for user={args.user}", flush=True)
    while time.time() < deadline:
        try:
            processes = iter_user_processes(args.user)
        except Exception as exc:
            print(f"[killer] ps failed: {exc}", flush=True)
            time.sleep(args.interval)
            continue

        for pid, cmd in processes:
            if pid == own_pid:
                continue
            if "legged_lab/scripts/train.py" not in cmd:
                continue
            if not any(pattern in cmd for pattern in OLD_PATTERNS):
                continue
            print(f"[killer] SIGINT stale train pid={pid}: {cmd}", flush=True)
            try:
                os.kill(pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                print(f"[killer] permission denied pid={pid}: {exc}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
