#!/usr/bin/env python3
"""Summarize a torch profiler chrome trace by kernel family.

The trace is scanned in chunks as raw bytes: kernel events carry
`"cat": "kernel", "name": ...` on one line and `"dur": ...` within the next
few lines, so a byte regex is enough. Full JSON decoding is avoided on
purpose: python_function events with callstacks make these traces several GB
and object-at-a-time decoding takes tens of minutes.

Output answers the question the trace is captured for: did the grouped GEMM
ops run the CUTLASS/RTC kernel, or fall back to one cuBLAS GEMM per expert?

Usage: prof_kernel_summary.py <trace.json.gz> [--top N]
"""

import gzip
import os
import re
import sys

EVENT = re.compile(
    rb'"cat": "kernel", "name": "([^"]*)".{0,200000}?"dur": ([0-9.]+),',
    re.S,
)
DIGITS = re.compile(r"\d+")
CHUNK = 8 << 20
OVERLAP = 1 << 20


def family(name):
    return DIGITS.sub("#", name)


def scan(path, counts, totals):
    buf = b""
    events = 0
    with gzip.open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK)
            done = not chunk
            buf += chunk
            cut = len(buf) if done else max(0, len(buf) - OVERLAP)
            for m in EVENT.finditer(buf, 0, cut):
                name = m.group(1).decode("utf-8", "replace")
                dur = float(m.group(2))
                counts[name] = counts.get(name, 0) + 1
                totals[name] = totals.get(name, 0.0) + dur
                events += 1
            buf = buf[cut:]
            if done:
                return events


def main():
    args = list(sys.argv[1:])
    top = 8
    if "--top" in args:
        i = args.index("--top")
        top = int(args[i + 1])
        del args[i:i + 2]
    if len(args) != 1:
        print("usage: prof_kernel_summary.py <trace.json.gz> [--top N]")
        return 2
    path = args[0]
    if not os.path.exists(path):
        print(f"missing trace: {path}")
        return 1

    counts, totals = {}, {}
    events = scan(path, counts, totals)
    total_us = sum(totals.values())
    print(f"trace: {path}")
    print(
        f"size: {os.path.getsize(path) / 1e6:.1f} MB  kernel_events: {events}"
        f"  kernel_total: {total_us / 1000:.1f} ms"
    )
    if not events:
        print("no kernel events found: check the profile mode and the artifact")
        return 1

    def block(title, prefix, limit):
        keys = [k for k in totals if k.startswith(prefix)]
        agg_us = sum(totals[k] for k in keys)
        calls = sum(counts[k] for k in keys)
        print(f"\n{title}: {calls} launches  {agg_us / 1000:.1f} ms  ({100 * agg_us / total_us:.1f}%)")
        for k in sorted(keys, key=lambda x: -totals[x])[:limit]:
            print(f"  {counts[k]:6d} x {totals[k] / counts[k]:9.1f} us  {totals[k] / 1000:9.1f} ms  {k[:96]}")

    # The RTC/cutlass arm names its kernel after the operand and layout
    # combination, so fwd, dgrad and wgrad show up as distinct symbols.
    block("rtc grouped gemm", "nvte_grouped_gemm3", 12)
    block("cublas gemm", "gemm_ktype", 6)
    print(f"\ntop {top} kernels by total time:")
    for k in sorted(totals, key=lambda x: -totals[x])[:top]:
        print(f"  {counts[k]:6d} x {totals[k] / counts[k]:9.1f} us  {totals[k] / 1000:9.1f} ms  {family(k)[:96]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
