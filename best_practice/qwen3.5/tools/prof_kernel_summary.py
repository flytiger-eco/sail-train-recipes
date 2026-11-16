#!/usr/bin/env python3
"""Summarize a torch profiler chrome trace by kernel family.

The trace is read as a stream: gzip -> JSON objects one at a time, so a
multi-GB trace never has to fit in memory. Output is a fixed, greppable
report that answers the question the trace is captured for: did the grouped
GEMM ops run the CUTLASS/RTC kernel, or did they fall back to one cuBLAS
GEMM per expert?

Usage: prof_kernel_summary.py <trace.json.gz> [--top N]
"""

import gzip
import json
import os
import re
import sys

_decoder = json.JSONDecoder()
_digits = re.compile(r"\d+")


def family(name):
    return _digits.sub("#", name)


def scan(path, counts, totals):
    buf = ""
    events = 0
    with gzip.open(path, "rt", errors="ignore") as f:
        while True:
            chunk = f.read(4 << 20)
            done = not chunk
            buf += chunk
            while True:
                i = 0
                while i < len(buf) and buf[i] in " \t\r\n,":
                    i += 1
                if i:
                    buf = buf[i:]
                if not buf:
                    break
                if buf[0] != "{":
                    # deviceProperties and other arrays: step past them
                    if len(buf) < 8192:
                        break
                    buf = buf[1:]
                    continue
                try:
                    obj, end = _decoder.raw_decode(buf)
                except ValueError:
                    break
                buf = buf[end:]
                if obj.get("cat") == "kernel" and "dur" in obj:
                    name = obj.get("name", "?")
                    counts[name] = counts.get(name, 0) + 1
                    totals[name] = totals.get(name, 0.0) + obj["dur"]
                    events += 1
            if done:
                return events


def main():
    args = [a for a in sys.argv[1:]]
    top = 8
    if "--top" in args:
        i = args.index("--top")
        top = int(args[i + 1])
        del args[i:i + 2]
    if len(args) != 1:
        print(__doc__.strip().splitlines()[0])
        return 2
    path = args[0]
    if not os.path.exists(path):
        print(f"missing trace: {path}")
        return 1

    counts, totals = {}, {}
    events = scan(path, counts, totals)
    total_us = sum(totals.values())
    print(f"trace: {path}")
    print(f"size: {os.path.getsize(path) / 1e6:.1f} MB  kernel_events: {events}  kernel_total: {total_us / 1000:.1f} ms")
    if not events:
        print("no kernel events found: check profile mode and artifact")
        return 1

    def block(title, pred, limit):
        keys = [k for k in totals if pred(k)]
        agg_us = sum(totals[k] for k in keys)
        agg_calls = sum(counts[k] for k in keys)
        print(f"\n{title}: {agg_calls} launches  {agg_us / 1000:.1f} ms  ({100 * agg_us / total_us:.1f}%)")
        for k in sorted(keys, key=lambda x: -totals[x])[:limit]:
            print(f"  {counts[k]:6d} x {totals[k] / counts[k]:9.1f} us  {totals[k] / 1000:9.1f} ms  {k[:96]}")

    # The RTC/cutlass arm names its kernel after the operand/layout combination,
    # so the three training phases show up as three distinct symbols.
    block("rtc grouped gemm", lambda k: k.startswith("nvte_grouped_gemm3"), 12)
    block("cublas gemm", lambda k: k.startswith("gemm_ktype"), 6)
    print(f"\ntop {top} kernels by total time:")
    for k in sorted(totals, key=lambda x: -totals[x])[:top]:
        print(f"  {counts[k]:6d} x {totals[k] / counts[k]:9.1f} us  {totals[k] / 1000:9.1f} ms  {family(k)[:96]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
