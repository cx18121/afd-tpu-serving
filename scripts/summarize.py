#!/usr/bin/env python3
"""Collect vllm bench serve results into CSV tables.

usage: scripts/summarize.py results

Reads results/<family>/<layout>/<name>.json, where family is "colocated" or
"afd" and name is one of
  w<ISL>x<OSL>[e]_c<N>_r<K>.json  random tokens; "e" = exact lengths,
                                  otherwise uniform in [0.8*L, L]
  sharegpt_c<N>_r<K>.json       ShareGPT conversations
  colocated_c<N>[_r<K>].json    first protocol (exact 1024/256, 200 prompts)
and writes
  results/summary.csv   one row per family, layout, workload, concurrency
                        (mean over repeats, with output throughput min/max)
  results/slo.csv       best output throughput per chip whose mean time per
                        output token stays within each latency target
"""
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

METRICS = [  # (csv column, key in the vllm bench serve JSON)
    ("completed", "completed"),
    ("failed", "failed"),  # nonzero means errors; check the server log
    ("duration_s", "duration"),
    ("request_throughput_req_s", "request_throughput"),
    ("output_throughput_tok_s", "output_throughput"),
    ("total_token_throughput_tok_s", "total_token_throughput"),
    ("mean_ttft_ms", "mean_ttft_ms"),
    ("median_ttft_ms", "median_ttft_ms"),
    ("p99_ttft_ms", "p99_ttft_ms"),
    ("mean_tpot_ms", "mean_tpot_ms"),
    ("median_tpot_ms", "median_tpot_ms"),
    ("p99_tpot_ms", "p99_tpot_ms"),
    ("mean_itl_ms", "mean_itl_ms"),
    ("p99_itl_ms", "p99_itl_ms"),
    ("mean_e2el_ms", "mean_e2el_ms"),
]
# per_user_tok_s = 1000 / mean TPOT: the decode speed one user sees.
COLUMNS = (["family", "layout", "workload", "chips", "concurrency", "repeats",
            "per_user_tok_s", "output_tok_s_per_chip"]
           + [c for c, _ in METRICS] + ["output_tok_s_min", "output_tok_s_max"])
CHIPS = {"tp8_ep": 8, "dp2_tp4_ep": 8, "tp4_ep": 4, "afd_dp2": 8}
TPOT_TARGETS_MS = (20, 50)
NAMES = [
    (re.compile(r"^w(\d+)x(\d+)(e?)_c(\d+)_r(\d+)\.json$"),
     lambda m: f"{m[1]}x{m[2]}" + ("-exact" if m[3] else "")),
    (re.compile(r"^sharegpt_c(\d+)_r(\d+)\.json$"), lambda m: "sharegpt"),
    (re.compile(r"^colocated_c(\d+)(?:_r(\d+))?\.json$"), lambda m: "1024x256-exact"),
]


def parse(name):
    for pattern, workload in NAMES:
        m = pattern.match(name)
        if m:
            c = int(m[4]) if name.startswith("w") else int(m[1])
            return workload(m), c
    return None


def main() -> None:
    root = Path(sys.argv[1])
    runs = defaultdict(list)
    for f in sorted(root.glob("*/*/*.json")):
        parsed = parse(f.name)
        if parsed:
            family, layout = f.parent.parent.name, f.parent.name
            runs[(family, layout, *parsed)].append(json.loads(f.read_text()))
    if not runs:
        sys.exit(f"no benchmark results under {root}/<family>/<layout>/")

    rows = []
    for (family, layout, workload, c), results in sorted(runs.items()):
        chips = CHIPS.get(layout)
        row = {"family": family, "layout": layout, "workload": workload, "chips": chips,
               "concurrency": c, "repeats": len(results)}
        for col, key in METRICS:
            vals = [r[key] for r in results if r.get(key) is not None]
            row[col] = statistics.mean(vals) if vals else None
        if row["mean_tpot_ms"]:
            row["per_user_tok_s"] = 1000.0 / row["mean_tpot_ms"]
        if chips and row["output_throughput_tok_s"]:
            row["output_tok_s_per_chip"] = row["output_throughput_tok_s"] / chips
        out = [r["output_throughput"] for r in results]
        row["output_tok_s_min"], row["output_tok_s_max"] = min(out), max(out)
        if row["failed"]:
            print(f"WARNING: {family}/{layout} {workload} c={c} had failed requests",
                  file=sys.stderr)
        rows.append(row)

    def write(path, cols, data):
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, cols)
            w.writeheader()
            for row in data:
                w.writerow({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})

    write(root / "summary.csv", COLUMNS, rows)

    # Latency-target readout: for each layout and workload, the best output
    # throughput per chip among points whose mean TPOT meets the target.
    slo = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["family"], row["layout"], row["workload"])].append(row)
    for (family, layout, workload), pts in sorted(groups.items()):
        for target in TPOT_TARGETS_MS:
            ok = [p for p in pts if p["mean_tpot_ms"] and p["mean_tpot_ms"] <= target
                  and p.get("output_tok_s_per_chip")]
            best = max(ok, key=lambda p: p["output_tok_s_per_chip"]) if ok else None
            slo.append({"family": family, "layout": layout, "workload": workload,
                        "tpot_target_ms": target,
                        "best_output_tok_s_per_chip": best["output_tok_s_per_chip"] if best else None,
                        "at_concurrency": best["concurrency"] if best else None,
                        "mean_tpot_ms": best["mean_tpot_ms"] if best else None})
    write(root / "slo.csv", ["family", "layout", "workload", "tpot_target_ms",
                             "best_output_tok_s_per_chip", "at_concurrency", "mean_tpot_ms"], slo)
    print((root / "summary.csv").read_text())
    print((root / "slo.csv").read_text())


if __name__ == "__main__":
    main()
