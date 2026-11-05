#!/usr/bin/env python3
"""Collect vllm bench serve results into summary.csv: one row per layout and
concurrency level, averaged over repeats.

usage: scripts/summarize.py results/colocated
Reads results/colocated/<layout>/colocated_c<N>[_r<K>].json.
"""
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# (csv column, key in the vllm bench serve JSON)
METRICS = [
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
# output_tok_s_min/max: spread across repeats, to judge noise.
COLUMNS = (["layout", "chips", "concurrency", "repeats", "per_user_tok_s"]
           + [c for c, _ in METRICS] + ["output_tok_s_min", "output_tok_s_max"])
CHIPS = {"tp8_ep": 8, "dp2_tp4_ep": 8, "tp4_ep": 4}
NAME = re.compile(r"^colocated_c(\d+)(?:_r(\d+))?\.json$")


def main() -> None:
    root = Path(sys.argv[1])
    runs = defaultdict(list)  # (layout, concurrency) -> [result dict]
    for f in sorted(root.glob("*/colocated_c*.json")):
        m = NAME.match(f.name)
        if m:
            runs[(f.parent.name, int(m.group(1)))].append(json.loads(f.read_text()))
    if not runs:
        sys.exit(f"no <layout>/colocated_c*.json under {root}")

    rows = []
    for (layout, c), results in sorted(runs.items()):
        row = {"layout": layout, "chips": CHIPS.get(layout, ""), "concurrency": c,
               "repeats": len(results)}
        for col, key in METRICS:
            vals = [r[key] for r in results if r.get(key) is not None]
            row[col] = statistics.mean(vals) if vals else None
        if row["mean_tpot_ms"]:
            row["per_user_tok_s"] = 1000.0 / row["mean_tpot_ms"]
        out = [r["output_throughput"] for r in results]
        row["output_tok_s_min"], row["output_tok_s_max"] = min(out), max(out)
        if row["failed"]:
            print(f"WARNING: {layout} c={c} had failed requests", file=sys.stderr)
        rows.append(row)

    out_path = root / "summary.csv"
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
    print(out_path.read_text())


if __name__ == "__main__":
    main()
