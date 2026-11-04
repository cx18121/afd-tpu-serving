#!/usr/bin/env python3
"""Collect vllm bench serve results into summary.csv, one row per layout and
concurrency level.

usage: scripts/summarize.py results/colocated
Reads results/colocated/<layout>/colocated_c<N>.json.
"""
import csv
import json
import re
import sys
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
COLUMNS = ["layout", "chips", "concurrency", "per_user_tok_s"] + [c for c, _ in METRICS]
CHIPS = {"tp8_ep": 8, "dp2_tp4_ep": 8, "tp4_ep": 4}


def main() -> None:
    root = Path(sys.argv[1])
    rows = []
    for f in sorted(root.glob("*/colocated_c*.json")):
        r = json.loads(f.read_text())
        layout = f.parent.name
        row = {
            "layout": layout,
            "chips": CHIPS.get(layout, ""),
            "concurrency": int(re.search(r"_c(\d+)\.json$", f.name).group(1)),
            "per_user_tok_s": 1000.0 / r["mean_tpot_ms"] if r.get("mean_tpot_ms") else None,
        }
        row.update({col: r.get(key) for col, key in METRICS})
        rows.append(row)
    if not rows:
        sys.exit(f"no <layout>/colocated_c*.json under {root}")
    rows.sort(key=lambda r: (r["layout"], r["concurrency"]))
    for r in rows:
        if r.get("failed"):
            print(f"WARNING: {r['layout']} c={r['concurrency']} had {r['failed']} failed requests",
                  file=sys.stderr)
    out = root / "summary.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
    print(out.read_text())


if __name__ == "__main__":
    main()
