#!/usr/bin/env python3
"""Collect vllm bench serve result JSONs (colocated_c*.json) into summary.csv.

usage: scripts/summarize.py results/colocated
"""
import csv
import json
import re
import sys
from pathlib import Path

COLUMNS = [
    ("concurrency", None),
    ("completed", "completed"),
    ("duration_s", "duration"),
    ("request_throughput_req_s", "request_throughput"),
    ("output_throughput_tok_s", "output_throughput"),
    ("total_token_throughput_tok_s", "total_token_throughput"),
    ("per_user_tok_s", None),  # 1000 / mean TPOT: decode speed one user sees
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


def main() -> None:
    d = Path(sys.argv[1])
    rows = []
    for f in d.glob("colocated_c*.json"):
        r = json.loads(f.read_text())
        c = int(re.search(r"_c(\d+)\.json$", f.name).group(1))
        row = {"concurrency": c, "per_user_tok_s": 1000.0 / r["mean_tpot_ms"]}
        for col, key in COLUMNS:
            if key:
                row[col] = r.get(key)
        rows.append(row)
    if not rows:
        sys.exit(f"no colocated_c*.json in {d}")
    rows.sort(key=lambda r: r["concurrency"])
    out = d / "summary.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, [c for c, _ in COLUMNS])
        w.writeheader()
        for row in rows:
            w.writerow({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
    print(out.read_text())


if __name__ == "__main__":
    main()
