#!/usr/bin/env bash
# Copy results from the bucket into results/, then write
# results/colocated/summary.csv and a CCL-Bench workload card per layout.
#
#   scripts/fetch_results.sh
set -euo pipefail
source "$(dirname "$0")/config.sh"
REPO=$(cd "$(dirname "$0")/.." && pwd)

mkdir -p "$REPO/results"
gcloud storage rsync -r "$BUCKET/$BUCKET_PREFIX/results/" "$REPO/results/"
if [ -d "$REPO/results/colocated" ]; then
  python3 "$REPO/scripts/summarize.py" "$REPO/results/colocated"
  for d in "$REPO"/results/colocated/*/; do
    python3 "$REPO/scripts/workload_card.py" "${d%/}"
  done
fi
echo "Profile traces: $BUCKET/$BUCKET_PREFIX/profile/   Logs: $BUCKET/$BUCKET_PREFIX/logs/"
