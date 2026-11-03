#!/usr/bin/env bash
# Copy one run's results from the bucket into results/, then write
# results/colocated/summary.csv and a CCL-Bench workload card per layout.
#
#   scripts/fetch_results.sh            # latest run
#   scripts/fetch_results.sh RUN_ID     # e.g. 20261103-140000-charlie-v6e
set -euo pipefail
source "$(dirname "$0")/config.sh"
REPO=$(cd "$(dirname "$0")/.." && pwd)

RUN_ID=${1:-}
if [ -z "$RUN_ID" ]; then
  # Newest run that has results. Run ids start with a timestamp, so sorting
  # them is chronological; requests that never started have only scripts/.
  RUN_ID=$(gcloud storage ls "$RUNS_URI/*/results/" 2>/dev/null \
    | sed -n 's#^.*/runs/\([^/]*\)/results/.*#\1#p' | sort -u | tail -1 || true)
  [ -n "$RUN_ID" ] || { echo "No runs with results under $RUNS_URI/" >&2; exit 1; }
fi
echo "Run $RUN_ID"
gcloud storage cat "$RUNS_URI/$RUN_ID/status.txt" 2>/dev/null | tail -3 || true

# Replace local results with exactly this run's, so runs never mix.
rm -rf "$REPO/results/colocated" "$REPO/results/smoke"
mkdir -p "$REPO/results"
gcloud storage rsync -r "$RUNS_URI/$RUN_ID/results/" "$REPO/results/"
echo "$RUN_ID" > "$REPO/results/RUN_ID"
if [ -d "$REPO/results/colocated" ]; then
  python3 "$REPO/scripts/summarize.py" "$REPO/results/colocated"
  for d in "$REPO"/results/colocated/*/; do
    python3 "$REPO/scripts/workload_card.py" "${d%/}" "$RUN_ID"
  done
fi
echo "Profile traces: $RUNS_URI/$RUN_ID/profile/   Logs: $RUNS_URI/$RUN_ID/logs/"
