#!/usr/bin/env bash
# Follow an --auto run: prints the VM state and the latest status line when
# either changes, and exits once the VM is gone.
#
#   scripts/watch_run.sh NAME ZONE RUN_ID
set -uo pipefail
source "$(dirname "$0")/config.sh"
if [ $# -ne 3 ]; then
  echo "usage: $0 NAME ZONE RUN_ID" >&2
  exit 2
fi
NAME=$1 ZONE=$2 RUN_ID=$3
last="" seen_running=false
while true; do
  # Only "not found" means the VM is gone; other errors are transient.
  if ! s=$(gcloud compute instances describe "$NAME" --project="$PROJECT" --zone="$ZONE" \
      --format='value(status)' 2>&1); then
    case "$s" in *"was not found"*) s=GONE ;; *) s=UNKNOWN ;; esac
  fi
  [ "$s" = RUNNING ] && seen_running=true
  st=$(gcloud storage cat "$RUNS_URI/$RUN_ID/status.txt" 2>/dev/null | tail -1)
  line="$s | $st"
  if [ "$line" != "$last" ]; then echo "$(date +%T) $line"; last=$line; fi
  if [ "$s" = GONE ]; then
    if $seen_running; then echo "VM gone after running. Results: scripts/fetch_results.sh $RUN_ID"
    else echo "VM gone without running (request expired or was cancelled)."; fi
    exit 0
  fi
  sleep 30
done
