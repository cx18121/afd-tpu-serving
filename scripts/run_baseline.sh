#!/usr/bin/env bash
# Serve the model on the VM, run the benchmark sweep and profile, upload to the
# bucket, then pull results here and write results/colocated/summary.csv.
#
#   scripts/run_baseline.sh NAME ZONE            # Qwen3-30B-A3B sweep (~1-1.5h cold)
#   scripts/run_baseline.sh NAME ZONE smoke      # Qwen3-0.6B setup check (~10 min)
#
# The run is detached on the VM, so a dropped SSH connection or closed laptop
# does not stop it; rerunning this script with the same mode while it is in
# progress just resumes watching.
set -euo pipefail
source "$(dirname "$0")/config.sh"
REPO=$(cd "$(dirname "$0")/.." && pwd)

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
  echo "usage: $0 NAME ZONE [baseline|smoke]" >&2
  exit 2
fi
NAME=$1 ZONE=$2 MODE=${3:-baseline}
SSH=(gcloud compute ssh "$NAME" --project="$PROJECT" --zone="$ZONE" --command)
STATUS=$VM_WORKDIR/logs/$MODE.status
OUT=$VM_WORKDIR/logs/$MODE.out

state=$("${SSH[@]}" "sudo cat $STATUS 2>/dev/null || echo NONE")
if [ "$state" != RUNNING ]; then
  gcloud compute scp --project="$PROJECT" --zone="$ZONE" \
    "$REPO/scripts/config.sh" "$REPO"/scripts/vm/*.sh "$NAME":~/afd-scripts/
  "${SSH[@]}" "sudo mkdir -p $VM_WORKDIR/logs && sudo setsid nohup bash -c \"bash \$HOME/afd-scripts/vm_baseline.sh $MODE > $OUT 2>&1\" < /dev/null > /dev/null 2>&1 & sleep 5; sudo cat $STATUS"
fi

echo "Watching $MODE on $NAME (log: $OUT)"
"${SSH[@]}" "sudo touch $OUT"
while true; do
  state=$("${SSH[@]}" "sudo cat $STATUS 2>/dev/null || echo NONE")
  "${SSH[@]}" "sudo tail -n 3 $OUT 2>/dev/null; sudo tail -n 1 $VM_WORKDIR/logs/server_$MODE.log 2>/dev/null | cut -c1-200" || true
  case "$state" in
    DONE) break ;;
    FAILED|NONE) echo "Run $MODE ended as $state. Server log tail:" >&2
      "${SSH[@]}" "sudo tail -n 60 $VM_WORKDIR/logs/server_$MODE.log" >&2 || true
      exit 1 ;;
  esac
  sleep 60
done

RUN_NAME=$([ "$MODE" = baseline ] && echo colocated || echo smoke)
mkdir -p "$REPO/results/$RUN_NAME"
gcloud storage rsync -r "$BUCKET/$BUCKET_PREFIX/results/$RUN_NAME/" "$REPO/results/$RUN_NAME/"
python3 "$REPO/scripts/summarize.py" "$REPO/results/$RUN_NAME"
echo "Results uploaded to $BUCKET/$BUCKET_PREFIX/ and copied to results/$RUN_NAME/."
echo "Next: scripts/teardown.sh $NAME $ZONE"
