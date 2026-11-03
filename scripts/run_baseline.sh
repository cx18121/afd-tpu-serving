#!/usr/bin/env bash
# Serve Qwen3-30B-A3B on the VM, run the benchmark sweep and a profiled run for
# each colocated layout, upload everything to the bucket, then pull results here
# and write results/colocated/summary.csv.
#
#   scripts/run_baseline.sh NAME ZONE              # all steps for the VM's chip count
#   scripts/run_baseline.sh NAME ZONE tp8_ep       # only the listed steps
#
# Default steps: 8 chips -> smoke tp8_ep dp2_tp4_ep; 4 chips -> smoke tp4_ep.
# smoke is a ~10 minute Qwen3-0.6B check that the container, chips, and
# benchmark client work before the long runs.
#
# Each step runs detached on the VM, so a dropped SSH connection or closed
# laptop does not stop it. Rerunning this script resumes watching a step that is
# still running and skips steps already DONE.
set -euo pipefail
source "$(dirname "$0")/config.sh"
REPO=$(cd "$(dirname "$0")/.." && pwd)

if [ $# -lt 2 ]; then
  echo "usage: $0 NAME ZONE [STEP...]" >&2
  exit 2
fi
NAME=$1 ZONE=$2
shift 2
SSH=(gcloud compute ssh "$NAME" --project="$PROJECT" --zone="$ZONE" --command)

if [ $# -gt 0 ]; then
  STEPS=("$@")
else
  mt=$(gcloud compute instances describe "$NAME" --project="$PROJECT" --zone="$ZONE" \
    --format='value(machineType.basename())')
  case "$mt" in
    ct6e-standard-8t) STEPS=(smoke tp8_ep dp2_tp4_ep) ;;
    ct6e-standard-4t) STEPS=(smoke tp4_ep) ;;
    *) echo "No default steps for $mt; pass them explicitly." >&2; exit 2 ;;
  esac
fi

gcloud compute scp --project="$PROJECT" --zone="$ZONE" \
  "$REPO/scripts/config.sh" "$REPO"/scripts/vm/*.sh "$NAME":~/afd-scripts/

run_step() {
  local step=$1 state
  local status=$VM_WORKDIR/logs/$step.status out=$VM_WORKDIR/logs/$step.out
  state=$("${SSH[@]}" "sudo cat $status 2>/dev/null || echo NONE")
  case "$state" in
    DONE) echo "== $step already DONE, skipping"; return 0 ;;
    RUNNING) echo "== $step already running, watching" ;;
    *) echo "== starting $step"
       "${SSH[@]}" "sudo mkdir -p $VM_WORKDIR/logs && sudo setsid nohup bash -c \"bash \$HOME/afd-scripts/vm_baseline.sh $step > $out 2>&1\" < /dev/null > /dev/null 2>&1 &"
       sleep 10 ;;
  esac
  while true; do
    state=$("${SSH[@]}" "sudo cat $status 2>/dev/null || echo NONE")
    echo "-- $(date +%T) $step $state | $("${SSH[@]}" "sudo tail -n 1 $out 2>/dev/null | cut -c1-160" || true)"
    case "$state" in
      DONE) return 0 ;;
      FAILED|NONE)
        echo "$step ended as $state. Last lines of its log and the server log:" >&2
        "${SSH[@]}" "sudo tail -n 30 $out; sudo tail -n 60 $VM_WORKDIR/logs/server_$step.log" >&2 || true
        return 1 ;;
    esac
    sleep 60
  done
}

for step in "${STEPS[@]}"; do
  run_step "$step"
done

mkdir -p "$REPO/results"
gcloud storage rsync -r "$BUCKET/$BUCKET_PREFIX/results/" "$REPO/results/"
if [ -d "$REPO/results/colocated" ]; then
  python3 "$REPO/scripts/summarize.py" "$REPO/results/colocated"
  for d in "$REPO"/results/colocated/*/; do
    python3 "$REPO/scripts/workload_card.py" "${d%/}"
  done
fi
echo "Results are in $BUCKET/$BUCKET_PREFIX/ and results/. Next: scripts/teardown.sh $NAME $ZONE"
