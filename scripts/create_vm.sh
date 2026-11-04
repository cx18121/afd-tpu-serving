#!/usr/bin/env bash
# Request a flex-start TPU VM (v6e or v5p) and wait until it is RUNNING.
#
#   scripts/create_vm.sh NAME ZONE CHIPS HOURS [--auto]
#   scripts/create_vm.sh charlie-v6e us-east5-a 8 4
#   scripts/create_vm.sh charlie-v6e us-east5-a 8 4 --auto
#
# Without --auto, it waits until the VM is RUNNING; then run setup_vm.sh,
# run_baseline.sh, and teardown.sh yourself.
# With --auto, the VM runs everything itself once it starts (setup, smoke test,
# every layout for its chip count, uploads) and then deletes itself. The script
# returns right after the request is queued; you can close your laptop.
# Progress: gcloud storage cat gs://<bucket>/colocated/runs/<run id>/status.txt
# Override the steps with AFD_STEPS="smoke tp8_ep"; keep the VM after the run
# with AFD_SELF_DELETE=false (then delete it yourself).
#
# CHIPS is 1, 4, or 8 for TPU v6e (ct6e-standard-{1,4,8}t), or v5p-4 for one
# TPU v5p host (ct5p-hightpu-4t, 4 chips, 95 GB HBM each). HOURS sets --max-run-duration;
# the VM deletes itself (and its disk) when that runs out, so upload results
# with teardown.sh before then. If a zone has no capacity, try us-east5-b,
# then us-central1-a.
set -euo pipefail
source "$(dirname "$0")/config.sh"

if [ $# -lt 4 ] || [ $# -gt 5 ] || { [ $# -eq 5 ] && [ "$5" != --auto ]; }; then
  echo "usage: $0 NAME ZONE CHIPS HOURS [--auto]" >&2
  exit 2
fi
NAME=$1 ZONE=$2 CHIPS=$3 HOURS=$4 AUTO=${5:-}

case "$CHIPS" in
  1|4|8) FAMILY=v6e N=$CHIPS MACHINE_TYPE="ct6e-standard-${CHIPS}t" ;;
  v5p-4) FAMILY=v5p N=4 MACHINE_TYPE=ct5p-hightpu-4t ;;
  *) echo "CHIPS must be 1, 4, 8 (v6e) or v5p-4 (got $CHIPS)" >&2; exit 2 ;;
esac
if ! [[ "$HOURS" =~ ^[0-9]+$ ]] || [ "$HOURS" -lt 1 ] || [ "$HOURS" -gt 8 ]; then
  echo "HOURS must be a whole number from 1 to 8 (got $HOURS)" >&2
  exit 2
fi

# Project-wide limits: count chips of this TPU family in every non-terminated
# instance, including PENDING requests. Refuse if the listing itself fails.
# This does not stop two people who submit at the same moment; coordinate in
# the team chat.
if ! listing=$(gcloud compute instances list --project="$PROJECT" \
    --filter='status!=TERMINATED' --format='value(machineType.basename())'); then
  echo "Refusing: could not list instances to check the chip limits." >&2
  exit 1
fi
in_use=0
while read -r mt; do
  case "$FAMILY:$mt" in
    v6e:ct6e-standard-*t) n=${mt#ct6e-standard-}; in_use=$((in_use + ${n%t})) ;;
    v5p:ct5p-hightpu-4t) in_use=$((in_use + 4)) ;;
  esac
done <<< "$listing"
LIMIT=$([ "$FAMILY" = v5p ] && echo "$MAX_V5P_CHIPS" || echo "$MAX_V6E_CHIPS")
if [ $((in_use + N)) -gt "$LIMIT" ]; then
  echo "Refusing: $in_use $FAMILY chips already requested or running, +$N would exceed $LIMIT." >&2
  gcloud compute instances list --project="$PROJECT"
  exit 1
fi

AUTO_ARGS=()
if [ "$AUTO" = --auto ]; then
  case "$CHIPS" in
    8) default_steps="smoke tp8_ep dp2_tp4_ep" ;;
    4|v5p-4) default_steps="smoke tp4_ep" ;;
    1) default_steps="smoke" ;;
  esac
  STEPS=${AFD_STEPS:-$default_steps}
  RUN_ID="$(date +%Y%m%d-%H%M%S)-$NAME"
  RUN_URI="$RUNS_URI/$RUN_ID"
  # The VM runs afd-cleanup --force at this point, before the hard cap, so
  # results are uploaded and the VM deleted even if a step hangs.
  DEADLINE_MIN=$((HOURS * 60 - 25))
  # Snapshot the scripts and effective settings for the VM to fetch.
  HERE=$(dirname "$0")
  snap=$(mktemp -d)
  write_config_snapshot "$snap/config.sh"
  gcloud storage cp "$snap/config.sh" "$HERE"/vm/*.sh "$RUN_URI/scripts/" --quiet
  rm -rf "$snap"
  AUTO_ARGS=(--metadata-from-file="startup-script=$HERE/vm/startup.sh"
    --metadata="afd-run-id=$RUN_ID,afd-run-uri=$RUN_URI,afd-steps=$STEPS,afd-deadline-min=$DEADLINE_MIN,afd-workdir=$VM_WORKDIR,afd-self-delete=${AFD_SELF_DELETE:-true}")
  echo "Auto mode: run $RUN_ID, steps '$STEPS', forced cleanup after ${DEADLINE_MIN} min"
fi

echo "Requesting $NAME: $MACHINE_TYPE in $ZONE, FLEX_START, max run ${HOURS}h"
gcloud compute instances create "$NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --provisioning-model=FLEX_START \
  --request-valid-for-duration=2h \
  --max-run-duration="${HOURS}h" \
  --instance-termination-action=DELETE \
  --maintenance-policy=TERMINATE \
  --image-project=ubuntu-os-accelerator-images \
  --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
  --boot-disk-size=200GB \
  --scopes=cloud-platform \
  ${AUTO_ARGS[@]+"${AUTO_ARGS[@]}"} \
  --async

if [ "$AUTO" = --auto ]; then
  echo "Queued. The VM runs and deletes itself once it starts; nothing else to do."
  echo "Request state: gcloud compute instances describe $NAME --zone=$ZONE --format='value(status)'"
  echo "Progress once running: gcloud storage cat $RUN_URI/status.txt"
  echo "Results when done: scripts/fetch_results.sh $RUN_ID"
  exit 0
fi

# Flex-start VMs stay PENDING until chips free up (up to the 2h request window).
while true; do
  # Only "not found" means the request is gone; other errors (network, expired
  # login) are transient, so keep waiting.
  if ! status=$(gcloud compute instances describe "$NAME" --project="$PROJECT" --zone="$ZONE" \
      --format='value(status)' 2>&1); then
    case "$status" in *"was not found"*) status=GONE ;; *) status=UNKNOWN ;; esac
  fi
  echo "$(date +%T) $status"
  case "$status" in
    RUNNING) break ;;
    PENDING|PROVISIONING|STAGING|UNKNOWN|"") sleep 30 ;;
    *) echo "VM did not start (status $status)." >&2; exit 1 ;;
  esac
done

# SSH is usually ready a minute or so after RUNNING.
for _ in $(seq 1 20); do
  if gcloud compute ssh "$NAME" --project="$PROJECT" --zone="$ZONE" --command=true -- -o ConnectTimeout=10 >/dev/null 2>&1; then
    echo "$NAME is RUNNING and reachable over SSH."
    exit 0
  fi
  sleep 15
done
echo "$NAME is RUNNING but SSH is not answering yet; retry setup_vm.sh in a minute." >&2
