#!/usr/bin/env bash
# Request a flex-start TPU v6e VM and wait until it is RUNNING.
#
#   scripts/create_vm.sh NAME ZONE CHIPS HOURS
#   scripts/create_vm.sh charlie-v6e us-east5-a 4 4
#
# CHIPS is 1, 4, or 8 (ct6e-standard-{1,4,8}t). HOURS sets --max-run-duration;
# the VM deletes itself (and its disk) when that runs out, so upload results
# with teardown.sh before then. If a zone has no capacity, try us-east5-b,
# then us-central1-a.
set -euo pipefail
source "$(dirname "$0")/config.sh"

if [ $# -ne 4 ]; then
  echo "usage: $0 NAME ZONE CHIPS HOURS" >&2
  exit 2
fi
NAME=$1 ZONE=$2 CHIPS=$3 HOURS=$4

case "$CHIPS" in
  1|4|8) MACHINE_TYPE="ct6e-standard-${CHIPS}t" ;;
  *) echo "CHIPS must be 1, 4, or 8 (got $CHIPS)" >&2; exit 2 ;;
esac
if ! [[ "$HOURS" =~ ^[0-9]+$ ]] || [ "$HOURS" -lt 1 ] || [ "$HOURS" -gt 8 ]; then
  echo "HOURS must be a whole number from 1 to 8 (got $HOURS)" >&2
  exit 2
fi

# Project-wide limit: count v6e chips in every non-terminated instance.
in_use=0
while read -r mt; do
  [[ "$mt" =~ ^ct6e-standard-([0-9]+)t$ ]] && in_use=$((in_use + BASH_REMATCH[1]))
done < <(gcloud compute instances list --project="$PROJECT" \
  --filter='status!=TERMINATED' --format='value(machineType.basename())')
if [ $((in_use + CHIPS)) -gt "$MAX_V6E_CHIPS" ]; then
  echo "Refusing: $in_use v6e chips already requested or running, +$CHIPS would exceed $MAX_V6E_CHIPS." >&2
  gcloud compute instances list --project="$PROJECT"
  exit 1
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
  --async

# Flex-start VMs stay PENDING until chips free up (up to the 2h request window).
while true; do
  status=$(gcloud compute instances describe "$NAME" --project="$PROJECT" --zone="$ZONE" \
    --format='value(status)' 2>/dev/null || echo GONE)
  echo "$(date +%T) $status"
  case "$status" in
    RUNNING) break ;;
    PENDING|PROVISIONING|STAGING|"") sleep 30 ;;
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
