#!/usr/bin/env bash
# Prepare a running TPU VM: install Docker if missing, pull the pinned image,
# restore the compile cache from the bucket, and copy the on-VM scripts over.
#
#   scripts/setup_vm.sh NAME ZONE
set -euo pipefail
source "$(dirname "$0")/config.sh"

if [ $# -ne 2 ]; then
  echo "usage: $0 NAME ZONE" >&2
  exit 2
fi
NAME=$1 ZONE=$2
SSH=(gcloud compute ssh "$NAME" --project="$PROJECT" --zone="$ZONE")

"${SSH[@]}" --command="mkdir -p ~/afd-scripts"
gcloud compute scp --project="$PROJECT" --zone="$ZONE" \
  "$(dirname "$0")/config.sh" "$(dirname "$0")"/vm/*.sh "$NAME":~/afd-scripts/

"${SSH[@]}" --command="sudo bash ~/afd-scripts/vm_setup.sh"
