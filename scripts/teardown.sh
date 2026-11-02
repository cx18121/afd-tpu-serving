#!/usr/bin/env bash
# Upload results, profile traces, logs, and the compile cache to the bucket,
# delete the VM, and list what is still running in the project.
#
#   scripts/teardown.sh NAME ZONE
#
# Flex-start VMs cannot be stopped and their disks are deleted with them, so
# anything not uploaded here is lost.
set -euo pipefail
source "$(dirname "$0")/config.sh"

if [ $# -ne 2 ]; then
  echo "usage: $0 NAME ZONE" >&2
  exit 2
fi
NAME=$1 ZONE=$2
DEST=$BUCKET/$BUCKET_PREFIX

if gcloud compute instances describe "$NAME" --project="$PROJECT" --zone="$ZONE" \
     --format='value(status)' 2>/dev/null | grep -qx RUNNING; then
  echo "Uploading from $NAME to $DEST/"
  if ! gcloud compute ssh "$NAME" --project="$PROJECT" --zone="$ZONE" --command "
      set -e
      sudo docker rm -f afd-vllm >/dev/null 2>&1 || true
      for d in results profile logs cache; do
        if [ -d $VM_WORKDIR/\$d ] && [ -n \"\$(sudo ls -A $VM_WORKDIR/\$d)\" ]; then
          sudo gcloud storage rsync -r $VM_WORKDIR/\$d/ $DEST/\$d/ --quiet
        fi
      done
      sudo du -sh $VM_WORKDIR/* 2>/dev/null"; then
    read -r -p "Upload failed. Delete $NAME anyway? [y/N] " ans
    [ "$ans" = y ] || exit 1
  fi
fi

gcloud compute instances delete "$NAME" --project="$PROJECT" --zone="$ZONE" --quiet || true
echo "Instances still in project $PROJECT:"
gcloud compute instances list --project="$PROJECT"
gcloud storage du -s "$DEST/" 2>/dev/null || true
