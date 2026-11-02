#!/usr/bin/env bash
# Runs on the TPU VM as root (called by scripts/setup_vm.sh).
set -euo pipefail
source "$(dirname "$0")/config.sh"

ls /dev/vfio/ >/dev/null || { echo "No /dev/vfio: TPU devices not visible." >&2; exit 1; }

if ! command -v docker >/dev/null; then
  echo "Installing Docker"
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io
fi
systemctl start docker

docker pull "$IMAGE"

mkdir -p "$VM_WORKDIR"/{cache,hf,results,profile,logs}

# Restore the compile cache from earlier runs, if any.
if gcloud storage ls "$BUCKET/$BUCKET_PREFIX/cache/" >/dev/null 2>&1; then
  echo "Restoring compile cache from $BUCKET/$BUCKET_PREFIX/cache/"
  gcloud storage rsync -r "$BUCKET/$BUCKET_PREFIX/cache/" "$VM_WORKDIR/cache/"
else
  echo "No compile cache in the bucket yet; the first start will compile from scratch."
fi

df -h / | tail -1
echo "Setup done."
