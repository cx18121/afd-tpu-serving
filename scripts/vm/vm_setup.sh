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

timeout 1800 docker pull "$IMAGE"

mkdir -p "$VM_WORKDIR"/{cache,hf,results,profile,logs}

# Run id: names this run's folder in the bucket (runs/<run id>/). Auto mode
# writes it before setup; otherwise make one from the date and host name.
[ -s "$VM_WORKDIR/run_id" ] || echo "$(date +%Y%m%d-%H%M%S)-$(hostname)" > "$VM_WORKDIR/run_id"
echo "Run id: $(cat "$VM_WORKDIR/run_id")"

# Restore the compile cache from earlier runs, if any.
if [ "$KEEP_CACHE" != true ]; then
  echo "KEEP_CACHE=false: compiling from scratch (about 13-16 minutes per layout)."
elif gcloud storage ls "$CACHE_URI/" >/dev/null 2>&1; then
  echo "Restoring compile cache from $CACHE_URI/"
  timeout 1200 gcloud storage rsync -r "$CACHE_URI/" "$VM_WORKDIR/cache/" \
    || echo "WARN: cache restore failed or timed out; compiling from scratch"
else
  echo "No compile cache in the bucket yet; the first start will compile from scratch."
fi

df -h / | tail -1
echo "Setup done."
