#!/bin/bash
# GCE startup script for VMs created with `create_vm.sh ... --auto`.
# Google runs it as root at every boot. It fetches this run's scripts from the
# bucket and hands off to autorun.sh, which runs the baseline and deletes the VM.
md() { curl -sf -H 'Metadata-Flavor: Google' \
  "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }

# Run once, even if the VM reboots.
[ -e /var/lib/afd-autorun-started ] && exit 0
touch /var/lib/afd-autorun-started

DIR=/root/afd-scripts
mkdir -p "$DIR"
URI=$(md attributes/afd-scripts-uri)
for i in 1 2 3 4 5; do
  gcloud storage cp "$URI/*" "$DIR/" && break
  sleep 15
done
exec bash "$DIR/autorun.sh" >> /var/log/afd-autorun.log 2>&1
