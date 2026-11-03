#!/usr/bin/env bash
# Hands-off run on the TPU VM (started by startup.sh on --auto VMs):
# setup, each benchmark step, upload everything, then delete this VM.
#
# Progress goes to gs://<bucket>/<prefix>/status/<vm-name>.txt.
# Whatever happens, the exit trap uploads results and logs and deletes the
# VM, so a failure does not leave chips idle until --max-run-duration.
set -uo pipefail
cd "$(dirname "$0")"
source ./config.sh

md() { curl -sf -H 'Metadata-Flavor: Google' \
  "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
NAME=$(md name)
ZONE=$(md zone); ZONE=${ZONE##*/}
STEPS=$(md attributes/afd-steps)
SELF_DELETE=$(md attributes/afd-self-delete || echo true)
DEST=$BUCKET/$BUCKET_PREFIX
mkdir -p "$VM_WORKDIR/logs"
STATUS_LOG=$VM_WORKDIR/logs/status.txt
STATUS_URI=$DEST/status/$NAME.txt

status() {
  echo "$(date "+%F %T %Z") $*" | tee -a "$STATUS_LOG"
  gcloud storage cp "$STATUS_LOG" "$STATUS_URI" --quiet >/dev/null 2>&1 || true
}

finish() {
  status "uploading results, profile, logs, cache"
  docker rm -f afd-vllm >/dev/null 2>&1 || true
  cp /var/log/afd-autorun.log "$VM_WORKDIR/logs/autorun_$NAME.log" 2>/dev/null || true
  for d in results profile logs cache; do
    if [ -d "$VM_WORKDIR/$d" ] && [ -n "$(ls -A "$VM_WORKDIR/$d")" ]; then
      gcloud storage rsync -r "$VM_WORKDIR/$d/" "$DEST/$d/" --quiet || status "WARN upload of $d failed"
    fi
  done
  if [ "$SELF_DELETE" = true ]; then
    status "deleting VM $NAME"
    gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet
  else
    status "finished; afd-self-delete=false, VM left running"
  fi
}
trap finish EXIT

status "VM $NAME up in $ZONE; steps: $STEPS"
if ! bash ./vm_setup.sh > "$VM_WORKDIR/logs/setup.out" 2>&1; then
  status "FAILED setup (see logs/setup.out)"
  exit 1
fi

for step in $STEPS; do
  status "running $step"
  if bash ./vm_baseline.sh "$step" > "$VM_WORKDIR/logs/$step.out" 2>&1; then
    status "DONE $step"
  else
    status "FAILED $step (see logs/$step.out and logs/server_$step.log)"
    # A failed smoke test means the setup is broken; skip the long runs.
    [ "$step" = smoke ] && exit 1
  fi
done
status "ALL STEPS FINISHED"
