#!/usr/bin/env bash
# Hands-off run on the TPU VM (started by startup.sh on --auto VMs):
# setup, each benchmark step, cache upload, then afd-cleanup (upload results,
# profile, logs; delete this VM).
#
# Progress: gs://<bucket>/<prefix>/runs/<run id>/status.txt
# startup.sh already armed a deadline timer that runs afd-cleanup --force, so a
# hang here still ends before --max-run-duration.
set -uo pipefail
cd "$(dirname "$0")"
source ./config.sh

md() { curl -sf --max-time 10 -H 'Metadata-Flavor: Google' \
  "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
NAME=$(md name)
ZONE=$(md zone); ZONE=${ZONE##*/}
STEPS=$(md attributes/afd-steps)
RUN_ID=$(md attributes/afd-run-id)
STATUS_LOG=$VM_WORKDIR/logs/status.txt
mkdir -p "$VM_WORKDIR/logs"
echo "$RUN_ID" > "$VM_WORKDIR/run_id"

status() {
  echo "$(date '+%F %T %Z') $*" | tee -a "$STATUS_LOG"
  timeout 120 gcloud storage cp "$STATUS_LOG" "$RUNS_URI/$RUN_ID/status.txt" --quiet >/dev/null 2>&1 || true
}

finish() {
  if [ "$KEEP_CACHE" = true ]; then
    status "uploading compile cache"
    timeout 1500 gcloud storage rsync -r "$VM_WORKDIR/cache/" "$CACHE_URI/" --quiet \
      || status "WARN cache upload failed or timed out"
  fi
  status "final upload, then deleting VM"
  /usr/local/bin/afd-cleanup
}
trap finish EXIT

status "VM $NAME up in $ZONE; run $RUN_ID; steps: $STEPS"
if ! timeout 2700 bash ./vm_setup.sh > "$VM_WORKDIR/logs/setup.out" 2>&1; then
  status "FAILED setup (see logs/setup.out)"
  exit 1
fi

for step in $STEPS; do
  status "running $step"
  # Cold start (download + compile) is bounded inside vm_baseline.sh; this is
  # a backstop for anything else hanging.
  case "$step" in afd_xfer|afd_jax) script=./vm_afd.sh ;; *) script=./vm_baseline.sh ;; esac
  if timeout 7200 bash "$script" "$step" > "$VM_WORKDIR/logs/$step.out" 2>&1; then
    status "DONE $step"
  else
    status "FAILED $step (see logs/$step.out and logs/server_$step.log)"
    # A failed smoke test means the setup is broken; skip the long runs.
    [ "$step" = smoke ] && exit 1
  fi
done
status "ALL STEPS FINISHED"
