#!/usr/bin/env bash
# Runs on the TPU VM as root (called by autorun.sh for steps named afd_*).
#
#   vm_afd.sh afd_xfer    AFD transfer feasibility test (afd/transfer_test.py)
#
# Writes $VM_WORKDIR/results/<step>/ and uploads it to the run folder.
set -Eeuo pipefail
source "$(dirname "$0")/config.sh"

STEP=${1:?usage: vm_afd.sh afd_xfer}
W=$VM_WORKDIR
STATUS_FILE=$W/logs/$STEP.status
mkdir -p "$W/logs" "$W/results/$STEP"
echo RUNNING > "$STATUS_FILE"
trap 'echo FAILED > "$STATUS_FILE"' ERR
DEST=$RUNS_URI/$(vm_run_id)

case "$STEP" in
  afd_xfer)
    docker rm -f afd-test >/dev/null 2>&1 || true
    # One process per chip, as in serving. Each design runs in its own
    # torchrun with a timeout, so a hang shows up as TIMEOUT, not a stuck VM.
    timeout 5400 docker run --rm --name afd-test \
      --privileged --net=host -v /dev/shm:/dev/shm --shm-size 150gb \
      -v "$(dirname "$0")/transfer_test.py:/afd/transfer_test.py:ro" \
      -v "$W/results/$STEP:/out" \
      -e PYTHONUNBUFFERED=1 \
      --entrypoint python3 "$IMAGE" \
      /afd/transfer_test.py --launch --platform tpu --out /out \
        --designs "${AFD_DESIGNS:-homo,hetero_ordered}" \
        --splits 4,2 --tokens 1,8,32,128 --timeout 300 \
      2>&1 | tee "$W/logs/$STEP.log"
    ;;
  *) echo "unknown AFD step $STEP" >&2; exit 2 ;;
esac

timeout 600 gcloud storage rsync -r "$W/results/" "$DEST/results/" --quiet || echo "WARN: upload failed"
timeout 600 gcloud storage rsync -r "$W/logs/" "$DEST/logs/" --quiet || true
echo DONE > "$STATUS_FILE"
