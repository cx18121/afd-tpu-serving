#!/usr/bin/env bash
# Find TPU v6e Spot capacity and start a hands-off (--auto) run there.
#
#   scripts/launch_spot.sh CHIPS HOURS [STEPS]
#   scripts/launch_spot.sh 8 4                     # smoke tp8_ep dp2_tp4_ep
#   scripts/launch_spot.sh 1 1 smoke               # cheap end-to-end check
#
# Spot requests start at once or fail at once, so this tries one zone after
# another (first the zones that had capacity before) until a VM starts, then
# follows it with watch_run.sh until it deletes itself. Each failed try costs
# nothing. Needs this machine awake and gcloud logged in while it searches;
# once a VM starts, the VM finishes and deletes itself on its own.
set -uo pipefail
source "$(dirname "$0")/config.sh"
HERE=$(dirname "$0")

if [ $# -lt 2 ]; then
  echo "usage: $0 CHIPS HOURS [STEPS]" >&2
  exit 2
fi
CHIPS=$1 HOURS=$2 STEPS=${3:-}

# v6e zones with Spot quota for this project (us-east4 has 0). The first two
# had capacity on 2026-11-05 when the program zones were out of stock.
ZONES=(asia-south1-c asia-south1-b us-east5-a us-east5-b us-east5-c
  us-central1-a us-central1-b us-central1-c us-east1-b us-east1-d us-west1-c
  europe-west4-a europe-west4-b asia-northeast1-b asia-east1-c
  asia-southeast1-b southamerica-east1-c southamerica-west1-a)
ROUNDS=${ROUNDS:-20}

for round in $(seq 1 "$ROUNDS"); do
  for z in "${ZONES[@]}"; do
    name="${USER:-afd}-spot-$z"
    log=$(mktemp)
    AFD_STEPS=$STEPS "$HERE/create_vm.sh" "$name" "$z" "$CHIPS" "$HOURS" --auto --spot > "$log" 2>&1
    rc=$?
    case $rc in
      0)
        run_id=$(sed -n 's/^Auto mode: run \([^,]*\),.*/\1/p' "$log")
        echo "$(date +%T) started $name in $z (run $run_id)"
        rm -f "$log"
        exec "$HERE/watch_run.sh" "$name" "$z" "$run_id"
        ;;
      3) echo "$(date +%T) round $round $z: stockout" ;;
      *)
        echo "$(date +%T) round $round $z: $(grep -E 'Refusing|ERROR|Quota' "$log" | head -1 | cut -c1-160)"
        # Refused by our chip limit or an expired login: stop rather than spin.
        grep -qE 'Refusing|eauthentication' "$log" && { rm -f "$log"; exit 1; }
        ;;
    esac
    rm -f "$log"
  done
  echo "$(date +%T) no capacity in any zone this round; trying again in 2 minutes"
  sleep 120
done
echo "Gave up after $ROUNDS rounds."
exit 1
