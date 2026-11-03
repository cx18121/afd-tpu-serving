# Shared settings for the colocated baseline. Sourced by the other scripts.
# Change the baseline here, not in the individual scripts.

PROJECT="${PROJECT:-cs5470-fa26-team27}"
BUCKET="${BUCKET:-gs://cs5470-fa26-team27-data}"
BUCKET_PREFIX="${BUCKET_PREFIX:-colocated}"

# Pinned so reruns get the same build. v0.32.0 was :latest on 2026-11-02.
IMAGE="${IMAGE:-vllm/vllm-torchtpu:v0.32.0@sha256:2c7a81cd62716779b1a4491e11c2758fa2e76196b40178260c85115e4ff7298e}"

# Project-wide chip limits from the course rules.
MAX_V6E_CHIPS="${MAX_V6E_CHIPS:-8}"

# Directory on the VM that holds the compile cache, HF weights, results, and logs.
VM_WORKDIR="${VM_WORKDIR:-/opt/afd}"

# Each run's results, profiles, logs, and status go under runs/<run id>/ so
# runs never overwrite each other. The compile cache is shared across runs.
RUNS_URI="$BUCKET/$BUCKET_PREFIX/runs"
CACHE_URI="$BUCKET/$BUCKET_PREFIX/cache"

# Write the settings in effect (including environment overrides) to FILE, so
# the VM uses the same values as the laptop that launched it.
write_config_snapshot() {
  {
    echo "# Snapshot of scripts/config.sh values, written $(date '+%F %T %Z')"
    for v in PROJECT BUCKET BUCKET_PREFIX IMAGE MAX_V6E_CHIPS VM_WORKDIR; do
      printf '%s=%q\n' "$v" "${!v}"
    done
    echo 'RUNS_URI="$BUCKET/$BUCKET_PREFIX/runs"'
    echo 'CACHE_URI="$BUCKET/$BUCKET_PREFIX/cache"'
    echo 'vm_run_id() { cat "$VM_WORKDIR/run_id"; }'
  } > "$1"
}

# On the VM: the run id chosen when the run was set up.
vm_run_id() { cat "$VM_WORKDIR/run_id"; }
