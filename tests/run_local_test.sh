#!/usr/bin/env bash
# Run the real scripts/vm/vm_baseline.sh without a TPU: a "fake VM" container
# drives Docker on this machine, the pinned vllm-torchtpu image runs
# tests/mock_openai_server.py instead of vllm serve, and the real
# `vllm bench serve` client benchmarks it. gcloud is stubbed, so nothing is
# uploaded. Then summarize.py and workload_card.py run on the output.
#
#   tests/run_local_test.sh                  # needs Docker
#   MOCK_GARBAGE=1 tests/run_local_test.sh   # expect the sanity check to fail
#
# On Apple silicon the image runs emulated without AVX, and the TPU plugin's
# libraries need AVX, so the run gets through server startup and the sanity
# check, then stops at the first `vllm bench serve` (exit 132). The benchmark
# part needs an x86 machine with AVX.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
T="$REPO/.localtest"
rm -rf "$T"
mkdir -p "$T/work" "$T/bin"

cat > "$T/bin/gcloud" <<'EOF'
#!/bin/bash
echo "gcloud $*" >> "$VM_WORKDIR/gcloud_calls.log"
EOF
chmod +x "$T/bin/gcloud"

# The server container does not see this script's environment, so garbage
# mode gets its own copy of the mock server.
MOCK="$REPO/tests/mock_openai_server.py"
if [ "${MOCK_GARBAGE:-}" = 1 ]; then
  sed 's/^GARBAGE = .*/GARBAGE = True/' "$MOCK" > "$T/mock_garbage.py"
  MOCK="$T/mock_garbage.py"
fi

# Settings the real VM gets from config.sh, pointed at the test directory.
cat > "$T/env.sh" <<EOF
export VM_WORKDIR="$T/work"
export AFD_MOCK_SERVER="$MOCK"
export CONCURRENCIES="4" NUM_PROMPTS=8 WARMUPS="1:2"
export PATH="$T/bin:\$PATH"
EOF
echo "test-run" > "$T/work/run_id"

# Stage the scripts the way the VM has them: config.sh next to the vm/ scripts.
mkdir -p "$T/scripts"
cp "$REPO"/scripts/vm/*.sh "$T/scripts/"
VM_WORKDIR="$T/work" bash -c "source '$REPO/scripts/config.sh'; write_config_snapshot '$T/scripts/config.sh'"

# The fake VM uses the same absolute paths as this machine, because the Docker
# daemon resolves the -v paths in vm_baseline.sh on this machine.
docker run --rm --net=host \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$REPO:$REPO" -w "$REPO" \
  docker:cli sh -c "
    apk add -q bash curl python3 coreutils >/dev/null
    source $T/env.sh
    bash $T/scripts/vm_baseline.sh tp8_ep
  " 2>&1 | tee "$T/run.log"
status=${PIPESTATUS[0]}

echo "== status file: $(cat "$T/work/logs/tp8_ep.status" 2>/dev/null)"
echo "== uploads that would have happened:"
sort -u "$T/work/gcloud_calls.log" 2>/dev/null | sed "s#$T#<test>#g"
if [ -d "$T/work/results/colocated/tp8_ep" ]; then
  python3 "$REPO/scripts/summarize.py" "$T/work/results/colocated"
  python3 "$REPO/scripts/workload_card.py" "$T/work/results/colocated/tp8_ep" test-run
fi
exit "$status"
