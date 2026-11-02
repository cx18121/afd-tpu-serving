# Shared settings for the colocated baseline. Sourced by the other scripts.
# Change the baseline here, not in the individual scripts.

PROJECT="${PROJECT:-cs5470-fa26-team27}"
BUCKET="${BUCKET:-gs://cs5470-fa26-team27-data}"
BUCKET_PREFIX="${BUCKET_PREFIX:-colocated}"

# Pinned so reruns get the same build. v0.32.0 was :latest on 2026-11-02.
IMAGE="${IMAGE:-vllm/vllm-torchtpu:v0.32.0@sha256:2c7a81cd62716779b1a4491e11c2758fa2e76196b40178260c85115e4ff7298e}"

# Project-wide chip limits from the course rules.
MAX_V6E_CHIPS=8

# Directory on the VM that holds the compile cache, HF weights, results, and logs.
VM_WORKDIR="${VM_WORKDIR:-/opt/afd}"
