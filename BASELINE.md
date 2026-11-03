# Colocated baseline: Qwen3-30B-A3B on TPU v6e

This is the "before" point for the AFD study. Attention and experts share the
same 8 TPU v6e chips. AFD will be compared against it at the same model,
workload, and chip budget.

> Status: scripts ready, waiting for TPU capacity. Results, versions, and
> problems below are filled in after the first run.

## Run it

You need `gcloud` logged in with access to project `cs5470-fa26-team27`.
Ask the team before creating a VM: the project allows at most 8 v6e chips at
once, and an 8-chip VM uses all of them.

### Hands-off (`--auto`)

The VM runs everything itself once it starts and deletes itself at the end,
so you can close your laptop right after the request is queued.

```bash
scripts/create_vm.sh charlie-v6e us-east5-a 8 4 --auto      # queue; returns at once
gcloud storage cat gs://cs5470-fa26-team27-data/colocated/status/charlie-v6e.txt   # progress
scripts/fetch_results.sh                                     # after "deleting VM" appears
gcloud compute instances list                                # confirm nothing is left
```

If any step fails, the VM still uploads its logs and deletes itself. A failed
smoke test skips the long runs. A flex-start request waits in the queue for
at most 2 hours (Google's limit); if it expires, run the same command again.

### Step by step (for debugging)

```bash
scripts/create_vm.sh charlie-v6e us-east5-a 8 4   # NAME ZONE CHIPS HOURS; waits until RUNNING
scripts/setup_vm.sh  charlie-v6e us-east5-a       # Docker, pinned image, compile cache restore
scripts/run_baseline.sh charlie-v6e us-east5-a    # smoke test, both layouts, profile, upload
scripts/teardown.sh  charlie-v6e us-east5-a       # upload results + cache, delete VM, list instances
```

This mode needs your laptop awake until teardown: if it sleeps after the VM
starts, the VM sits idle until its max run duration.

If a zone has no capacity, the request stays PENDING. Delete it and try
us-east5-b, then us-central1-a. Keep only one request open at a time so two
VMs cannot start together and exceed the chip limit.

`run_baseline.sh` takes about 3 hours on a cold cache. It runs detached on the
VM, so closing your laptop does not stop it; rerun the same command to resume
watching. Each finished benchmark is uploaded at once, so results survive even
if the VM hits its max run duration.

## What it measures

| | |
|---|---|
| Model | `Qwen/Qwen3-30B-A3B`, bf16 (48 layers, 128 experts, 8 active per token, 4 KV heads) |
| Hardware | 1 x `ct6e-standard-8t` (8 TPU v6e chips, 32 GB HBM each), FLEX_START |
| Workload | `vllm bench serve`, random dataset, 1024 input / 256 output tokens, 200 prompts, `--ignore-eos`, temperature 0, seed 42 |
| Concurrency | 1, 4, 8, 16, 32, 64, 128 (one Pareto point each) |
| Layouts | `tp8_ep`: TP=8, EP=8. `dp2_tp4_ep`: 2 replicas of TP=4, EP=8 across both |

Two layouts give a stronger colocated frontier: AFD should beat the best
colocated setup, not a weak one. The 4-chip layout `tp4_ep` runs on a
`ct6e-standard-4t` with the same scripts.

## Server settings

From the vllm-torchtpu v6e recipe, with the changes marked:

```
--enable-expert-parallel --attention-backend CUSTOM
--max-model-len 4096 --max-num-batched-tokens 4096
--max-num-seqs 128                  # changed: covers c=128, skips compiling 256/512 batch buckets
--gpu-memory-utilization 0.92 --async-scheduling --no-enable-prefix-caching
                                    # changed: no --kv-cache-dtype fp8 (see bug #2435 below)
```

The exact command for each run is saved as `results/colocated/<layout>/serve_cmd.txt`.

Container flags follow the vllm-torchtpu Docker guide: `--privileged
--net=host -v /dev/shm:/dev/shm --shm-size 150gb`.

## Known issues that shaped the setup

- **CUSTOM attention + fp8 KV cache decodes garbage for this model**
  ([#2435](https://github.com/vllm-project/vllm-torchtpu/issues/2435), open).
  The kernel is wrong when 2 x (KV heads per chip) differs from the KV
  packing (2 for bf16, 4 for fp8). Qwen3-30B-A3B has 4 KV heads, so each
  chip holds 1 at TP=4 and TP=8. With fp8 that is wrong; with bf16 it is
  correct. bf16 KV still leaves far more KV space than c=128 needs. The run
  checks greedy replies with known answers before benchmarking
  (`sanity.json`).
- **Smoke test uses the default attention backend.** Qwen3-0.6B at TP=1 has 8
  KV heads on one chip, another layout #2435 breaks.
- **`VLLM_XLA_CACHE_PATH` does nothing.** It is in the docs' env-var table,
  but the v0.32.0 source never reads it. The compile cache is controlled by
  `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` (compiled TPU binaries) and
  `VLLM_CACHE_ROOT` (Python-side artifacts). Both point at
  `/opt/afd/cache` on the VM, which is synced to
  `gs://cs5470-fa26-team27-data/colocated/cache/`.
- Possibly relevant, not yet hit: concurrent cold compiles sharing one cache
  can collide (#1566, may affect the DP=2 layout), and a warm start may not
  be much faster than a cold one (#1644).

## Versions

Image pinned in `scripts/config.sh`:
`vllm/vllm-torchtpu:v0.32.0@sha256:2c7a81cd62716779b1a4491e11c2758fa2e76196b40178260c85115e4ff7298e`
(this was `:latest` on 2026-11-02). Model revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.
Package versions inside the image are recorded per run in `versions.txt`.

## Results

TBD after the first run. Full table: `results/colocated/summary.csv`.
`per_user_tok_s` is 1000 / mean TPOT, the decode speed one user sees.

## Outputs

| What | Where |
|---|---|
| Per-run benchmark JSON | `results/colocated/<layout>/colocated_c<N>.json`, also in the bucket |
| Summary | `results/colocated/summary.csv` |
| Profile traces (c=16, separate unmeasured run) | `gs://cs5470-fa26-team27-data/colocated/profile/<layout>/` |
| CCL-Bench workload card | `results/colocated/<layout>/workload_card.yaml` |
| Server and benchmark logs | `gs://cs5470-fa26-team27-data/colocated/logs/` |
| Compile cache | `gs://cs5470-fa26-team27-data/colocated/cache/` |

Profiling costs about 11% TTFT and 4% throughput (vllm-torchtpu docs), so
the profiled run is separate from the measured sweep.

## VM time log

| Date | Zone | Machine | Outcome | TPU hours |
|---|---|---|---|---|
| 2026-11-02 | us-east5-a | ct6e-standard-4t | Pending 1 h, cancelled when we switched to 8 chips | 0 |
| 2026-11-02 | us-east5-a | ct6e-standard-8t | Pending ~65 min, no capacity, cancelled | 0 |
| 2026-11-02 | us-central1-a | ct6e-standard-8t | Pending 2 h 19 min, expired (`ZONE_RESOURCE_POOL_EXHAUSTED`) | 0 |
| 2026-11-02 | us-central1-a | ct6e-standard-8t | Pending 2 h, expired (same) | 0 |
| 2026-11-03 | us-central1-a | ct6e-standard-8t | Pending ~55 min, cancelled for the night | 0 |
