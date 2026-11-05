# Colocated baseline: Qwen3-30B-A3B on TPU v6e

This is the "before" point for the AFD study. Attention and experts share the
same 8 TPU v6e chips. AFD will be compared against it at the same model,
workload, and chip budget.

> Status: first full run done on 2026-11-05 (run
> `20261105-134448-charlie-spot-asia-south1-c`, 8 v6e chips, Spot,
> asia-south1-c). Open items: repeat the suspect TP=8 c=32 point, and re-take
> the DP=2 profile, which the disk-full error cut short.

## Run it

You need `gcloud` logged in with access to project `cs5470-fa26-team27`.
Ask the team before creating a VM: the project allows at most 8 v6e chips at
once, and an 8-chip VM uses all of them.

### Hands-off (`--auto`)

The VM runs everything itself once it starts and deletes itself at the end,
so you can close your laptop right after the request is queued.

```bash
scripts/create_vm.sh charlie-v6e us-east5-a 8 4 --auto   # queue; prints the run id
gcloud storage cat gs://cs5470-fa26-team27-data/colocated/runs/<run id>/status.txt   # progress
scripts/fetch_results.sh <run id>                         # after "cleanup" appears
gcloud compute instances list                             # confirm nothing is left
```

Safeguards, all on the VM:

- A failed step does not stop later layouts. A failed smoke test skips the
  long runs.
- Each benchmark and upload has a time limit. A timer armed at boot uploads
  results and deletes the VM 25 minutes before `--max-run-duration`, even if
  a step hangs or the scripts never downloaded.
- Uploads retry 3 times. If results still cannot be uploaded, the VM is kept
  for someone to rescue them, until that timer deletes it.
- If `gcloud` cannot delete the VM, it deletes itself through the Compute REST
  API.

A flex-start request waits in the queue for at most 2 hours (Google's limit).
If it expires, run the same command again.

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
VMs cannot start together and exceed the chip limit. `create_vm.sh` refuses a
request that would exceed 8 v6e chips, counting pending requests, but two
people submitting at the same moment can both pass the check.

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
  `gs://cs5470-fa26-team27-data/colocated/cache/` (shared by all runs).
- **Getting TPUs is the hard part.** From 2026-11-02 to 11-05, every
  flex-start request in the program zones (us-east5-a, us-east5-b,
  us-central1-a; 1, 4, and 8 v6e chips; 4 v5p chips) expired with
  `ZONE_RESOURCE_POOL_EXHAUSTED` (`reason: stockout`), and about 200 Spot
  probes in those zones failed the same way. That error is capacity, not
  quota. The first VM came from a 1-chip Spot probe in **asia-south1-b**
  after probing all 20 v6e zones. Our `TPUS_PER_TPU_FAMILY` quota allows 48
  v6e chips in europe-west4, asia-east1, asia-northeast1, asia-south1,
  asia-southeast1, southamerica-east1/west1, and us-south1, and 0 in us-east4.
  Course rules still cap the project at 8 v6e chips at once.
- **First real run (Qwen3-0.6B smoke test, asia-south1-b):** setup took
  about 2 minutes, the server was healthy after 271 s (download + compile),
  and all four greedy replies were correct. The sanity check wrongly failed
  "1,2,...,30" because it expected spaces after commas; it now ignores
  spacing. The VM uploaded its logs and deleted itself as designed.
- **Disk full during the DP=2 profile run.** The compile cache for the two
  layouts reached 112 GB; with the 61 GB model, two profile traces (6 GB),
  and the image, the 200 GB disk filled. The DP=2 sweep had already finished,
  but its profile trace is incomplete. Fix pending (bigger disk or not
  keeping the cache, see below).
- Possibly relevant, not yet hit: concurrent cold compiles sharing one cache
  can collide (#1566, may affect the DP=2 layout), and a warm start may not
  be much faster than a cold one (#1644).

## Versions

Image pinned in `scripts/config.sh`:
`vllm/vllm-torchtpu:v0.32.0@sha256:2c7a81cd62716779b1a4491e11c2758fa2e76196b40178260c85115e4ff7298e`
(this was `:latest` on 2026-11-02). Model revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.
Package versions inside the image are recorded per run in `versions.txt`.

## Results

Qwen3-30B-A3B, 8 x TPU v6e, 1024 input / 256 output tokens, 200 prompts per
point, every point 200/200 completed and 0 failed. Full table:
`results/colocated/summary.csv`. Per-user tok/s is 1000 / mean TPOT, the
decode speed one user sees.

| Concurrency | TP=8+EP output tok/s | per-user tok/s | mean TTFT | DP=2xTP=4+EP output tok/s | per-user tok/s | mean TTFT |
|---|---|---|---|---|---|---|
| 1 | **183** | **188** | 39 ms | 142 | 146 | 52 ms |
| 4 | **665** | **177** | 95 ms | 556 | 148 | 122 ms |
| 8 | **1,195** | **162** | 137 ms | 1,000 | 138 | 205 ms |
| 16 | **2,116** | **153** | 200 ms | 1,729 | 126 | 273 ms |
| 32 | 1,856 (suspect) | 65 | 305 ms | **2,862** | **112** | 355 ms |
| 64 | **3,997** | 78 | 372 ms | 3,829 | **78** | 520 ms |
| 128 | 5,264 | 58 | 928 ms | **5,624** | **64** | 995 ms |

- TP=8 is faster per user up to c=16; two TP=4 replicas give more total
  throughput at c=32 and c=128. The colocated frontier uses the better layout
  at each point.
- The TP=8 c=32 point is suspect: lower throughput than c=16, and mean TPOT
  (15.3 ms) far above median (9.3 ms), so a few requests stalled. The server
  log shows no compiles and low KV use. Repeat it before using it.
- p99 inter-token latency rises to 34 to 126 ms at c >= 16 in both layouts,
  likely because new prompts (prefill) share steps with ongoing decodes.
- Server startup (weights download + compile, cold cache): 977 s for TP=8,
  781 s for DP=2xTP=4.
- Measured on Spot capacity in asia-south1-c; one run per point, so run-to-run
  variance is not yet known.

## Outputs

| What | Where |
|---|---|
| Everything from one run | `gs://cs5470-fa26-team27-data/colocated/runs/<run id>/` (results, profile, logs, status.txt) |
| Per-run benchmark JSON | `results/colocated/<layout>/colocated_c<N>.json` (copied by `fetch_results.sh`) |
| Summary | `results/colocated/summary.csv` |
| Profile traces (c=16, separate unmeasured run) | `runs/<run id>/profile/<layout>/` |
| CCL-Bench workload card | `results/colocated/<layout>/workload_card.yaml` |
| Server and benchmark logs | `runs/<run id>/logs/` |
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
| 2026-11-03 | us-central1-a | ct6e-standard-1t (auto-mode test) | Pending 2 h 13 min, expired (same) | 0 |
| 2026-11-03 | us-east5-b | ct6e-standard-1t (auto-mode test) | Pending 2 h 15 min, expired (same) | 0 |
| 2026-11-04 to 05 | program zones | flex v6e-1 x3, v5p-4 x2, Spot v6e-1 | Re-queued every 2 h, ~200 Spot stockouts | 0 |
| 2026-11-05 | asia-south1-b | ct6e-standard-1t Spot (smoke) | Ran 8 min; smoke failed on a too-strict check; self-deleted | 0.13 |
| 2026-11-05 | asia-south1-b | ct6e-standard-1t Spot (smoke) | Ran 9 min; smoke passed end to end; self-deleted | 0.15 |
| 2026-11-05 | asia-south1-c | ct6e-standard-8t Spot (full baseline) | Ran 73 min; both layouts measured; disk full in last profile; self-deleted | 9.7 |

Total so far: about 10 v6e chip-hours.
