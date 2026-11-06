# Attention-FFN disaggregation (AFD) on TPU v6e

AFD runs correctly in vLLM on 8 TPU v6e chips: 4 chips run attention for
every request, 4 chips run all 128 experts. This first version (v1) is
19 to 44% slower than colocated serving; the reasons and the plan to close the
gap are below. Colocated baseline: [BASELINE.md](BASELINE.md).

## Run it

```bash
scripts/launch_spot.sh 8 3 afd_dp2        # finds Spot capacity, serves, benchmarks, deletes itself
scripts/fetch_results.sh <run id>         # results/afd/afd_dp2/ and results/summary.csv
```

`afd_dp2` uses the same server flags as the colocated `dp2_tp4_ep` layout plus
the plugin in `afd/plugin/` (enabled with `AFD_ROLE_SPLIT=1`), and every
request carries the header `X-data-parallel-rank: 0`.

## Design

```
attention replica (DP rank 0, chips 0-3, TP=4)   all requests, all KV cache
        | tokens: all-gather across replicas            ^ results: reduce-scatter,
        v                                               | then all-reduce over 4 chips
expert replica   (DP rank 1, chips 4-7)           chip j holds experts 32j..32j+31
```

The plugin (`afd/plugin/afd_tpu/__init__.py`, a vLLM general plugin, no fork
of vLLM or vllm-torchtpu) changes three things:

1. **Expert placement.** It patches `FusedMoEParallelConfig.make` so the
   expert-parallel size is the TP size (4) and the rank is the TP rank: chip j
   of each replica loads experts 32j to 32j+31.
2. **Role as runtime data.** On the attention replica it sets the layer's
   `_experts_start` buffer to 128. vllm-torchtpu's fused MoE kernel then finds
   no local expert for any token (weight 0, id -1), so attention chips do no
   expert work. The buffer is runtime data, so every chip still runs the same
   compiled program, which torch_tpu requires (see Findings).
3. **Request pinning.** vLLM routes a request with header
   `X-data-parallel-rank: 0` to replica 0. Replica 1 gets no requests; vLLM
   keeps it in step with padding ("dummy") batches.

Dispatch and combine are vLLM's existing data-parallel MoE collectives
(all-gather before the experts, reduce-scatter after, then the TP all-reduce),
the same ones the colocated `dp2_tp4_ep` layout runs. After the reduce-scatter,
attention chip j holds the result of experts 32j to 32j+31; the TP all-reduce
over the 4 attention chips adds all 128.

## Findings: how AFD can be built on torch_tpu

Measured with `afd/transfer_test.py` and `afd/jax_role_test.py` on 8 v6e chips
(4 attention + 4 expert, and 2 + 6), each checked against a float32 reference.
Results: `results/afd_xfer*/`, `results/afd_jax/`.

| Approach | Result |
|---|---|
| Separate compiled program per role (PyTorch) | Hangs. With exchange order forced by data dependencies it no longer hangs but returns scrambled data. |
| One program with `torch.cond` | Not supported: the compiler tries to read the condition on the host. |
| Uneven all-to-all (send only real data) | Fails to compile: "asymmetric input buffer sizes across ranks". |
| Eager `isend`/`irecv` outside the compiled step | Correct, but 1.2 to 1.4 ms per layer (about 90 times the in-program exchange). Each transfer needs its own tag, or back-to-back transfers hang. |
| **One shared program, role as data** | **Correct; 15 to 90 us per layer for dispatch + combine.** |
| JAX `shard_map` with `lax.cond` on the device index | Correct, and each chip skips the other role's work (stand-alone JAX test, one process driving all 8 chips; needs `--ulimit nofile` raised). |

vllm-torchtpu hit the same constraint: a comment in its MoE routing code
records that compiling rank-specific values into each rank's program
"desync[ed] the in-graph EP all-to-all/gather collectives and halt[ed] the TPU
core", and its fix was to pass them as runtime data, which this design reuses.

Exchange cost (single shared program, median of 3 runs, 4:4 split): 15 us at
1 token per attention chip, 19 us at 8, 62 us at 32, 87 us at 128. That is
about 13 to 23% of a whole colocated TP=8 layer at the same batch (TPOT / 48).
Large-batch timings varied between runs (128 tokens: 83 to 328 us).

## Correctness

- All four known-answer prompts pass (`results/afd/afd_dp2/sanity.json`).
- Greedy continuations (`greedy.json`) match the colocated reference until a
  near-tie. At each divergence the colocated token is AFD's runner-up with
  nearly equal probability (0.239 vs 0.239, 0.442 vs 0.442, 0.233 vs 0.206,
  0.525 vs 0.463). Greedy output also varies within AFD itself when other
  requests share the batch, so exact token matching is not a usable test in
  bf16. A missing or double-counted expert would push the reference token far
  down AFD's ranking.
- Not yet run: GSM8K for AFD and colocated (`afd/gsm8k_eval.py`, wired into
  the benchmark script, verified only against the mock server).

## Results (v1)

Same protocol as the baseline: random tokens, 1024 in / 256 out, 200 prompts
per point. Run `20261106-170404-charlie-spot-asia-south1-c` (one sweep);
colocated DP=2xTP=4 is the mean of three runs.

| Concurrency | AFD output tok/s | per-user tok/s | TTFT | Colocated DP=2xTP=4 output tok/s | per-user tok/s | TTFT | AFD vs colocated |
|---|---|---|---|---|---|---|---|
| 1 | 116 | 118 | 60 ms | 142 | 146 | 52 ms | -19% |
| 4 | 413 | 111 | 183 ms | 558 | 148 | 118 ms | -26% |
| 8 | 728 | 102 | 305 ms | 1,010 | 139 | 196 ms | -28% |
| 16 | 1,254 | 92 | 406 ms | 1,729 | 127 | 276 ms | -28% |
| 32 | 1,802 | 68 | 525 ms | 2,832 | 112 | 373 ms | -36% |
| 64 | 2,429 | 48 | 809 ms | 3,650 | 72 | 533 ms | -33% |
| 128 | 3,016 | 34 | 1,969 ms | 5,425 | 61 | 1,021 ms | -44% |

Why v1 is slower:

1. **Each chip group idles half of every layer.** Attention must finish before
   the experts start and the experts before the next layer's attention, so the
   4 attention chips wait while the 4 expert chips work and vice versa.
   Colocated serving keeps all 8 busy. AFD systems (MegaScale-Infer) split
   batches into micro-batches and alternate them between the chip groups.
2. **Half the attention capacity.** Every request goes through 4 chips instead
   of two replicas of 4, which shows most in time to first token.
3. **Wasted work.** The expert replica runs attention on padding batches and
   computes experts for its own padding tokens as well as the attention
   replica's tokens; the attention chips load 32 experts each that they never
   use.

## Next

1. Remove the wasted work in 3 (mask the expert replica's own tokens; avoid
   loading experts on attention chips).
2. Micro-batch overlap so both chip groups work at once.
3. Other attention:expert splits (2:6 needs a different attention layout).
4. GSM8K for AFD and colocated; the fuller benchmark protocol (`WORKLOADS`,
   `LENGTHS=range`, `NUM_PROMPTS=auto`, `SHAREGPT=1` in `scripts/vm/vm_baseline.sh`).
5. Per-layer time breakdown from the profile traces
   (`gs://.../runs/20261106-170404-charlie-spot-asia-south1-c/profile/`) with CCL-Bench.
