# AFD for MoE serving on Cloud TPU v6e

CS 5470 Team 27 (Charlie Xue, Akhil Chilaka, Rohan Anne, Jerry Ji). We compare
colocated serving of Qwen3-30B-A3B with attention-FFN disaggregation (AFD), where
attention and experts run on separate TPU chips, using vLLM with the
vllm-torchtpu backend on 8 TPU v6e chips.

| Document | Contents |
|---|---|
| [BASELINE.md](BASELINE.md) | Colocated baseline: how to run it, settings, results, problems hit, VM time |
| [AFD.md](AFD.md) | AFD design, how to run it, what works on TPU and what does not, first results |
| [results/README.md](results/README.md) | Which run each result file came from |
| `poster/` | Midpoint poster (`poster.pdf`), its figures and the scripts that make them |

Quick start (needs `gcloud` access to project `cs5470-fa26-team27`):

```bash
scripts/launch_spot.sh 1 1 smoke      # cheap end-to-end check (Qwen3-0.6B, 1 chip)
scripts/launch_spot.sh 8 4            # colocated baseline: tp8_ep and dp2_tp4_ep
scripts/launch_spot.sh 8 3 afd_dp2    # AFD
scripts/fetch_results.sh              # results/ and results/summary.csv
gcloud compute instances list         # confirm nothing is left running
```

Every VM serves, benchmarks, uploads to `gs://cs5470-fa26-team27-data/colocated/runs/<run id>/`
and deletes itself. The project allows at most 8 v6e and 8 v5p chips at once.
