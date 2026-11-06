#!/usr/bin/env python3
"""GSM8K accuracy against a running vLLM server, with lm-evaluation-harness.

The standard vLLM accuracy check: the OpenAI completions endpoint, GSM8K
test split (1,319 problems), 5-shot, greedy. Two scores come out:
strict-match (answer in the "#### N" format) and flexible-extract (last
number in the reply). A serving change that computes the model wrongly drops
these sharply; a correct one matches the reference layout within noise.

  pip install "lm-eval[api]==0.4.13"
  python3 gsm8k_eval.py --port 8000 --model Qwen/Qwen3-30B-A3B --out gsm8k.json \
      [--header X-data-parallel-rank=0] [--limit 100]
"""
import argparse
import json

import lm_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--header", default="", help="KEY=VALUE sent with every request")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="first N problems only")
    args = ap.parse_args()

    header = {"Authorization": "Bearer EMPTY"}
    if args.header:
        key, value = args.header.split("=", 1)
        header[key] = value

    results = lm_eval.simple_evaluate(
        model="local-completions",
        model_args={
            "model": args.model,
            "base_url": f"http://localhost:{args.port}/v1/completions",
            "num_concurrent": args.concurrency,
            "max_retries": 3,
            "tokenized_requests": False,
            "tokenizer_backend": None,
            "max_gen_toks": 256,
            "header": header,
        },
        tasks=["gsm8k"],
        num_fewshot=5,
        limit=args.limit,
        log_samples=False,
    )
    scores = results["results"]["gsm8k"]
    summary = {
        "task": "gsm8k", "num_fewshot": 5, "limit": args.limit,
        "header": args.header or None, "lm_eval_version": lm_eval.__version__,
        "scores": {k: v for k, v in scores.items() if isinstance(v, (int, float))},
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print("GSM8K", json.dumps(summary["scores"]), flush=True)


if __name__ == "__main__":
    main()
