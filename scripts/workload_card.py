#!/usr/bin/env python3
"""Write a CCL-Bench workload card for one colocated layout's profiled run.

usage: scripts/workload_card.py results/colocated/tp8_ep RUN_ID
Reads versions.txt from that directory and writes workload_card.yaml next to
it. Field layout follows CCL-Bench's workload_card_template.yaml.
"""
import re
import sys
from pathlib import Path

LAYOUTS = {  # layout: (tp, dp_replicate, ep, chips)
    "tp8_ep": (8, 1, 8, 8),
    "dp2_tp4_ep": (4, 2, 8, 8),
    "tp4_ep": (4, 1, 4, 4),
}
PROFILE_CONCURRENCY = 16

CARD = """\
version: 1
description: >
  Colocated serving baseline for attention-FFN disaggregation (AFD) study.
  Qwen3-30B-A3B served by vLLM with the vllm-torchtpu plugin, attention and
  experts on the same chips, layout {layout}. Trace covers one profiled
  `vllm bench serve` run: random dataset, {concurrency} concurrent requests,
  {num_prompts} prompts, 1024 input / 256 output tokens, after the measured
  (unprofiled) sweep. CS 5470 Team 27.
hf_url: https://huggingface.co/Qwen/Qwen3-30B-A3B
trace_url: gs://cs5470-fa26-team27-data/colocated/runs/{run_id}/profile/{layout}/
contributor: Charlie Xue
contact: cx267@cornell.edu

workload:
  model:
    phase: inference
    moe: true
    granularity: model_fwd
    model_family: qwen3-30b-a3b
    precision: bf16
    epochs: 1
    iteration: {num_prompts}  # requests in the profiled window
    model_arch:
      num_params: 30532122624
      num_params_embedding: 311164928  # vocab 151936 * hidden 2048
      num_layers: 48
      num_heads: 32
      head_dim: 128
  data:
    batch_size: {concurrency}  # max concurrent requests
    seq_len: 1280
    input_len: 1024
    output_len: 256
    dataset: random
  hardware:
    network_topo:
      topology: torus_2d  # single v6e host, chips joined by ICI
      bandwidth_gbps:
        - 0    # no scale-out traffic: single host
        - 800  # ICI, 100 GB/s unidirectional per link
    xpu_spec:
      type: TPU
      model: tpu_v6e
      total_count: {chips}
      count_per_node: {chips}
    driver_version: libtpu_{libtpu}

Model-executor:
  framework:
    name: vllm  # vllm {vllm} + vllm-torchtpu {plugin}
    compiler_tool_selection: xla  # torch_tpu, torch.compile(backend="tpu")
  model_plan_parallelization:
    dp_replicate: {dp}
    dp_shard: 1
    tp: {tp}
    pp: 1
    cp: 1
    ep: {ep}
    pp_mb: 1
  communication_library:
    name: xla
    version: libtpu_{libtpu}
    env: {{}}
  protocol_selection:
    - ici

metric_source:
  traces:
    - json   # rank_<N>/*.pt.trace.json.gz (Chrome trace format)
    - xplane # plugins/profile/<timestamp>/rank<N>_*.xplane.pb
  metrics_specific_trace: []
"""


def version(text: str, pkg: str) -> str:
    m = re.search(rf"^{re.escape(pkg)}\s+(\S+)", text, re.MULTILINE | re.IGNORECASE)
    return m.group(1) if m else "unknown"


def main() -> None:
    d = Path(sys.argv[1])
    layout = d.name
    run_id = sys.argv[2]
    tp, dp, ep, chips = LAYOUTS[layout]
    versions = (d / "versions.txt").read_text() if (d / "versions.txt").exists() else ""
    card = CARD.format(
        layout=layout, run_id=run_id, tp=tp, dp=dp, ep=ep, chips=chips,
        concurrency=PROFILE_CONCURRENCY, num_prompts=2 * PROFILE_CONCURRENCY,
        vllm=version(versions, "vllm"),
        # The image installs the plugin as vllm-torchtpu-core 0.0.0, so take
        # the release from the image tag instead.
        plugin=(re.search(r"vllm-torchtpu:(v[\w.]+)", versions) or [None, "unknown"])[1],
        libtpu=version(versions, "libtpu"),
    )
    (d / "workload_card.yaml").write_text(card)
    print(f"wrote {d / 'workload_card.yaml'}")


if __name__ == "__main__":
    main()
