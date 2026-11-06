#!/usr/bin/env python3
"""Figures for the midpoint poster, from measured data only.

usage: python3 poster/make_figures.py      (writes poster/fig_*.svg)
Reads results/colocated/<layout>/colocated_c<N>_r<K>.json.
"""
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "poster"
RES = ROOT / "results" / "colocated"
CHIPS = 8
LAYOUTS = {
    "tp8_ep": ("TP=8 + EP=8 (one replica)", "#1f5aa6", "o"),
    "dp2_tp4_ep": ("DP=2 x TP=4 + EP=8 (two replicas)", "#d9822b", "s"),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "axes.titlesize": 17,
    "axes.labelsize": 16, "legend.fontsize": 13, "axes.spines.top": False,
    "axes.spines.right": False, "svg.fonttype": "path",
})


def load():
    data = defaultdict(lambda: defaultdict(list))  # layout -> c -> [result]
    for f in RES.glob("*/colocated_c*.json"):
        m = re.match(r"colocated_c(\d+)(?:_r\d+)?\.json$", f.name)
        if m and f.parent.name in LAYOUTS:
            data[f.parent.name][int(m.group(1))].append(json.loads(f.read_text()))
    return data


def mean(results, key):
    return statistics.mean(r[key] for r in results)


def pareto(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    frontier = []
    for layout, (label, color, marker) in LAYOUTS.items():
        cs = sorted(data[layout])
        x = [1000 / mean(data[layout][c], "mean_tpot_ms") for c in cs]
        y = [mean(data[layout][c], "output_throughput") / CHIPS for c in cs]
        ax.plot(x, y, marker=marker, color=color, lw=2.5, ms=10, label=label)
        for c, xi, yi in zip(cs, x, y):
            dx, dy = (-38, 6) if layout == "dp2_tp4_ep" and c >= 64 else (8, -14 if layout == "dp2_tp4_ep" else 6)
            ax.annotate(f"c={c}", (xi, yi), textcoords="offset points", xytext=(dx, dy),
                        fontsize=12, color=color)
            frontier.append((xi, yi))
        # Spread across repeats (only visible where runs disagree).
        for c, xi in zip(cs, x):
            outs = [r["output_throughput"] / CHIPS for r in data[layout][c]]
            if max(outs) - min(outs) > 0.1 * statistics.mean(outs):
                ax.vlines(xi, min(outs), max(outs), color=color, lw=6, alpha=0.3)
    # Pareto frontier: best throughput at each per-user speed.
    pts = sorted(frontier, key=lambda p: -p[0])
    best, front = -1, []
    for p in pts:
        if p[1] > best:
            front.append(p)
            best = p[1]
    ax.plot(*zip(*front), ls="--", color="#333", lw=1.5, label="Colocated frontier")
    ax.set_xlim(50, 200)
    ax.text(80.5, 225, "c=32: 3 runs\nranged 224-362", fontsize=11, color=LAYOUTS["tp8_ep"][1])
    ax.set_xlabel("Per-user decode speed (tokens/s per request)")
    ax.set_ylabel("Output throughput per chip (tokens/s)")
    ax.set_title("Throughput vs. per-user speed (8 x v6e)")
    ax.legend(loc="upper right", frameon=False, fontsize=12)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pareto.svg")
    fig.savefig(OUT / "preview" / "fig_pareto.png", dpi=70)


def latency(data):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    panels = [("mean_ttft_ms", "Mean time to first token (ms)"),
              ("mean_tpot_ms", "Mean time per token (ms)"),
              ("p99_itl_ms", "p99 inter-token gap (ms)")]
    for ax, (key, title) in zip(axes, panels):
        for layout, (label, color, marker) in LAYOUTS.items():
            cs = sorted(data[layout])
            ax.plot(cs, [mean(data[layout][c], key) for c in cs], marker=marker,
                    color=color, lw=2.5, ms=9, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 8, 16, 32, 64, 128], ["1", "4", "8", "16", "32", "64", "128"])
        ax.set_xlabel("Concurrent requests")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, loc="upper left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_latency.svg")
    fig.savefig(OUT / "preview" / "fig_latency.png", dpi=70)


def memory():
    # Parameter counts from the Qwen3-30B-A3B config (48 layers, hidden 2048,
    # 32 q / 4 kv heads x 128, 128 experts x 768, vocab 151936, untied).
    gb = 2 / 1e9  # bf16 bytes per parameter, in GB
    experts = 48 * 128 * 3 * 2048 * 768 * gb
    attention = 48 * (2048 * 4096 + 2 * 2048 * 512 + 4096 * 2048) * gb
    embed = 2 * 151936 * 2048 * gb
    kv_per_token = 48 * 4 * 128 * 2 * 2  # bytes, bf16 K and V
    kv_c128 = 128 * 1280 * kv_per_token / 1e9
    fig, ax = plt.subplots(figsize=(8, 4.8))
    items = [("Expert FFN weights\n(128 experts x 48 layers)", experts, "#d9822b"),
             ("Attention weights", attention, "#1f5aa6"),
             ("Embeddings + LM head", embed, "#7a9cc6"),
             ("KV cache, 128 requests x\n1280 tokens (bf16)", kv_c128, "#5aa66b")]
    for i, (name, val, color) in enumerate(items):
        ax.barh(i, val, color=color)
        ax.text(val + 0.8, i, f"{val:.1f} GB", va="center", fontsize=14)
    for k in (1, 2):
        ax.axvline(32 * k, color="#999", ls=":", lw=1.5)
        ax.text(32 * k + 0.5, -0.55, f"{k} chip{'s' if k > 1 else ''} ({32 * k} GB)",
                ha="left", va="bottom", fontsize=12, color="#555")
    ax.set_yticks(range(len(items)), [i[0] for i in items])
    ax.set_xlabel("Memory (GB)")
    ax.set_xlim(0, 75)
    ax.set_ylim(len(items) - 0.5, -0.9)
    ax.set_title("Qwen3-30B-A3B memory, bf16")
    fig.tight_layout()
    fig.savefig(OUT / "fig_memory.svg")
    fig.savefig(OUT / "preview" / "fig_memory.png", dpi=70)


def capacity():
    # Counted from our request logs, 2026-11-02 to 11-05 (Spot attempts plus
    # flex-start requests, each flex request waiting up to 2 h).
    zones = ["us-east5-a", "us-east5-b", "us-central1-a",
             "12 other zones\n(US, EU, Asia)", "asia-south1-b/c"]
    attempts = [116, 106, 113, 14, 5]
    started = [0, 0, 0, 1, 4]  # the other-zone VM was reclaimed within seconds
    fig, ax = plt.subplots(figsize=(8, 4.2))
    zones = [z.replace("\n", " ") for z in zones]
    ax.barh(zones, attempts, color="#c9c9c9", label="Requests (Spot + flex-start)")
    ax.barh(zones, started, color="#2e8b57", label="VM actually started")
    for i, (a, st) in enumerate(zip(attempts, started)):
        ax.text(a + 2, i, f"{st} of {a} started", va="center", fontsize=13)
    ax.invert_yaxis()
    ax.set_xlim(0, 160)
    ax.set_xlabel("Requests, Nov 2-5")
    ax.set_title("TPU requests vs. VMs that started")
    ax.legend(frameon=False, loc="lower right", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_capacity.svg")
    fig.savefig(OUT / "preview" / "fig_capacity.png", dpi=70)


if __name__ == "__main__":
    (OUT / "preview").mkdir(exist_ok=True)
    d = load()
    pareto(d)
    latency(d)
    memory()
    capacity()
    print("wrote", sorted(p.name for p in OUT.glob("fig_*.svg")))
