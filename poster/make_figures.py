#!/usr/bin/env python3
"""Figures for the midpoint poster, from measured data in results/.

usage: python3 poster/make_figures.py      (writes poster/fig_*.svg)

Each figure is drawn at its printed size in inches, so font sizes below are
real point sizes on the 48 x 36 in poster.
"""
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "poster"
RES = ROOT / "results"
CHIPS = 8
BLUE, ORANGE, GREEN, GREY, INK = "#2b5c9e", "#d98a2b", "#2e8b57", "#9aa0a6", "#1d1d1f"
LAYOUTS = {
    "tp8_ep": ("Colocated, TP=8", BLUE, "o"),
    "dp2_tp4_ep": ("Colocated, 2 replicas of TP=4", ORANGE, "s"),
    "afd_dp2": ("AFD v1, 4 attention + 4 expert chips", GREEN, "^"),
}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 30, "axes.labelsize": 32,
    "xtick.labelsize": 28, "ytick.labelsize": 28, "legend.fontsize": 28,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 2,
    "xtick.major.width": 2, "ytick.major.width": 2, "xtick.major.size": 8,
    "ytick.major.size": 8, "svg.fonttype": "path", "svg.hashsalt": "afd-poster", "axes.edgecolor": INK,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
})


def load():
    data = defaultdict(lambda: defaultdict(list))  # layout -> c -> [result]
    for f in RES.glob("*/*/colocated_c*.json"):
        m = re.match(r"colocated_c(\d+)(?:_r\d+)?\.json$", f.name)
        if m and f.parent.name in LAYOUTS:
            data[f.parent.name][int(m.group(1))].append(json.loads(f.read_text()))
    return data


def mean(results, key):
    return statistics.mean(r[key] for r in results)


def save(fig, name):
    # Fixed salt and no date, so unchanged figures rebuild byte-for-byte.
    fig.savefig(OUT / f"{name}.svg", metadata={"Date": None})
    (OUT / "preview").mkdir(exist_ok=True)
    fig.savefig(OUT / "preview" / f"{name}.png", dpi=40)
    plt.close(fig)


RED, DARK, LIGHT = "#b31b1b", "#3c4043", "#b0b4b9"
LIGHT_FILL = {BLUE: "#dbe6f4", GREEN: "#dcefe3", RED: "#f6dada"}


def ylabel_top(ax, text):
    """Horizontal y-axis label above the axis (easier to read than rotated)."""
    ax.text(0, 1.03, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=30)


def throughput(data, size):
    """Total output tokens/s vs concurrent requests, AFD vs both colocated layouts."""
    fig, ax = plt.subplots(figsize=size)
    series = [("tp8_ep", "Colocated, TP=8", LIGHT, "--"),
              ("dp2_tp4_ep", "Colocated, DP=2 × TP=4", DARK, "-"),
              ("afd_dp2", "AFD v1", RED, "-")]
    ys = {}
    for layout, label, color, ls in series:
        cs = sorted(data[layout])
        ys[layout] = {c: mean(data[layout][c], "output_throughput") for c in cs}
        ax.plot(cs, [ys[layout][c] for c in cs], ls=ls, color=color, lw=6, marker="o",
                ms=16, zorder=3 if layout == "afd_dp2" else 2, label=label)
    base, afd = ys["dp2_tp4_ep"], ys["afd_dp2"]
    cs = sorted(afd)
    ax.fill_between(cs, [afd[c] for c in cs], [base[c] for c in cs], color=RED, alpha=0.08,
                    lw=0, zorder=1)
    gaps = {c: afd[c] / base[c] - 1 for c in afd}
    for c, g in gaps.items():
        ax.text(c * 1.15, afd[c] - 120, f"{g:.0%}".replace("-", "−"),
                ha="left", va="top",
                fontsize=28, color=RED)
    print("throughput gap vs 2xTP4:", {c: f"{g:.0%}" for c, g in gaps.items()})
    ax.set_xscale("log", base=2)
    cs = sorted(afd)
    ax.set_xticks(cs, [str(c) for c in cs])
    ax.minorticks_off()
    ax.set_xlim(0.8, 160)
    ax.set_ylim(-300, 5800)
    ax.set_yticks(range(0, 5001, 1000), [f"{v:,}" for v in range(0, 5001, 1000)])
    handles, labels = ax.get_legend_handles_labels()
    order = [1, 0, 2]  # DP=2 x TP=4, TP=8, AFD
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left",
              frameon=False, fontsize=30, handlelength=2.6)
    ax.set_xlabel("Concurrent requests")
    ylabel_top(ax, "Output tokens/s, 8 chips")
    fig.tight_layout()
    save(fig, "fig_throughput")


def exchange(data, size):
    """Per-layer dispatch+combine cost between chip groups (afd/transfer_test.py,
    single shared program), median over 3 runs, vs a whole colocated layer."""
    pts = defaultdict(list)
    for r in ("afd_xfer", "afd_xfer_run4", "afd_xfer_run6"):
        for row in json.loads((RES / r / "summary.json").read_text()):
            if row["design"] == "homo" and row["status"] == "ok":
                for b, p in row["points"].items():
                    pts[(row["split"], int(b))].append(p["per_layer_us_attn"])
    fig, ax = plt.subplots(figsize=size)
    bs = [1, 8, 32, 128]
    layer = [mean(data["tp8_ep"][c], "mean_tpot_ms") * 1000 / 48 for c in bs]
    xfer = [statistics.median(pts[("4:4", b)]) for b in bs]
    ys = list(range(len(bs)))[::-1]
    h = 0.62
    ax.barh(ys, layer, height=h, color="#d5d8dc")
    ax.barh(ys, xfer, height=h, color=RED)
    for i, (y, lv, xv) in enumerate(zip(ys, layer, xfer)):
        first = i == len(bs) - 1
        ax.text(xv + 5, y, f"{xv:.0f} µs{' exchange' if first else ''} ({xv / lv:.0%})",
                va="center", fontsize=27, color=RED, fontweight="bold")
        ax.text(lv + 6, y, f"{lv:.0f} µs{' whole layer' if first else ''}", va="center",
                fontsize=27, color=DARK)
    print("exchange share of layer:", [f"{x / l:.0%}" for x, l in zip(xfer, layer)])
    ax.set_yticks(ys, [f"{b} token" + ("s" if b > 1 else "") for b in bs])
    ax.tick_params(axis="y", length=0, pad=14)
    ax.set_xlim(0, 480)
    ax.set_ylim(-0.5, ys[0] + 0.5)
    ax.set_xticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_exchange")


def roles(size):
    """One MoE layer of the shared program (afd/plugin): what each chip does.
    From the plugin docstring: tokens are all-gathered across the two replicas,
    experts run, and a reduce-scatter returns each token's partial sums."""
    fig, ax = plt.subplots(figsize=size)
    idle = "#e8eaed"
    rows = [  # label, cells (color, text) per chip or one spanning cell
        ("Attention", [(BLUE, "attn")] * 4 + [(idle, "pad")] * 4),
        ("Send tokens", [(RED, "each token is copied to all 4 expert chips|all-gather")]),
        ("Experts", [(idle, "none")] * 4 + [(GREEN, f"{32 * j}–{32 * j + 31}") for j in range(4)]),
        ("Return results", [(RED, "each token’s expert outputs are added up on the attention chips|reduce-scatter")]),
    ]
    w, gap = 1.0, 0.12
    xs = [i * (w + gap) + (0.35 if i >= 4 else 0) for i in range(8)]
    right = xs[-1] + w
    for r, (label, cells) in enumerate(rows):
        y = -r * 1.25
        ax.text(-0.2, y, label, ha="right", va="center", fontsize=32, fontweight="bold")
        if len(cells) == 1:
            color, text = cells[0]
            ax.add_patch(FancyBboxPatch((0, y - 0.42), right, 0.84,
                         boxstyle="round,pad=0,rounding_size=0.03", fc=LIGHT_FILL[color],
                         ec=color, lw=2.5))
            main, op = text.split("|")
            ax.text(right / 2, y + 0.13, main, ha="center", va="center", color=INK, fontsize=22)
            ax.text(right / 2, y - 0.22, op, ha="center", va="center", color=RED, fontsize=20)
            continue
        for x, (color, text) in zip(xs, cells):
            ax.add_patch(FancyBboxPatch((x, y - 0.45), w, 0.9,
                         boxstyle="round,pad=0,rounding_size=0.03", fc=LIGHT_FILL.get(color, color),
                         ec="#bdc1c6" if color == idle else color, lw=2.5,
                         ls="--" if color == idle else "-"))
            ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=24 if "–" in text else 28,
                    color="#80868b" if color == idle else INK)
    ax.text((xs[0] + xs[3] + w) / 2, 0.85, "4 attention chips", ha="center", fontsize=32, color=BLUE,
            fontweight="bold")
    ax.text((xs[4] + xs[7] + w) / 2, 0.85, "4 expert chips", ha="center", fontsize=32, color=GREEN,
            fontweight="bold")
    ax.set_xlim(-3.2, right + 0.05)
    ax.set_ylim(-4.35, 1.35)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    save(fig, "fig_roles")


if __name__ == "__main__":
    # Slot sizes measured from the laid-out poster by build.py.
    sizes_file = OUT / "fig_sizes.json"
    sizes = json.loads(sizes_file.read_text()) if sizes_file.exists() else {}
    size = lambda name, default: tuple(sizes.get(name, default))  # noqa: E731
    d = load()
    throughput(d, size("fig_throughput", (19.0, 16.0)))
    exchange(d, size("fig_exchange", (12.5, 8.6)))
    roles(size("fig_roles", (14.0, 7.0)))
    print("wrote", sorted(p.name for p in OUT.glob("fig_*.svg")))
