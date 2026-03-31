"""
compare.py
================
6-chart comparison for benchmark.py (full CPU) results:
  1. Batch Vector Search QPS — 100K
  2. Batch Vector Search QPS — 200K
  3. Index Build Time — 100K
  4. Index Build Time — 200K
  5. Concurrent RAG Throughput (req/s)
  6. Concurrent RAG Avg TTFT (ms)

Usage:
  python compare.py 9950x3d2.json 9850x3d.json 9700x.json 285k.json
  python compare.py *.json --output comparison.png
"""

import argparse
import json
import re

import matplotlib.pyplot as plt
import numpy as np

# ── Colors ────────────────────────────────────────────────────────────────
COLORS = ["#c026d3", "#E24B4A", "#D85A30", "#1D9E75", "#378ADD", "#7F77DD", "#888780", "#F0997B"]
BG     = "#FAFAFA"
PANEL  = "#FFFFFF"
TEXT   = "#2C2C2A"
MUTED  = "#73726C"
GRID   = "#E8E8E8"
DIFF_WORSE  = "#E24B4A"
DIFF_BETTER = "#1D9E75"


def load(path):
    with open(path) as f:
        return json.load(f)


def short_cpu(name):
    m = re.search(r'(\d{4}X3D\d*)', name, re.IGNORECASE)
    if m: return m.group(1).upper()
    m = re.search(r'(\d{4}X\b)', name, re.IGNORECASE)
    if m: return m.group(1).upper()
    m = re.search(r'(\d{4}XT\b)', name, re.IGNORECASE)
    if m: return m.group(1).upper()
    m = re.search(r'Ryzen\s+\d+\s+(\d{4})\b', name, re.IGNORECASE)
    if m: return m.group(1)
    m = re.search(r'EPYC\s+(\d{4}\w*)', name, re.IGNORECASE)
    if m: return f"EPYC {m.group(1).upper()}"
    m = re.search(r'Ultra\s+(\d+)\s+(\d{3}\w*\s*\w*)', name, re.IGNORECASE)
    if m: return f"Ultra {m.group(1)} {m.group(2).strip()}"
    m = re.search(r'(i\d-\d{4,5}\w*)', name, re.IGNORECASE)
    if m: return m.group(1)
    parts = name.split()
    return " ".join(parts[2:4]) if len(parts) >= 4 else name[:20]


def get_l3(d):
    l3 = d["meta"].get("l3_cache", "")
    m = re.search(r'(\d+)\s*MiB', l3)
    return f"{m.group(1)}MB" if m else l3


def val_label(ax, x, y, text, color=TEXT, fontsize=9, pt_offset=6):
    from matplotlib.transforms import offset_copy
    trans = offset_copy(ax.transData, fig=ax.figure, x=0, y=pt_offset, units="points")
    ax.text(x, y, text, ha="center", va="bottom", color=color,
            fontsize=fontsize, fontweight="bold", transform=trans)


def diff_label(ax, x, y, val, base, higher_better=True, fontsize=8, pt_offset=18):
    if base == 0:
        return
    pct = (val - base) / base * 100
    sign = "+" if pct > 0 else ""
    if higher_better:
        color = DIFF_BETTER if pct > 0 else DIFF_WORSE
    else:
        color = DIFF_BETTER if pct < 0 else DIFF_WORSE
    from matplotlib.transforms import offset_copy
    trans = offset_copy(ax.transData, fig=ax.figure, x=0, y=pt_offset, units="points")
    ax.text(x, y, f"{sign}{pct:.0f}%", ha="center", va="bottom", color=color,
            fontsize=fontsize, fontweight="bold", transform=trans)


def style_ax(ax, ylabel=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linestyle="-", alpha=0.7)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=10)


def find_by_dbsize(results_list, db_size):
    for r in results_list:
        if r["db_size"] == db_size:
            return r
    return None


# ── Bar chart helper ──────────────────────────────────────────────────────

def plot_bars(ax, datasets, labels, get_val, title, ylabel, higher_better=True,
              fmt_fn=None, unit=""):
    n = len(datasets)
    vals = [get_val(d) for d in datasets]

    # Skip if all None
    if all(v is None for v in vals):
        ax.text(0.5, 0.5, "Data not available", ha="center", va="center",
                color=MUTED, fontsize=11, transform=ax.transAxes)
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold", pad=10)
        return

    vals = [v if v is not None else 0 for v in vals]
    x = np.arange(n)
    colors = [COLORS[i % len(COLORS)] for i in range(n)]

    ax.bar(x, vals, 0.65, color=colors, alpha=0.92, edgecolor="white", linewidth=0.3)

    # Add top margin so value + diff labels don't get clipped
    max_val = max(vals)
    ax.set_ylim(top=max_val * 1.25)

    base = vals[0]
    for i in range(n):
        if fmt_fn:
            txt = fmt_fn(vals[i])
        else:
            txt = f"{vals[i]:,.0f}{unit}"
        val_label(ax, x[i], vals[i], txt, fontsize=8)
        if i > 0 and base > 0:
            diff_label(ax, x[i], vals[i], vals[i], base,
                       higher_better=higher_better, fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TEXT, fontsize=9)
    ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold", pad=10)
    style_ax(ax, ylabel)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full CPU benchmark comparison (2-8 CPUs)")
    parser.add_argument("results", nargs="+", help="Result JSON files")
    parser.add_argument("--output", default="comparison.png")
    args = parser.parse_args()

    if len(args.results) < 2 or len(args.results) > 8:
        print("Error: Need 2-8 result files")
        return

    datasets = [load(p) for p in args.results]

    labels = []
    for d in datasets:
        name = short_cpu(d["meta"]["cpu"])
        l3 = get_l3(d)
        labels.append(f"{name}\n({l3})" if l3 else name)

    fig = plt.figure(figsize=(24, 16))
    fig.patch.set_facecolor(BG)

    fig.suptitle(
        "x3d-rag-benchmark  —  Full CPU Benchmark for RAG AI Pipeline\n"
        "Personal PC & small-team on-premises RAG (100K–200K vectors, single-node)",
        color=TEXT, fontsize=14, fontweight="bold", y=0.99
    )

    gs = fig.add_gridspec(3, 2, left=0.06, right=0.97, top=0.86, bottom=0.06,
                          wspace=0.22, hspace=0.40)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    # 1. QPS 100K
    plot_bars(ax1, datasets, labels,
              lambda d: (find_by_dbsize(d["batch_vector_search"], 100000) or {}).get("qps"),
              "Batch vector search QPS — 100K (higher = better)",
              "QPS", higher_better=True)

    # 2. QPS 200K
    plot_bars(ax2, datasets, labels,
              lambda d: (find_by_dbsize(d["batch_vector_search"], 200000) or {}).get("qps"),
              "Batch vector search QPS — 200K (higher = better)",
              "QPS", higher_better=True)

    # 3. Index Build 100K
    plot_bars(ax3, datasets, labels,
              lambda d: (find_by_dbsize(d["index_build"], 100000) or {}).get("build_time_s"),
              "Index build time — 100K (lower = better)",
              "Seconds", higher_better=False,
              fmt_fn=lambda v: f"{v:.2f}s")

    # 4. Index Build 200K
    plot_bars(ax4, datasets, labels,
              lambda d: (find_by_dbsize(d["index_build"], 200000) or {}).get("build_time_s"),
              "Index build time — 200K (lower = better)",
              "Seconds", higher_better=False,
              fmt_fn=lambda v: f"{v:.2f}s")

    # 5. Concurrent RAG throughput
    plot_bars(ax5, datasets, labels,
              lambda d: d["concurrent_rag"]["throughput_qps"] if d.get("concurrent_rag") else None,
              "Concurrent RAG throughput — 8 workers (higher = better)",
              "req/s", higher_better=True,
              fmt_fn=lambda v: f"{v:.1f}")

    # 6. Concurrent RAG TTFT
    plot_bars(ax6, datasets, labels,
              lambda d: d["concurrent_rag"]["avg_ttft_ms"] if d.get("concurrent_rag") else None,
              "Concurrent RAG avg TTFT — 8 workers (lower = better)",
              "ms", higher_better=False,
              fmt_fn=lambda v: f"{v:.1f}ms")

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=COLORS[i % len(COLORS)], label=labels[i].replace("\n", " "))
               for i in range(len(datasets))]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(datasets), 7),
               bbox_to_anchor=(0.5, 0.92), facecolor=PANEL, labelcolor=TEXT,
               fontsize=9, framealpha=0.95, edgecolor=GRID)

    # Footer
    parts = []
    for d in datasets:
        cpu = d["meta"]["cpu"]
        l3 = d["meta"].get("l3_cache", "")
        parts.append(f"{cpu} ({l3})")
    fig.text(0.5, 0.005,
             "  |  ".join(parts) + "  |  github.com/sorrymannn/x3d-rag-benchmark",
             ha="center", color="#999999", fontsize=7)

    plt.savefig(args.output, dpi=150, facecolor=BG)
    print(f"Saved: {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
