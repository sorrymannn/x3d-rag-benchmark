"""
compare_multi.py
================
Compare multiple (2-6) benchmark results in a single chart.

Usage:
  python compare_multi.py result1.json result2.json result3.json result4.json
  python compare_multi.py *.json --output multi_comparison.png
"""

import argparse
import json
import re

import matplotlib.pyplot as plt
import numpy as np

# ── Colors ────────────────────────────────────────────────────────────────────
COLORS = ["#E84B4B", "#4B9BE8", "#4BE88A", "#E8B84B", "#B84BE8", "#E84B9B"]
BG     = "#1A1A2E"
PANEL  = "#16213E"
TEXT   = "#E0E0E0"
GRID   = "#2A2A4A"
ERR_KW = None  # no error bars in multi-comparison (cleaner visuals)


def load(path):
    with open(path) as f:
        return json.load(f)


def short_cpu(name):
    m = re.search(r'(\d{4}X3D)', name, re.IGNORECASE)
    if m: return m.group(1).upper()
    m = re.search(r'(\d{4}X\b)', name, re.IGNORECASE)
    if m: return m.group(1).upper()
    m = re.search(r'(\d{4}XT\b)', name, re.IGNORECASE)
    if m: return m.group(1).upper()
    m = re.search(r'Ryzen\s+\d+\s+(\d{4})\b', name, re.IGNORECASE)
    if m: return m.group(1)
    m = re.search(r'EPYC\s+(\d{4}\w*)', name, re.IGNORECASE)
    if m: return f"EPYC {m.group(1).upper()}"
    m = re.search(r'Threadripper\s+(?:PRO\s+)?(\d{4}\w*)', name, re.IGNORECASE)
    if m: return f"TR {m.group(1).upper()}"
    m = re.search(r'Ultra\s+(\d+)\s+(\d{3}\w*)', name, re.IGNORECASE)
    if m: return f"Ultra {m.group(1)} {m.group(2).upper()}"
    m = re.search(r'(i\d-\d{4,5}\w*)', name, re.IGNORECASE)
    if m: return m.group(1)
    m = re.search(r'Xeon\s+(?:W-|Gold\s+|Platinum\s+|Silver\s+)?(\w+)', name, re.IGNORECASE)
    if m: return f"Xeon {m.group(1)}"
    m = re.search(r'\b(\d{3,5}[A-Z0-9]*)\b', name, re.IGNORECASE)
    if m: return m.group(1).upper()
    parts = name.split()
    return " ".join(parts[2:4]) if len(parts) >= 4 else name[:20]


def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT)
    ax.spines[:].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linestyle="--", alpha=0.5)


# ── Plot 1: Vector Search QPS ─────────────────────────────────────────────────

def plot_qps(ax, datasets, labels):
    n = len(datasets)
    sizes = [r["db_size"] for r in datasets[0]["vector_search"]]
    n_sizes = len(sizes)
    total_w = 0.75
    w = total_w / n

    for j, (d, label) in enumerate(zip(datasets, labels)):
        qps = [r["qps"] for r in d["vector_search"]]
        err = [r.get("qps_stddev", 0) for r in d["vector_search"]]
        x = np.arange(n_sizes) - total_w/2 + w*j + w/2
        ax.bar(x, qps, w * 0.9, label=label, color=COLORS[j], alpha=0.85,
               )

    ax.set_xticks(np.arange(n_sizes))
    ax.set_xticklabels([f"{s//1000}K\nvectors" for s in sizes], color=TEXT)
    ax.set_ylabel("QPS (higher is better)", color=TEXT)
    ax.set_title("Vector Search QPS (FAISS HNSW)", color=TEXT, fontsize=12, pad=10)
    style_ax(ax)


# ── Plot 2: P99 Latency line ──────────────────────────────────────────────────

def plot_p99(ax, datasets, labels):
    sizes = [r["db_size"] for r in datasets[0]["vector_search"]]
    xs = range(len(sizes))

    for j, (d, label) in enumerate(zip(datasets, labels)):
        p99 = [r["latency_p99_ms"] for r in d["vector_search"]]
        ax.plot(xs, p99, "o-", color=COLORS[j],
                label=label, linewidth=2, markersize=5, alpha=0.85)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s//1000}K" for s in sizes], color=TEXT)
    ax.set_ylabel("P99 Latency (ms, lower is better)", color=TEXT)
    ax.set_title("Vector Search P99 Latency", color=TEXT, fontsize=12, pad=10)
    style_ax(ax)


# ── Plot 3: P50/P95/P99 bar (largest DB) ─────────────────────────────────────

def plot_latency_bars(ax, datasets, labels):
    n = len(datasets)
    metrics = ["P50", "P95", "P99"]
    n_metrics = len(metrics)
    total_w = 0.75
    w = total_w / n

    for j, (d, label) in enumerate(zip(datasets, labels)):
        r = d["vector_search"][-1]
        vals = [r["latency_p50_ms"], r["latency_p95_ms"], r["latency_p99_ms"]]
        errs = [r.get("latency_p50_stddev", 0), 0, r.get("latency_p99_stddev", 0)]
        x = np.arange(n_metrics) - total_w/2 + w*j + w/2
        ax.bar(x, vals, w * 0.9, label=label, color=COLORS[j], alpha=0.85,
               )

    db_size = datasets[0]["vector_search"][-1]["db_size"]
    ax.set_xticks(np.arange(n_metrics))
    ax.set_xticklabels(metrics, color=TEXT)
    ax.set_ylabel("Latency (ms, lower is better)", color=TEXT)
    ax.set_title(f"Search Latency Distribution\n({db_size//1000}K vectors, largest DB)",
                 color=TEXT, fontsize=12, pad=10)
    style_ax(ax)


# ── Plot 4: RAG Pipeline — Vector Search Latency ─────────────────────────────

def plot_rag(ax, datasets, labels):
    has_rag = all(d.get("rag_ttft") for d in datasets)
    if not has_rag:
        ax.text(0.5, 0.5,
                "RAG data not available\n(run without --skip-rag)",
                ha="center", va="center", color=TEXT, fontsize=11,
                transform=ax.transAxes)
        ax.set_facecolor(PANEL)
        ax.set_title("RAG Pipeline — Vector Search Latency", color=TEXT, fontsize=12, pad=10)
        return

    n = len(datasets)
    metrics = ["P50", "P95", "P99"]
    n_metrics = len(metrics)
    total_w = 0.75
    w = total_w / n

    for j, (d, label) in enumerate(zip(datasets, labels)):
        rag = d["rag_ttft"]
        vals = [rag["vector_search"]["p50_ms"],
                rag["vector_search"]["p95_ms"],
                rag["vector_search"]["p99_ms"]]
        err_val = rag["vector_search"].get("stddev_ms", 0)
        errs = [err_val] * 3
        x = np.arange(n_metrics) - total_w/2 + w*j + w/2
        ax.bar(x, vals, w * 0.9, label=label, color=COLORS[j], alpha=0.85,
               )

    ax.set_xticks(np.arange(n_metrics))
    ax.set_xticklabels(metrics, color=TEXT)
    ax.set_ylabel("Vector Search Latency in RAG (ms)", color=TEXT)
    ax.set_title("RAG Pipeline — Vector Search Latency\n(lower is better)",
                 color=TEXT, fontsize=12, pad=10)
    style_ax(ax)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare 2-6 benchmark results in a single chart")
    parser.add_argument("results", nargs="+", help="Result JSON files (2-6)")
    parser.add_argument("--output", default="multi_comparison.png")
    args = parser.parse_args()

    if len(args.results) < 2:
        print("Error: Need at least 2 result files")
        return
    if len(args.results) > 6:
        print("Error: Maximum 6 result files supported")
        return

    datasets = [load(p) for p in args.results]
    labels = [short_cpu(d["meta"]["cpu"]) for d in datasets]

    # Handle duplicate labels by appending L3 cache info
    if len(set(labels)) != len(labels):
        new_labels = []
        for d, label in zip(datasets, labels):
            l3 = d["meta"].get("l3_cache", "")
            if l3:
                new_labels.append(f"{label} ({l3})")
            else:
                new_labels.append(label)
        if len(set(new_labels)) != len(new_labels):
            new_labels = [f"{l} [{i+1}]" for i, l in enumerate(new_labels)]
        labels = new_labels

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor(BG)

    runs = datasets[0]["vector_search"][0].get("runs", 1)
    fig.suptitle(
        f"x3d-rag-benchmark  |  Multi-CPU Comparison\n"
        f"RAG Vector Search CPU Performance  "
        f"({runs} runs, trimmed mean)",
        color=TEXT, fontsize=14, fontweight="bold", y=0.98
    )

    plot_qps          (axes[0][0], datasets, labels)
    plot_p99          (axes[0][1], datasets, labels)
    plot_latency_bars (axes[1][0], datasets, labels)
    plot_rag          (axes[1][1], datasets, labels)

    # Single shared legend at top, between title and charts
    handles, leg_labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, leg_labels,
               loc="upper center", bbox_to_anchor=(0.5, 0.93),
               ncol=len(datasets), facecolor=PANEL, labelcolor=TEXT,
               fontsize=10, framealpha=0.9, edgecolor=GRID)

    # Footer
    footer_parts = [f"CPU {i+1}: {d['meta']['cpu']}" for i, d in enumerate(datasets)]
    footer = "  |  ".join(footer_parts)
    footer += "  |  github.com/sorrymannn/x3d-rag-benchmark"
    fig.text(0.5, 0.01, footer, ha="center", color="#888888", fontsize=7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    plt.savefig(args.output, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
