"""
compare.py
==========
Compare two benchmark results and generate charts.

Usage:
  python3 compare.py 9700x.json 9800x3d.json
  python3 compare.py 9700x.json 9800x3d.json --output comparison.png
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Colors ────────────────────────────────────────────────────────────────────
COLOR_A = "#E84B4B"   # red (non-X3D)
COLOR_B = "#4B9BE8"   # blue (X3D)
BG      = "#1A1A2E"
PANEL   = "#16213E"
TEXT    = "#E0E0E0"
GRID    = "#2A2A4A"


def load(path):
    with open(path) as f:
        return json.load(f)


def short_cpu(name):
    """
    Extract model name from full CPU string.
    e.g. "AMD Ryzen 7 9800X3D 8-Core Processor" -> "9800X3D"
    """
    import re

    # AMD X3D: 9800X3D, 9950X3D, 7800X3D, 5800X3D
    m = re.search(r'(\d{4}X3D)', name, re.IGNORECASE)
    if m: return m.group(1).upper()

    # AMD Ryzen X-suffix: 9700X, 9900X, 7700X
    m = re.search(r'(\d{4}X\d*)', name, re.IGNORECASE)
    if m: return m.group(1).upper()

    # AMD Ryzen number only: 9700, 9900
    m = re.search(r'Ryzen\s+\d+\s+(\d{4})', name, re.IGNORECASE)
    if m: return m.group(1)

    # Intel Core: i9-14900K, i7-13700K
    m = re.search(r'(i\d-\d{4,5}\w*)', name, re.IGNORECASE)
    if m: return m.group(1)

    # Intel Ultra: Ultra 9 285K, Ultra 7 265K
    m = re.search(r'Ultra\s+(\d+\s+\d+\w*)', name, re.IGNORECASE)
    if m: return f"Ultra {m.group(1)}"

    # AMD EPYC / Threadripper: 4585PX, 7763, 3970X
    m = re.search(r'(?:EPYC|Threadripper)\s+(\w+)', name, re.IGNORECASE)
    if m: return m.group(1)

    parts = name.split()
    if len(parts) >= 4: return parts[3]
    return name[:20]


def plot_vector_search(ax, data_a, data_b, label_a, label_b):
    """Vector Search QPS by DB size with error bars"""
    sizes_a = [r["db_size"] for r in data_a["vector_search"]]
    qps_a   = [r["qps"]     for r in data_a["vector_search"]]
    err_a   = [r.get("qps_stddev", 0) for r in data_a["vector_search"]]
    sizes_b = [r["db_size"] for r in data_b["vector_search"]]
    qps_b   = [r["qps"]     for r in data_b["vector_search"]]
    err_b   = [r.get("qps_stddev", 0) for r in data_b["vector_search"]]

    x = np.arange(len(sizes_a))
    w = 0.35

    ax.bar(x - w/2, qps_a, w, label=label_a, color=COLOR_A, alpha=0.85,
           yerr=err_a, error_kw=dict(ecolor="white", capsize=4, linewidth=1.2))
    ax.bar(x + w/2, qps_b, w, label=label_b, color=COLOR_B, alpha=0.85,
           yerr=err_b, error_kw=dict(ecolor="white", capsize=4, linewidth=1.2))

    for i, (qa, qb) in enumerate(zip(qps_a, qps_b)):
        diff = (qb - qa) / qa * 100
        color = COLOR_B if diff > 0 else COLOR_A
        sign = "+" if diff > 0 else ""
        ax.text(i, max(qa, qb) * 1.08, f"{sign}{diff:.1f}%",
                ha="center", va="bottom", color=color,
                fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s//1000}K\nvectors" for s in sizes_a], color=TEXT)
    ax.set_ylabel("QPS (higher is better)", color=TEXT)
    ax.set_title("Vector Search QPS (FAISS HNSW)", color=TEXT, fontsize=12, pad=10)
    ax.legend(facecolor=PANEL, labelcolor=TEXT)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT)
    ax.spines[:].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linestyle="--", alpha=0.5)


def plot_p99_latency(ax, data_a, data_b, label_a, label_b):
    """P99 Latency comparison"""
    sizes = [r["db_size"] for r in data_a["vector_search"]]
    p99_a = [r["latency_p99_ms"] for r in data_a["vector_search"]]
    p99_b = [r["latency_p99_ms"] for r in data_b["vector_search"]]

    ax.plot(range(len(sizes)), p99_a, "o-", color=COLOR_A,
            label=label_a, linewidth=2, markersize=7)
    ax.plot(range(len(sizes)), p99_b, "o-", color=COLOR_B,
            label=label_b, linewidth=2, markersize=7)
    ax.fill_between(range(len(sizes)), p99_a, p99_b,
                    alpha=0.15, color=COLOR_B)

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{s//1000}K" for s in sizes], color=TEXT)
    ax.set_ylabel("P99 Latency (ms, lower is better)", color=TEXT)
    ax.set_title("Vector Search P99 Latency", color=TEXT, fontsize=12, pad=10)
    ax.legend(facecolor=PANEL, labelcolor=TEXT)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT)
    ax.spines[:].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linestyle="--", alpha=0.5)


def plot_concurrent(ax, data_a, data_b, label_a, label_b):
    """Concurrent Search QPS"""
    conc_a = sorted(data_a["concurrent_search"].keys(), key=int)
    conc_b = sorted(data_b["concurrent_search"].keys(), key=int)

    qps_a = [data_a["concurrent_search"][k]["qps"] for k in conc_a]
    qps_b = [data_b["concurrent_search"][k]["qps"] for k in conc_b]

    x = np.arange(len(conc_a))
    w = 0.35

    err_a = [data_a["concurrent_search"][k].get("qps_stddev", 0) for k in conc_a]
    err_b = [data_b["concurrent_search"][k].get("qps_stddev", 0) for k in conc_b]

    ax.bar(x - w/2, qps_a, w, label=label_a, color=COLOR_A, alpha=0.85,
           yerr=err_a, error_kw=dict(ecolor="white", capsize=4, linewidth=1.2))
    ax.bar(x + w/2, qps_b, w, label=label_b, color=COLOR_B, alpha=0.85,
           yerr=err_b, error_kw=dict(ecolor="white", capsize=4, linewidth=1.2))

    for i, (qa, qb) in enumerate(zip(qps_a, qps_b)):
        diff = (qb - qa) / qa * 100
        color = COLOR_B if diff > 0 else COLOR_A
        sign = "+" if diff > 0 else ""
        ax.text(i, max(qa, qb) * 1.08, f"{sign}{diff:.1f}%",
                ha="center", va="bottom", color=color,
                fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{k} concurrent" for k in conc_a], color=TEXT)
    ax.set_ylabel("QPS (higher is better)", color=TEXT)
    ax.set_title("Concurrent Search QPS", color=TEXT, fontsize=12, pad=10)
    ax.legend(facecolor=PANEL, labelcolor=TEXT)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT)
    ax.spines[:].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linestyle="--", alpha=0.5)


def plot_rag_ttft(ax, data_a, data_b, label_a, label_b):
    """RAG TTFT comparison"""
    rag_a = data_a.get("rag_ttft")
    rag_b = data_b.get("rag_ttft")

    if not rag_a or not rag_b:
        ax.text(0.5, 0.5, "RAG TTFT data not available\n(run without --skip-rag)",
                ha="center", va="center", color=TEXT, fontsize=11,
                transform=ax.transAxes)
        ax.set_facecolor(PANEL)
        ax.set_title("RAG End-to-End TTFT", color=TEXT, fontsize=12, pad=10)
        return

    metrics = ["P50", "P95", "P99"]
    vs_a = [rag_a["vector_search"]["p50_ms"],
            rag_a["vector_search"]["p95_ms"],
            rag_a["vector_search"]["p99_ms"]]
    vs_b = [rag_b["vector_search"]["p50_ms"],
            rag_b["vector_search"]["p95_ms"],
            rag_b["vector_search"]["p99_ms"]]

    x = np.arange(len(metrics))
    w = 0.35

    ax.bar(x - w/2, vs_a, w, label=label_a, color=COLOR_A, alpha=0.85)
    ax.bar(x + w/2, vs_b, w, label=label_b, color=COLOR_B, alpha=0.85)

    for i, (va, vb) in enumerate(zip(vs_a, vs_b)):
        diff = (vb - va) / va * 100
        color = COLOR_B if diff < 0 else COLOR_A
        sign = "+" if diff > 0 else ""
        ax.text(i, max(va, vb) * 1.05, f"{sign}{diff:.1f}%",
                ha="center", va="bottom", color=color,
                fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color=TEXT)
    ax.set_ylabel("Vector Search Latency (ms)", color=TEXT)
    ax.set_title("RAG Vector Search Latency\n(lower is better)", color=TEXT, fontsize=12, pad=10)
    ax.legend(facecolor=PANEL, labelcolor=TEXT)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT)
    ax.spines[:].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linestyle="--", alpha=0.5)


def main():
    parser = argparse.ArgumentParser(description="x3d-rag-benchmark result comparison")
    parser.add_argument("result_a", help="First result JSON (non-X3D)")
    parser.add_argument("result_b", help="Second result JSON (X3D)")
    parser.add_argument("--output", default="comparison.png",
                        help="Output image path (default: comparison.png)")
    args = parser.parse_args()

    data_a = load(args.result_a)
    data_b = load(args.result_b)

    label_a = short_cpu(data_a["meta"]["cpu"])
    label_b = short_cpu(data_b["meta"]["cpu"])

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(BG)

    fig.suptitle(
        f"x3d-rag-benchmark  |  {label_a} vs {label_b}\n"
        f"RAG Vector Search CPU Performance",
        color=TEXT, fontsize=14, fontweight="bold", y=0.98
    )

    plot_vector_search(axes[0][0], data_a, data_b, label_a, label_b)
    plot_p99_latency  (axes[0][1], data_a, data_b, label_a, label_b)
    plot_concurrent   (axes[1][0], data_a, data_b, label_a, label_b)
    plot_rag_ttft     (axes[1][1], data_a, data_b, label_a, label_b)

    meta_text = (
        f"CPU A: {data_a['meta']['cpu']}  |  "
        f"CPU B: {data_b['meta']['cpu']}  |  "
        f"github.com/sorrymannn/x3d-rag-benchmark"
    )
    fig.text(0.5, 0.01, meta_text, ha="center", color="#888888", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(args.output, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
