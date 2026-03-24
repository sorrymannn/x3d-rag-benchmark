"""
compare.py
================
Designed for slides — minimal, impactful.

Usage:
  python compare.py result1.json result2.json result3.json result4.json
  python compare.py *.json --output comparison.png
"""

import argparse
import json
import re

import matplotlib.pyplot as plt
import numpy as np

# ── Colors (9850X3D=red, 9700X=gold, 285K=blue, 265K=green) ──────────────
COLORS = ["#E84B4B", "#E8B84B", "#4B9BE8", "#4BE88A", "#B84BE8", "#E84B9B"]
BG     = "#1A1A2E"
PANEL  = "#16213E"
TEXT   = "#E0E0E0"
GRID   = "#2A2A4A"


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
    m = re.search(r'Ultra\s+(\d+)\s+(\d{3}\w*)', name, re.IGNORECASE)
    if m: return f"Ultra {m.group(1)} {m.group(2).upper()}"
    m = re.search(r'(i\d-\d{4,5}\w*)', name, re.IGNORECASE)
    if m: return m.group(1)
    m = re.search(r'\b(\d{3,5}[A-Z0-9]*)\b', name, re.IGNORECASE)
    if m: return m.group(1).upper()
    parts = name.split()
    return " ".join(parts[2:4]) if len(parts) >= 4 else name[:20]


def diff_text(ax, x_pos, val, base_val, higher_is_better=True):
    if base_val == 0:
        return
    diff = (val - base_val) / base_val * 100
    sign = "+" if diff > 0 else ""
    if higher_is_better:
        color = "#4BE88A" if diff > 0 else "#E84B4B"
    else:
        color = "#4BE88A" if diff < 0 else "#E84B4B"
    ax.text(x_pos, val, f"{sign}{diff:.0f}%",
            ha="center", va="bottom", color=color,
            fontsize=9, fontweight="bold")


def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT)
    ax.spines[:].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linestyle="--", alpha=0.5)


# ── Plot 1: QPS (500K + 1000K only) ──────────────────────────────────────

def plot_qps(ax, datasets, labels):
    n = len(datasets)
    # Pick 500K and 1000K
    all_sizes = [r["db_size"] for r in datasets[0]["vector_search"]]
    pick_sizes = [s for s in all_sizes if s >= 500_000]
    pick_indices = [all_sizes.index(s) for s in pick_sizes]

    n_sizes = len(pick_sizes)
    total_w = 0.7
    w = total_w / n

    base_qps = [datasets[0]["vector_search"][i]["qps"] for i in pick_indices]

    for j, (d, label) in enumerate(zip(datasets, labels)):
        qps = [d["vector_search"][i]["qps"] for i in pick_indices]
        x = np.arange(n_sizes) - total_w/2 + w*j + w/2
        ax.bar(x, qps, w * 0.85, label=label, color=COLORS[j], alpha=0.88)
        if j > 0:
            for i in range(n_sizes):
                diff_text(ax, x[i], qps[i], base_qps[i], higher_is_better=True)

    ax.set_xticks(np.arange(n_sizes))
    ax.set_xticklabels([f"{s//1000}K vectors" for s in pick_sizes], color=TEXT, fontsize=11)
    ax.set_ylabel("QPS (higher is better)", color=TEXT, fontsize=11)
    ax.set_title("Vector Search QPS (FAISS HNSW)", color=TEXT, fontsize=14, fontweight="bold", pad=12)
    style_ax(ax)


# ── Plot 2: Latency Distribution P50 + P95 (largest DB) ──────────────────

def plot_latency(ax, datasets, labels):
    n = len(datasets)
    metrics = ["P50", "P95"]
    n_metrics = len(metrics)
    total_w = 0.7
    w = total_w / n

    r_base = datasets[0]["vector_search"][-1]
    base_vals = [r_base["latency_p50_ms"], r_base["latency_p95_ms"]]

    for j, (d, label) in enumerate(zip(datasets, labels)):
        r = d["vector_search"][-1]
        vals = [r["latency_p50_ms"], r["latency_p95_ms"]]
        x = np.arange(n_metrics) - total_w/2 + w*j + w/2
        ax.bar(x, vals, w * 0.85, label=label, color=COLORS[j], alpha=0.88)
        if j > 0:
            for i in range(n_metrics):
                diff_text(ax, x[i], vals[i], base_vals[i], higher_is_better=False)

    db_size = datasets[0]["vector_search"][-1]["db_size"]
    ax.set_xticks(np.arange(n_metrics))
    ax.set_xticklabels(metrics, color=TEXT, fontsize=11)
    ax.set_ylabel("Latency (ms, lower is better)", color=TEXT, fontsize=11)
    ax.set_title(f"Search Latency ({db_size//1000}K vectors)",
                 color=TEXT, fontsize=14, fontweight="bold", pad=12)
    style_ax(ax)


# ── Plot 3: RAG Pipeline P50 + P95 ───────────────────────────────────────

def plot_rag(ax, datasets, labels):
    has_rag = all(d.get("rag_ttft") for d in datasets)
    if not has_rag:
        ax.text(0.5, 0.5, "RAG data not available\n(run without --skip-rag)",
                ha="center", va="center", color=TEXT, fontsize=11, transform=ax.transAxes)
        ax.set_facecolor(PANEL)
        ax.set_title("RAG Pipeline", color=TEXT, fontsize=14, fontweight="bold", pad=12)
        return

    n = len(datasets)
    metrics = ["P50", "P95"]
    n_metrics = len(metrics)
    total_w = 0.7
    w = total_w / n

    rag_base = datasets[0]["rag_ttft"]
    base_vals = [rag_base["vector_search"]["p50_ms"], rag_base["vector_search"]["p95_ms"]]

    for j, (d, label) in enumerate(zip(datasets, labels)):
        rag = d["rag_ttft"]
        vals = [rag["vector_search"]["p50_ms"], rag["vector_search"]["p95_ms"]]
        x = np.arange(n_metrics) - total_w/2 + w*j + w/2
        ax.bar(x, vals, w * 0.85, label=label, color=COLORS[j], alpha=0.88)
        if j > 0:
            for i in range(n_metrics):
                diff_text(ax, x[i], vals[i], base_vals[i], higher_is_better=False)

    ax.set_xticks(np.arange(n_metrics))
    ax.set_xticklabels(metrics, color=TEXT, fontsize=11)
    ax.set_ylabel("Vector Search Latency in RAG (ms)", color=TEXT, fontsize=11)
    ax.set_title("RAG Pipeline — Vector Search Latency\n(lower is better)",
                 color=TEXT, fontsize=14, fontweight="bold", pad=12)
    style_ax(ax)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="comparison (2-6 CPUs)")
    parser.add_argument("results", nargs="+", help="Result JSON files (2-6)")
    parser.add_argument("--output", default="comparison.png")
    args = parser.parse_args()

    if len(args.results) < 2 or len(args.results) > 6:
        print("Error: Need 2-6 result files")
        return

    datasets = [load(p) for p in args.results]
    labels = [short_cpu(d["meta"]["cpu"]) for d in datasets]

    if len(set(labels)) != len(labels):
        new_labels = []
        for d, label in zip(datasets, labels):
            l3 = d["meta"].get("l3_cache", "")
            new_labels.append(f"{label} ({l3})" if l3 else label)
        if len(set(new_labels)) != len(new_labels):
            new_labels = [f"{l} [{i+1}]" for i, l in enumerate(new_labels)]
        labels = new_labels

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor(BG)

    runs = datasets[0]["vector_search"][0].get("runs", 1)
    fig.suptitle(
        f"x3d-rag-benchmark  |  RAG Vector Search CPU Performance  ({runs} runs, trimmed mean)",
        color=TEXT, fontsize=15, fontweight="bold", y=0.98
    )

    plot_qps    (axes[0], datasets, labels)
    plot_latency(axes[1], datasets, labels)
    plot_rag    (axes[2], datasets, labels)

    # Shared legend at top
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels,
               loc="upper center", bbox_to_anchor=(0.5, 0.92),
               ncol=len(datasets), facecolor=PANEL, labelcolor=TEXT,
               fontsize=11, framealpha=0.9, edgecolor=GRID)

    # Footer
    footer_parts = [f"{labels[i]}: {d['meta']['cpu']}" for i, d in enumerate(datasets)]
    fig.text(0.5, 0.01, "  |  ".join(footer_parts) + "  |  github.com/sorrymannn/x3d-rag-benchmark",
             ha="center", color="#888888", fontsize=7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.88])
    plt.savefig(args.output, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
