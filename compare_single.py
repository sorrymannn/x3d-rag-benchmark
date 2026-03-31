"""
compare.py
================
3-chart comparison focused on CPU cache impact:
  1. Vector Search QPS (throughput)
  2. Vector Search Latency P50/P95/P99 (largest DB)
  3. RAG Vector Search Latency P50/P95/P99 (in-pipeline)

Usage:
  python compare.py 9850x3d.json 9700x.json 285k.json 265k.json
  python compare.py *.json --output comparison.png
"""

import argparse
import json
import re

import matplotlib.pyplot as plt
import numpy as np

# ── Colors (9850X3D=red, 9700X=coral, 285K=blue, 265K=teal) ─────────────
COLORS = ["#E24B4A", "#F0997B", "#85B7EB", "#5DCAA5", "#B84BE8", "#E84B9B"]
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


# ── Plot 1: Vector Search QPS (all DB sizes) ─────────────────────────────

def plot_qps(ax, datasets, labels):
    n = len(datasets)
    all_sizes = [r["db_size"] for r in datasets[0]["vector_search"]]
    n_sizes = len(all_sizes)
    total_w = 0.75
    w = total_w / n

    base_qps = [datasets[0]["vector_search"][i]["qps"] for i in range(n_sizes)]

    for j, (d, label) in enumerate(zip(datasets, labels)):
        qps = [d["vector_search"][i]["qps"] for i in range(n_sizes)]
        x = np.arange(n_sizes) - total_w / 2 + w * j + w / 2
        ax.bar(x, qps, w * 0.82, label=label, color=COLORS[j], alpha=0.92,
               edgecolor="white", linewidth=0.3)
        for i in range(n_sizes):
            val_label(ax, x[i], qps[i], f"{qps[i]:,.0f}", fontsize=8)
            if j > 0:
                diff_label(ax, x[i], qps[i], qps[i], base_qps[i],
                           higher_better=True, fontsize=7.5)

    ax.set_xticks(np.arange(n_sizes))
    ax.set_xticklabels([f"{s // 1000}K" for s in all_sizes], color=TEXT, fontsize=10)
    ax.set_title("Vector Search QPS (throughput)", color=TEXT, fontsize=13,
                  fontweight="bold", pad=10)
    ax.set_ylim(bottom=5000)
    style_ax(ax, "QPS (higher = better)")


# ── Plot 2: Vector Search Latency P50/P95/P99 (largest DB, horizontal) ───

def plot_vs_latency(ax, datasets, labels):
    n = len(datasets)
    metrics = ["P50", "P95", "P99"]
    n_m = len(metrics)
    total_h = 0.75
    h = total_h / n

    r_base = datasets[0]["vector_search"][-1]
    db_size = r_base["db_size"]
    base_vals = [r_base["latency_p50_ms"], r_base["latency_p95_ms"], r_base["latency_p99_ms"]]

    max_val = 0
    for j, (d, label) in enumerate(zip(datasets, labels)):
        r = d["vector_search"][-1]
        vals = [r["latency_p50_ms"], r["latency_p95_ms"], r["latency_p99_ms"]]
        max_val = max(max_val, max(vals))
        y = np.arange(n_m) - total_h / 2 + h * (n - 1 - j) + h / 2
        ax.barh(y, vals, h * 0.82, label=label, color=COLORS[j], alpha=0.92,
                edgecolor="white", linewidth=0.3)
        for i in range(n_m):
            val_str = f"{vals[i]:.3f} ms"
            if j > 0:
                pct = (vals[i] - base_vals[i]) / base_vals[i] * 100
                color = DIFF_WORSE
                val_str += f"  (+{pct:.0f}%)"
            else:
                color = TEXT
            ax.text(vals[i] + 0.003, y[i], val_str, ha="left", va="center",
                    color=color, fontsize=8, fontweight="bold")

    ax.set_yticks(np.arange(n_m))
    ax.set_yticklabels(metrics, color=TEXT, fontsize=10)
    ax.invert_yaxis()
    ax.set_title(f"Vector Search Latency ({db_size // 1000}K vectors)",
                  color=TEXT, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Latency ms (lower = better)", color=MUTED, fontsize=10)
    ax.set_xlim(right=max_val * 1.6)
    style_ax(ax)


# ── Plot 3: RAG Vector Search Latency P50/P95/P99 (horizontal) ───────────

def plot_rag_vs(ax, datasets, labels):
    has_rag = all(d.get("rag_ttft") for d in datasets)
    if not has_rag:
        ax.text(0.5, 0.5, "RAG data not available\n(run without --skip-rag)",
                ha="center", va="center", color=MUTED, fontsize=11, transform=ax.transAxes)
        ax.set_facecolor(PANEL)
        ax.set_title("RAG Pipeline — Vector Search", color=TEXT, fontsize=13,
                      fontweight="bold", pad=10)
        return

    n = len(datasets)
    metrics = ["P50", "P95", "P99"]
    keys = ["p50_ms", "p95_ms", "p99_ms"]
    n_m = len(metrics)
    total_h = 0.75
    h = total_h / n

    base_vals = [datasets[0]["rag_ttft"]["vector_search"][k] for k in keys]

    max_val = 0
    for j, (d, label) in enumerate(zip(datasets, labels)):
        vals = [d["rag_ttft"]["vector_search"][k] for k in keys]
        max_val = max(max_val, max(vals))
        y = np.arange(n_m) - total_h / 2 + h * (n - 1 - j) + h / 2
        ax.barh(y, vals, h * 0.82, label=label, color=COLORS[j], alpha=0.92,
                edgecolor="white", linewidth=0.3)
        for i in range(n_m):
            val_str = f"{vals[i]:.3f} ms"
            if j > 0:
                pct = (vals[i] - base_vals[i]) / base_vals[i] * 100
                color = DIFF_WORSE
                val_str += f"  (+{pct:.0f}%)"
            else:
                color = TEXT
            ax.text(vals[i] + 0.005, y[i], val_str, ha="left", va="center",
                    color=color, fontsize=8, fontweight="bold")

    ax.set_yticks(np.arange(n_m))
    ax.set_yticklabels(metrics, color=TEXT, fontsize=10)
    ax.invert_yaxis()
    ax.set_title("RAG Pipeline — Vector Search Latency", color=TEXT, fontsize=13,
                  fontweight="bold", pad=10)
    ax.set_xlabel("Latency ms (lower = better)", color=MUTED, fontsize=10)
    ax.set_xlim(right=max_val * 1.55)
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

    # Build labels: CPU short name + L3 size
    labels = []
    for d in datasets:
        name = short_cpu(d["meta"]["cpu"])
        l3 = get_l3(d)
        labels.append(f"{name}\n({l3} L3)" if l3 else name)

    fig = plt.figure(figsize=(22, 9))
    fig.patch.set_facecolor(BG)

    runs = datasets[0]["vector_search"][0].get("runs", 1)
    fig.suptitle(
        f"x3d-rag-benchmark  —  CPU Performance in RAG AI Pipeline  ({runs} runs, trimmed mean)",
        color=TEXT, fontsize=14, fontweight="bold", y=0.98
    )

    # Create gridspec: legend row (small) + chart row (large)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 20],
                          left=0.05, right=0.98, top=0.93, bottom=0.06,
                          wspace=0.25, hspace=0.15)

    # Legend in top-center cell
    ax_leg = fig.add_subplot(gs[0, :])
    ax_leg.axis("off")

    # Chart axes
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[1, 2])

    plot_qps       (ax1, datasets, labels)
    plot_vs_latency(ax2, datasets, labels)
    plot_rag_vs    (ax3, datasets, labels)

    # Draw legend in the legend row
    handles, leg_labels = ax1.get_legend_handles_labels()
    ax_leg.legend(handles, leg_labels,
                  loc="center", ncol=len(datasets),
                  facecolor=PANEL, labelcolor=TEXT,
                  fontsize=10, framealpha=0.95, edgecolor=GRID)

    # Footer
    footer_parts = []
    for i, d in enumerate(datasets):
        cpu = d["meta"]["cpu"]
        l3 = d["meta"].get("l3_cache", "")
        footer_parts.append(f"{cpu} ({l3})")
    fig.text(0.5, 0.005,
             "  |  ".join(footer_parts) + "  |  github.com/sorrymannn/x3d-rag-benchmark",
             ha="center", color="#999999", fontsize=7)

    plt.savefig(args.output, dpi=150, facecolor=BG)
    print(f"Saved: {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
