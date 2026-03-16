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
import re

import matplotlib.pyplot as plt
import numpy as np

# ── Colors ────────────────────────────────────────────────────────────────────
COLOR_A = "#E84B4B"
COLOR_B = "#4B9BE8"
BG      = "#1A1A2E"
PANEL   = "#16213E"
TEXT    = "#E0E0E0"
GRID    = "#2A2A4A"


def load(path):
    with open(path) as f:
        return json.load(f)


def short_cpu(name):
    # AMD Ryzen X3D: 9800X3D, 9950X3D, 9850X3D, 7800X3D
    m = re.search(r'(\d{4}X3D)', name, re.IGNORECASE)
    if m: return m.group(1).upper()

    # AMD Ryzen X suffix: 9700X, 9900X, 7700X, 5600X
    m = re.search(r'(\d{4}X\b)', name, re.IGNORECASE)
    if m: return m.group(1).upper()

    # AMD Ryzen XT: 3800XT
    m = re.search(r'(\d{4}XT\b)', name, re.IGNORECASE)
    if m: return m.group(1).upper()

    # AMD Ryzen number only: 9700, 9900
    m = re.search(r'Ryzen\s+\d+\s+(\d{4})\b', name, re.IGNORECASE)
    if m: return m.group(1)

    # AMD EPYC: 4585PX, 4564P, 9654
    m = re.search(r'EPYC\s+(\d{4}\w*)', name, re.IGNORECASE)
    if m: return f"EPYC {m.group(1).upper()}"

    # AMD Threadripper
    m = re.search(r'Threadripper\s+(?:PRO\s+)?(\d{4}\w*)', name, re.IGNORECASE)
    if m: return f"TR {m.group(1).upper()}"

    # Intel Core Ultra: 285K, 265K, 225K
    m = re.search(r'Ultra\s+(\d+)\s+(\d{3}\w*)', name, re.IGNORECASE)
    if m: return f"Ultra {m.group(1)} {m.group(2).upper()}"

    # Intel Core i-series: i9-14900K, i7-13700KF
    m = re.search(r'(i\d-\d{4,5}\w*)', name, re.IGNORECASE)
    if m: return m.group(1)

    # Intel Xeon
    m = re.search(r'Xeon\s+(?:W-|Gold\s+|Platinum\s+|Silver\s+)?(\w+)',
                  name, re.IGNORECASE)
    if m: return f"Xeon {m.group(1)}"

    # Fallback
    m = re.search(r'\b(\d{3,5}[A-Z0-9]*)\b', name, re.IGNORECASE)
    if m: return m.group(1).upper()

    parts = name.split()
    return " ".join(parts[2:4]) if len(parts) >= 4 else name[:20]


def bar_with_err(ax, x, w, vals, errs, color, label):
    ax.bar(x, vals, w, label=label, color=color, alpha=0.85,
           yerr=errs,
           error_kw=dict(ecolor="white", capsize=4, linewidth=1.2))


def diff_label(ax, i, va, vb, ybase, higher_is_better=True):
    diff  = (vb - va) / va * 100
    sign  = "+" if diff > 0 else ""
    if higher_is_better:
        color = COLOR_B if diff > 0 else COLOR_A
    else:
        color = COLOR_B if diff < 0 else COLOR_A
    ax.text(i, ybase * 1.08, f"{sign}{diff:.1f}%",
            ha="center", va="bottom", color=color,
            fontsize=9, fontweight="bold")


def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT)
    ax.spines[:].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linestyle="--", alpha=0.5)


# ── Plot 1: Vector Search QPS ─────────────────────────────────────────────────

def plot_qps(ax, da, db, la, lb):
    sizes = [r["db_size"] for r in da["vector_search"]]
    qps_a = [r["qps"]     for r in da["vector_search"]]
    qps_b = [r["qps"]     for r in db["vector_search"]]
    err_a = [r.get("qps_stddev", 0) for r in da["vector_search"]]
    err_b = [r.get("qps_stddev", 0) for r in db["vector_search"]]

    x, w = np.arange(len(sizes)), 0.35
    bar_with_err(ax, x - w/2, w, qps_a, err_a, COLOR_A, la)
    bar_with_err(ax, x + w/2, w, qps_b, err_b, COLOR_B, lb)

    for i, (va, vb) in enumerate(zip(qps_a, qps_b)):
        diff_label(ax, i, va, vb, max(va, vb), higher_is_better=True)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s//1000}K\nvectors" for s in sizes], color=TEXT)
    ax.set_ylabel("QPS (higher is better)", color=TEXT)
    ax.set_title("Vector Search QPS (FAISS HNSW)", color=TEXT, fontsize=12, pad=10)
    ax.legend(facecolor=PANEL, labelcolor=TEXT)
    style_ax(ax)


# ── Plot 2: P99 Latency line ──────────────────────────────────────────────────

def plot_p99(ax, da, db, la, lb):
    sizes = [r["db_size"]          for r in da["vector_search"]]
    p99_a = [r["latency_p99_ms"]   for r in da["vector_search"]]
    p99_b = [r["latency_p99_ms"]   for r in db["vector_search"]]
    err_a = [r.get("latency_p99_stddev", 0) for r in da["vector_search"]]
    err_b = [r.get("latency_p99_stddev", 0) for r in db["vector_search"]]

    xs = range(len(sizes))
    ax.errorbar(xs, p99_a, yerr=err_a, fmt="o-", color=COLOR_A,
                label=la, linewidth=2, markersize=7, capsize=4)
    ax.errorbar(xs, p99_b, yerr=err_b, fmt="o-", color=COLOR_B,
                label=lb, linewidth=2, markersize=7, capsize=4)
    ax.fill_between(xs, p99_a, p99_b, alpha=0.12, color=COLOR_B)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s//1000}K" for s in sizes], color=TEXT)
    ax.set_ylabel("P99 Latency (ms, lower is better)", color=TEXT)
    ax.set_title("Vector Search P99 Latency", color=TEXT, fontsize=12, pad=10)
    ax.legend(facecolor=PANEL, labelcolor=TEXT)
    style_ax(ax)


# ── Plot 3: P50/P95/P99 bar (largest DB) ─────────────────────────────────────

def plot_latency_bars(ax, da, db, la, lb):
    r_a = da["vector_search"][-1]
    r_b = db["vector_search"][-1]

    metrics = ["P50", "P95", "P99"]
    vals_a  = [r_a["latency_p50_ms"], r_a["latency_p95_ms"], r_a["latency_p99_ms"]]
    vals_b  = [r_b["latency_p50_ms"], r_b["latency_p95_ms"], r_b["latency_p99_ms"]]
    err_a   = [r_a.get("latency_p50_stddev", 0), 0,
               r_a.get("latency_p99_stddev", 0)]
    err_b   = [r_b.get("latency_p50_stddev", 0), 0,
               r_b.get("latency_p99_stddev", 0)]

    x, w = np.arange(len(metrics)), 0.35
    bar_with_err(ax, x - w/2, w, vals_a, err_a, COLOR_A, la)
    bar_with_err(ax, x + w/2, w, vals_b, err_b, COLOR_B, lb)

    for i, (va, vb) in enumerate(zip(vals_a, vals_b)):
        diff_label(ax, i, va, vb, max(va, vb), higher_is_better=False)

    db_size = r_a["db_size"]
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color=TEXT)
    ax.set_ylabel("Latency (ms, lower is better)", color=TEXT)
    ax.set_title(f"Search Latency Distribution\n({db_size//1000}K vectors, largest DB)",
                 color=TEXT, fontsize=12, pad=10)
    ax.legend(facecolor=PANEL, labelcolor=TEXT)
    style_ax(ax)


# ── Plot 4: RAG TTFT ──────────────────────────────────────────────────────────

def plot_rag_ttft(ax, da, db, la, lb):
    rag_a = da.get("rag_ttft")
    rag_b = db.get("rag_ttft")

    if not rag_a or not rag_b:
        ax.text(0.5, 0.5,
                "RAG TTFT data not available\n(run without --skip-rag)",
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
    err_a = [rag_a["vector_search"].get("stddev_ms", 0)] * 3
    err_b = [rag_b["vector_search"].get("stddev_ms", 0)] * 3

    x, w = np.arange(len(metrics)), 0.35
    bar_with_err(ax, x - w/2, w, vs_a, err_a, COLOR_A, la)
    bar_with_err(ax, x + w/2, w, vs_b, err_b, COLOR_B, lb)

    for i, (va, vb) in enumerate(zip(vs_a, vs_b)):
        diff_label(ax, i, va, vb, max(va, vb), higher_is_better=False)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color=TEXT)
    ax.set_ylabel("Vector Search Latency in RAG (ms)", color=TEXT)
    ax.set_title("RAG Pipeline — Vector Search Latency\n(lower is better)",
                 color=TEXT, fontsize=12, pad=10)
    ax.legend(facecolor=PANEL, labelcolor=TEXT)
    style_ax(ax)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_a", help="First result JSON (non-X3D)")
    parser.add_argument("result_b", help="Second result JSON (X3D)")
    parser.add_argument("--output", default="comparison.png")
    args = parser.parse_args()

    da = load(args.result_a)
    db = load(args.result_b)
    la = short_cpu(da["meta"]["cpu"])
    lb = short_cpu(db["meta"]["cpu"])

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(BG)

    runs_a = da["vector_search"][0].get("runs", 1)
    runs_b = db["vector_search"][0].get("runs", 1)
    fig.suptitle(
        f"x3d-rag-benchmark  |  {la} vs {lb}\n"
        f"RAG Vector Search CPU Performance  "
        f"({runs_a} runs, trimmed mean)",
        color=TEXT, fontsize=14, fontweight="bold", y=0.98
    )

    plot_qps          (axes[0][0], da, db, la, lb)
    plot_p99          (axes[0][1], da, db, la, lb)
    plot_latency_bars (axes[1][0], da, db, la, lb)
    plot_rag_ttft     (axes[1][1], da, db, la, lb)

    fig.text(
        0.5, 0.01,
        f"CPU A: {da['meta']['cpu']}  (L3: {da['meta'].get('l3_cache', '?')})"
        f"  |  CPU B: {db['meta']['cpu']}  (L3: {db['meta'].get('l3_cache', '?')})"
        f"  |  github.com/sorrymannn/x3d-rag-benchmark",
        ha="center", color="#888888", fontsize=8
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(args.output, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
