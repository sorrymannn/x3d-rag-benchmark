# x3d-rag-benchmark

**CPU Performance Benchmark for RAG AI Pipelines**

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![FAISS](https://img.shields.io/badge/Meta-FAISS-blue.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-sentence--transformers-orange.svg)

An open-source benchmark for measuring how CPU cache and architecture affect
graph-based vector search and related stages in local/on-prem RAG pipelines.
Works with any x86 CPU (AMD, Intel).

This benchmark targets personal-PC and small-team, single-node setups
(roughly 100K–200K vectors). It is not intended to represent large-scale,
distributed vector database services.

---

## Why CPU Cache Can Matter in Graph-Based Vector Search

In graph-based ANN search such as HNSW, query performance can be sensitive to
cache behavior because traversal involves irregular memory accesses. Each hop
in the graph may reference a different region of memory, and whether that data
is in L3 cache (~4ns) or must be fetched from DRAM (~70ns) can affect throughput.

```
Typical RAG pipeline:

  User query
      ↓
  Embedding generation   (offline, or CPU/GPU)
      ↓
  Vector Search          (CPU) ← cache-sensitive in graph-based ANN
      ↓
  LLM generation         (GPU)
      ↓
  Response
```

A larger L3 cache (e.g. X3D V-Cache) can keep more of the HNSW graph resident,
potentially reducing DRAM accesses during search. This benchmark measures
whether and how much that difference shows up in practice.

---

## Benchmark Scripts

| Script | Purpose | Key Metrics |
|---|---|---|
| **benchmark.py** | Full multi-core CPU benchmark (main) | Batch QPS, Index Build, Concurrent RAG, Data Feeding |
| **benchmark_single.py** | Single-core vector search + RAG TTFT (lightweight) | Single-thread QPS, Latency P50/P95/P99, RAG TTFT |

| Compare Script | For |
|---|---|
| **compare.py** | benchmark.py results (6-chart: QPS, Build, RAG) |
| **compare_single.py** | benchmark_single.py results (3-chart: QPS, Latency, RAG) |

---

## What We Measure

### benchmark.py (main)

| Metric | Description | Scenario |
|---|---|---|
| **Batch Vector Search QPS** | All-core FAISS HNSW throughput | Multi-user / parallel retrieval |
| **Index Build Time** | HNSW construction from embeddings | One-time setup cost |
| **Concurrent RAG Throughput** | 8 workers: search(CPU) → LLM(GPU) | Shared RAG environment |
| **Concurrent RAG TTFT** | Time to first token under concurrency | User-perceived latency |
| **Data Feeding Throughput** | Tokenize(CPU) → GPU transfer rate | GPU utilization |

### benchmark_single.py (lightweight)

| Metric | Description | Cache Impact |
|---|---|---|
| **Vector Search QPS** | Single-thread FAISS HNSW queries/sec | **Direct** |
| **Vector Search Latency** | P50 / P95 / P99 breakdown | **Direct** |
| **RAG TTFT** | Time to first token (full pipeline) | Indirect |

---

## Methodology

Designed for reproducible, low-variance results:

| Design Choice | Reason |
|---|---|
| **Real Wikipedia embeddings** | Clusters by topic → stable HNSW traversal paths vs random vectors |
| **Trimmed mean** (drop outliers) | With 10 runs: ~5% each side dropped |
| **Embedding cache** | Same vectors reused across runs and machines |
| **OS-level variance controls** | CPU governor, NUMA, THP, process priority (auto-applied) |
| **Inter-run cooling** | 2s delay between runs prevents thermal drift |
| **Python GC disabled** | No garbage collection during measurement |

### DB Sizes

| Size | Suggested scenario | Notes |
|---|---|---|
| **100K vectors** | Personal RAG / small-team shared RAG | Suitable for single-user or modest shared knowledge bases |
| **200K vectors** | Small-team shared RAG / broader local knowledge base | Better for wider document coverage on a single node |

> These are rough usage tiers, not hard limits.
> Actual document/page coverage depends heavily on chunking strategy,
> chunk size, overlap, and document density.

### Recommended Environment

For lowest variance, **Linux native** (not WSL) is recommended:

```bash
# Ubuntu 24.04 recommended settings (applied automatically by benchmark)
# - CPU governor: performance
# - NUMA balancing: off
# - THP: never
# - Process priority: nice -20
```

Windows is also supported but may show higher variance due to OS interrupts.

### CV Quality Thresholds

| CV% | Rating | Action |
|---|---|---|
| ≤ 2% | **Excellent** | Stable, good for comparison |
| ≤ 3% | **Good** | Reliable for internal comparison |
| ≤ 5% | **Acceptable** | Usable with caveats |
| > 5% | **Noisy** | Try reducing background processes or reboot |

---

## Installation

### 1. Clone

```bash
git clone https://github.com/sorrymannn/x3d-rag-benchmark
cd x3d-rag-benchmark
```

### 2. Python packages

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

> `faiss-cpu` is available on PyPI and typically installs without issues on
> Linux, macOS, and Windows. If you encounter problems, see the
> [FAISS installation guide](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)
> for alternative methods including conda.

### 3. ollama + LLM model (needed for Concurrent RAG / RAG TTFT)

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
```

**Windows:**
```powershell
# Download and install from https://ollama.com/download
ollama pull llama3.2
```

> Concurrent RAG and RAG TTFT require ollama. Vector Search and Index Build benchmarks run without it.

---

## Usage

### benchmark.py (main — full multi-core)

**Linux / macOS:**
```bash
# Full benchmark (~20-30 min)
python3 benchmark.py --output 9850x3d.json

# Skip Concurrent RAG (no ollama needed)
python3 benchmark.py --skip-rag --output 9850x3d.json

# Quick test (~5 min)
python3 benchmark.py --quick --output test.json
```

**Windows:**

> **Note:** On Windows, use `python` instead of `python3` (e.g., `python benchmark.py --output 9700x.json`).

```powershell
python benchmark.py --output 9850x3d.json
python benchmark.py --skip-rag --output 9850x3d.json
python benchmark.py --quick --output test.json
```

#### Options

| Option | Default | Description |
|---|---|---|
| `--runs N` | 10 | Number of runs for batch search |
| `--skip-rag` | off | Skip concurrent RAG test |
| `--skip-feeding` | off | Skip data feeding test |
| `--skip-build` | off | Skip index build test |
| `--quick` | off | Small DB only, fewer runs |
| `--output FILE` | auto | Save results to specified JSON file |
| `--model NAME` | llama3.2 | ollama model for RAG |
| `--db-size N` | - | Override DB size (single value) |
| `--cache-dir PATH` | ./embedding_cache | Embedding cache directory |

### benchmark_single.py (lightweight — single-core)

```bash
# Vector Search only (no ollama needed)
python3 benchmark_single.py --skip-rag --output 9850x3d_single.json

# Full benchmark (Vector Search + RAG TTFT)
python3 benchmark_single.py --output 9850x3d_single.json

# Single-threaded for maximum stability
python3 benchmark_single.py --threads 1 --skip-rag --output 9850x3d_single.json
```

---

## Compare Results

### benchmark.py results (6-chart)

```bash
# Compare 2-8 CPUs
python3 compare.py 9950x3d2.json 9850x3d.json 9700x.json 285k.json 265k.json

# Custom output filename
python3 compare.py *.json --output my_comparison.png
```

Outputs `comparison.png` with 6 charts: QPS (100K/200K), Index Build (100K/200K), Concurrent RAG Throughput, Concurrent RAG TTFT. Percentage labels show difference vs the first CPU.

### benchmark_single.py results (3-chart)

```bash
python3 compare_single.py 9850x3d_single.json 9700x_single.json 285k_single.json
```

Outputs `comparison.png` with 3 charts: QPS, Search Latency (P50/P95), RAG Pipeline Latency.

---

## Sharing Embeddings Between Machines

For a fair comparison, use **identical embedding vectors** on all CPUs.

```bash
# Generate on first machine
python3 benchmark.py --output cpu_a.json

# Copy cache to second machine
scp -r ./embedding_cache/ user@other-machine:~/x3d-rag-benchmark/

# Run on second machine (loads from cache instantly)
python3 benchmark.py --output cpu_b.json
```

> Embedding cache is cross-platform compatible (numpy `.npy` files).

---

## Libraries Used

| Library | Source | Purpose |
|---|---|---|
| FAISS | Meta AI | Vector search engine |
| sentence-transformers | HuggingFace | Embedding model |
| ollama | Ollama | Local LLM server |
| datasets | HuggingFace | Wikipedia dataset |
| transformers | HuggingFace | Tokenizer (data feeding test) |

---

## Reproducibility Conditions

- Use Linux native for lowest variance (Ubuntu 24.04 recommended)
- Minimize background processes
- Reboot before benchmarking (recommended)
- Use identical RAM capacity and speed
- Use identical GPU
- Use `--runs 10` or higher for stable results
- Share `embedding_cache/` between machines for fair comparison

### BIOS Settings (recommended)

For best reproducibility, set identically on all systems:

| Setting | Recommendation |
|---|---|
| PBO / Turbo Boost | Same on all (all ON or all OFF) |
| XMP / EXPO | Same memory profile (e.g. DDR5-6000 CL30) |
| C-States | Disabled |
| Cool & Quiet | Disabled |

---

## Contributing

PRs and issues welcome.
Submit your results (JSON files) to the `results/` folder via PR.
Include: CPU model, motherboard, BIOS version, memory config, OS.
