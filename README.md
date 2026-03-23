# x3d-rag-benchmark

**CPU Performance Benchmark for RAG AI Pipelines**

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![FAISS](https://img.shields.io/badge/Meta-FAISS-blue.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-sentence--transformers-orange.svg)

An open-source benchmark measuring the impact of CPU cache and architecture
on RAG AI pipelines. Works with any x86 CPU (AMD, Intel).

---

## Why X3D Has an Advantage in RAG

Vector Search — the CPU's core task in RAG pipelines — is bottlenecked by
Random Memory Access Pattern. X3D V-Cache's large L3 cache (96MB) provides
a performance advantage through the same mechanism that makes it strong in gaming.

```
RAG Pipeline:

  User query
      ↓
  Embedding generation   (GPU)
      ↓
  Vector Search          (CPU) ← L3 cache effect
      ↓
  LLM generation         (GPU)
      ↓
  Response

Vector Search characteristics:
  - Randomly traverses HNSW graph nodes
  - Each traversal accesses a different memory address (Random Access)
  - Larger L3 cache → higher cache hit rate → lower latency
```

---

## Supported CPUs

Any x86 CPU supported by FAISS can be tested. The benchmark automatically detects CPU model, L3 cache size, and available instruction sets (AVX2 / AVX-512).

Example tested configurations include AMD Ryzen X3D (96MB L3), AMD Ryzen non-X3D (32MB L3), and Intel Core Ultra series.

---

## What We Measure

| Metric | Description | Cache Impact |
|---|---|---|
| Vector Search QPS | FAISS HNSW queries per second | **Direct** |
| Vector Search P99 Latency | Worst-case search latency | **Direct** |
| Latency Distribution | P50 / P95 / P99 breakdown | **Direct** |
| RAG TTFT | Time to first token (full pipeline) | Indirect |

---

## Methodology

Designed for reproducible, low-variance results:

| Design Choice | Reason |
|---|---|
| **Real Wikipedia embeddings** | Clusters by topic → stable HNSW traversal paths vs random vectors |
| **Configurable FAISS threads** | Default: all cores (auto). Use `--threads 1` if CV > 3% |
| **Trimmed mean** (drop outliers) | With 5 runs: drops 1 high + 1 low; with 10+: ~5% each side |
| **Embedding cache** | Same vectors reused across runs and machines |
| **OS-level variance controls** | CPU governor, NUMA, THP, process priority (auto-applied) |
| **Inter-run cooling** | 2s delay between runs prevents thermal drift |
| **Python GC disabled** | No garbage collection during measurement |

### Recommended Environment

For lowest variance, **Linux native** (not WSL) is recommended:

```bash
# Ubuntu 24.04 recommended settings (applied automatically by benchmark)
# - CPU governor: performance
# - NUMA balancing: off
# - THP: never
# - Process priority: nice -20
```

Windows is also supported but P99 may show higher variance due to OS interrupts.

### AVX-512 / AVX2

FAISS detects the highest supported instruction set at runtime. You can verify with:

```python
python -c "import faiss; print(faiss.get_compile_options())"
```

- **AMD Ryzen (Zen 5)**: Reports AVX-512 when enabled in BIOS
- **Intel Arrow Lake**: AVX-512 not supported, uses AVX2
- **BIOS AVX-512 DISABLE**: All CPUs use AVX2 (level playing field)

### CV Quality Thresholds

| CV% | Rating | Action |
|---|---|---|
| ≤ 2% | **EXCELLENT** | Very stable, publication quality |
| ≤ 3% | **GOOD** | Reliable for comparison |
| ≤ 5% | **OK** | Acceptable |
| > 5% | **NOISY** | Try `--threads 1` or reboot |

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

> **Note for Windows**: `faiss-cpu` installs directly via pip. No build tools required.

### 3. ollama + LLM model (only needed for RAG TTFT)

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

> RAG TTFT measurement is optional. Vector Search benchmark runs without ollama.

---

## Usage

**Linux / macOS:**
```bash
# Vector Search only (no ollama needed, ~30-45 min)
python3 benchmark.py --skip-rag --output 9850x3d.json  # use your CPU name

# Full benchmark (Vector Search + RAG TTFT)
python3 benchmark.py --output 9850x3d.json  # use your CPU name

# Quick test (~5 min)
python3 benchmark.py --quick --skip-rag
```

**Windows:**
```powershell
python benchmark.py --skip-rag --output 9850x3d.json  # use your CPU name
python benchmark.py --output 9850x3d.json  # use your CPU name
python benchmark.py --quick --skip-rag
```

### Options

| Option | Default | Description |
|---|---|---|
| `--runs N` | 5 | Number of runs to average |
| `--skip-rag` | off | Skip RAG TTFT, run Vector Search only |
| `--quick` | off | Small DB only, 3 runs |
| `--output FILE` | auto | Save results to specified JSON file |
| `--model NAME` | llama3.2 | ollama model for RAG TTFT |
| `--queries N` | 300 | Number of queries per run |
| `--db-size N` | - | Override DB size (single value) |
| `--cache-dir PATH` | ./embedding_cache | Embedding cache directory |
| `--rebuild` | off | Force rebuild embedding cache |
| `--threads N` | 0 (auto) | FAISS threads: 0=all cores, 1=single-threaded |

---

## Compare Results

Three comparison scripts are available:

### compare.py — 2 CPU comparison (detailed)

```bash
python3 compare.py 9700x.json 9850x3d.json  # compare two CPU results
```

Outputs `comparison.png` with 4 charts: QPS, P99 Latency, Latency Distribution, RAG Pipeline.

### compare_multi.py — 2-6 CPU comparison

```bash
python3 compare_multi.py 9850x3d.json 9700x.json 285k.json 265k.json
```

Outputs `multi_comparison.png`. Shows all CPUs side by side with % difference labels.

### compare_pitch.py — Pitch deck version (clean, minimal)

```bash
python3 compare_pitch.py 9850x3d.json 9700x.json 285k.json 265k.json --output pitch.png
```

Outputs `pitch_comparison.png`. 3 charts (QPS, Latency, RAG) with only the most important values. Designed for slides and presentations.

---

## Sharing Embeddings Between Machines

For a fair comparison, use **identical embedding vectors** on all CPUs.

```bash
# Generate on first machine
python3 benchmark.py --output cpu_a.json  # name after your CPU

# Copy cache to second machine
scp -r ./embedding_cache/ user@other-machine:~/x3d-rag-benchmark/

# Run on second machine (loads from cache instantly)
python3 benchmark.py --output cpu_b.json  # name after your CPU
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

---

## Reproducibility Conditions

- Use Linux native for lowest variance (Ubuntu 24.04 recommended)
- Minimize background processes
- Reboot before benchmarking (recommended)
- Use identical RAM capacity and speed
- Use identical GPU
- Use `--runs 10` or higher for publication-quality results
- Share `embedding_cache/` between machines for fair comparison
- Check AVX instruction set with `faiss.get_compile_options()`

### BIOS Settings (recommended)

For best reproducibility, set identically on all systems:

| Setting | Recommendation |
|---|---|
| PBO / Turbo Boost | Same on all (all ON or all OFF) |
| XMP / EXPO | Same memory profile (e.g. DDR5-6000 CL30) |
| C-States | Disabled |
| Cool & Quiet | Disabled |
| AVX-512 | Note the setting — affects FAISS instruction path |

---

## Contributing

PRs and issues welcome.
Submit your results (JSON files) to the `results/` folder via PR.
Include: CPU model, motherboard, BIOS version, memory config, OS, AVX setting.
