# x3d-rag-benchmark

**AMD Ryzen X3D V-Cache vs non-X3D — RAG AI Pipeline CPU Performance Benchmark**

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![FAISS](https://img.shields.io/badge/Meta-FAISS-blue.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-sentence--transformers-orange.svg)

An open-source benchmark measuring the impact of CPU (X3D vs non-X3D) on RAG AI pipelines
in real AI PC environments where GPU is also in use.

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
  Vector Search          (CPU) ← X3D V-Cache effect
      ↓
  LLM generation         (GPU)
      ↓
  Response

Vector Search characteristics:
  - Randomly traverses HNSW graph nodes
  - Each traversal accesses a different memory address (Random Access)
  - Larger L3 cache → higher cache hit rate → lower latency
  - 96MB V-Cache = same mechanism as X3D dominance in gaming
```

---

## What We Measure

| Metric | Description | X3D Impact |
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
| **Single-threaded FAISS** | Eliminates OS scheduler multi-core dispatch noise |
| **Trimmed mean** (drop top/bottom 5%) | Removes thermal throttle and outlier spikes |
| **5 runs by default** | Statistical reliability |
| **Embedding cache** | Same vectors reused across runs and machines |

> **Same CPU should produce CV < 3%** (coefficient of variation).
> If you see higher variance, increase `--runs`.

---

## Installation

### 1. Clone
```bash
git clone https://github.com/sorrymannn/x3d-rag-benchmark
cd x3d-rag-benchmark
```

### 2. Python packages
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. ollama + LLM model (for RAG TTFT only)
```bash
# Install ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull LLM model
ollama pull llama3.2
```

> RAG TTFT measurement is optional. Vector Search benchmark runs without ollama.

---

## Usage

```bash
# Full benchmark (Vector Search + RAG TTFT)
python3 benchmark.py

# Recommended: 5 runs averaged (default)
python3 benchmark.py --output 9700x.json
python3 benchmark.py --output 9800x3d.json

# Vector Search only (no ollama needed, ~30-45 min)
python3 benchmark.py --skip-rag

# Quick test (~5 min, includes embedding generation on first run)
python3 benchmark.py --quick --skip-rag
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

---

## Compare Results

Generate comparison charts from two JSON result files.

```bash
python3 compare.py 9700x.json 9800x3d.json
# → outputs comparison.png
```

### Chart contents
- **Vector Search QPS** — bar chart with error bars by DB size
- **Vector Search P99 Latency** — line chart with error bars
- **Latency Distribution** — P50 / P95 / P99 bar chart (largest DB)
- **RAG TTFT** — vector search latency inside full RAG pipeline

---

## Sharing Embeddings Between Machines

For a fair comparison, use **identical embedding vectors** on both CPUs.

```bash
# Generate on first machine
python3 benchmark.py --output 9700x.json

# Copy cache to second machine
scp -r ./embedding_cache/ user@9800x3d-machine:~/x3d-rag-benchmark/

# Run on second machine (loads from cache instantly)
python3 benchmark.py --output 9800x3d.json
```

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

- Minimize background processes
- Reboot before benchmarking (recommended)
- Use identical RAM capacity and speed
- Use identical GPU
- Use identical OS environment
- Use `--runs 5` or higher for publication-quality results
- Share `embedding_cache/` between machines for fair comparison

---

## Contributing

PRs and issues welcome.
Submit your results (JSON files) to the `results/` folder via PR.
