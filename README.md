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
| Concurrent Search | Multi-query throughput | **Direct** |
| RAG TTFT | Time to first token (full pipeline) | Indirect |

Results include **stddev and error bars** when using `--runs` for statistical reliability.

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

# Recommended: 3 runs averaged for reliable results
python3 benchmark.py --runs 3 --output 9700x.json
python3 benchmark.py --runs 3 --output 9800x3d.json

# Vector Search only (no ollama needed, ~30-45 min)
python3 benchmark.py --skip-rag

# Quick test (~3 min)
python3 benchmark.py --quick --skip-rag
```

### Options

| Option | Default | Description |
|---|---|---|
| `--runs N` | 1 | Repeat N times and average results (recommended: 3) |
| `--skip-rag` | off | Skip RAG TTFT, run Vector Search only |
| `--quick` | off | Test with small DB only (~3 min) |
| `--output FILE` | auto | Save results to specified JSON file |
| `--model NAME` | llama3.2 | ollama model for RAG TTFT |
| `--queries N` | 200 | Number of queries per run |
| `--cache-dir PATH` | ./embedding_cache | Embedding cache directory |
| `--rebuild` | off | Force rebuild embedding cache |
| `--db-size N` | - | Override DB size (single value) |

---

## Compare Results

Generate comparison charts from two JSON result files.

```bash
python3 compare.py 9700x.json 9800x3d.json
# → outputs comparison.png
```

Charts include **error bars** when multiple runs were used.

---

## Libraries Used

| Library | Source | Purpose |
|---|---|---|
| FAISS | Meta AI | Vector search engine |
| sentence-transformers | HuggingFace | Embedding model |
| ollama | Ollama | Local LLM server |
| datasets | HuggingFace | Public dataset |

---

## Reproducibility

For consistent results across machines:
- Minimize background processes
- Reboot before benchmarking (recommended)
- Use identical RAM capacity and speed
- Use identical GPU
- Use identical OS environment
- Use `--runs 3` or higher for statistical reliability

---

## Contributing

PRs and issues welcome.
Submit your results (JSON files) to the `results/` folder via PR.
