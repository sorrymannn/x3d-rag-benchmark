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
| **Configurable FAISS threads** | Default: all cores (auto). Use `--threads 1` if CV > 3% |
| **Trimmed mean** (drop top/bottom 5%) | Removes thermal throttle and outlier spikes |
| **5 runs by default** | Statistical reliability |
| **Embedding cache** | Same vectors reused across runs and machines |
| **OS-level variance controls** | CPU governor, NUMA, THP, process priority (auto-applied) |
| **Inter-run cooling** | 2s delay between runs prevents thermal drift |
| **Python GC disabled** | No garbage collection during measurement |

> **Same CPU should produce CV < 3%** (coefficient of variation).
> If CV > 3%, try `--threads 1` (single-threaded) or increase `--runs`.

### Automatic Variance Controls

The benchmark automatically applies these at startup (skips if no permission):

| Control | Linux | Windows | macOS |
|---|---|---|---|
| CPU frequency lock | `scaling_governor → performance` | High Performance power plan | N/A |
| NUMA balancing | `kernel.numa_balancing=0` | N/A | N/A |
| THP | `transparent_hugepage → never` | N/A | N/A |
| Process priority | `nice -20` | `HIGH_PRIORITY_CLASS` | `nice -20` |
| Python GC | Disabled during benchmark | same | same |
| Inter-run cooling | 2s between runs | same | same |

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

> **Windows 참고**: `faiss-cpu`는 pip로 바로 설치됩니다. Visual Studio 빌드 불필요.

### 3. ollama + LLM model (RAG TTFT 측정 시에만 필요)

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
```

**Windows:**
```powershell
# https://ollama.com/download 에서 설치 후
ollama pull llama3.2
```

> RAG TTFT는 선택사항입니다. Vector Search 벤치마크는 ollama 없이 동작합니다.

---

## Usage

**Linux / macOS:**
```bash
# Vector Search only (ollama 불필요, ~30-45분)
python3 benchmark.py --skip-rag --output 9800x3d.json

# Full benchmark (Vector Search + RAG TTFT)
python3 benchmark.py --output 9800x3d.json

# Quick test (~5분)
python3 benchmark.py --quick --skip-rag
```

**Windows:**
```powershell
# Vector Search only
python benchmark.py --skip-rag --output 9800x3d.json

# Full benchmark
python benchmark.py --output 9800x3d.json

# Quick test
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

Generate comparison charts from two JSON result files.

**Linux / macOS:**
```bash
python3 compare.py 9700x.json 9800x3d.json
```

**Windows:**
```powershell
python compare.py 9700x.json 9800x3d.json
```

→ `comparison.png` 생성

### Chart contents
- **Vector Search QPS** — bar chart with error bars by DB size
- **Vector Search P99 Latency** — line chart with error bars
- **Latency Distribution** — P50 / P95 / P99 bar chart (largest DB)
- **RAG TTFT** — vector search latency inside full RAG pipeline

---

## Sharing Embeddings Between Machines

For a fair comparison, use **identical embedding vectors** on both CPUs.

**Linux → Linux:**
```bash
# Generate on first machine
python3 benchmark.py --output 9700x.json

# Copy cache to second machine
scp -r ./embedding_cache/ user@other-machine:~/x3d-rag-benchmark/

# Run on second machine
python3 benchmark.py --output 9800x3d.json
```

**Windows → Windows (or cross-platform):**
```powershell
# Generate on first machine
python benchmark.py --output 9700x.json

# Copy embedding_cache folder to second machine via USB, network share, etc.
# Then run on second machine
python benchmark.py --output 9800x3d.json
```

> 임베딩 캐시는 OS에 관계없이 호환됩니다 (numpy `.npy` 파일).

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
- Use `--runs 5` or higher for publication-quality results
- Share `embedding_cache/` between machines for fair comparison

### BIOS Settings (recommended)

For best reproducibility, set identically on both systems:

| Setting | Recommendation |
|---|---|
| PBO / Turbo Boost | Same on both (both ON or both OFF) |
| XMP / EXPO | Same memory profile (e.g. DDR5-6000 CL30) |
| C-States | Disabled |
| Cool & Quiet | Disabled |

---

## Contributing

PRs and issues welcome.
Submit your results (JSON files) to the `results/` folder via PR.
Include: CPU model, motherboard, BIOS version, memory config, OS.
