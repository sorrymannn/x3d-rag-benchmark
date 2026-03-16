"""
x3d-rag-benchmark
=================
AMD Ryzen X3D V-Cache vs non-X3D CPU performance benchmark
for RAG AI pipelines

Measures:
  1. Vector Search Latency  - FAISS HNSW search latency
  2. RAG TTFT               - Time to first token (full pipeline)
  3. Concurrent Search      - Multi-query throughput

Libraries:
  - FAISS              (Meta AI)    : vector search engine
  - sentence-transformers (HuggingFace) : embedding model
  - ollama                          : local LLM server
  - datasets           (HuggingFace) : public dataset

Install:
  pip install faiss-cpu sentence-transformers ollama datasets numpy tqdm

Run:
  python3 benchmark.py
  python3 benchmark.py --runs 3           # average over 3 runs
  python3 benchmark.py --skip-rag         # vector search only
  python3 benchmark.py --quick --skip-rag # fast test (~3 min)
"""

import argparse
import json
import os
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

# ── Dependency check ──────────────────────────────────────────────────────────

def check_deps():
    missing = []
    for pkg in ["faiss", "sentence_transformers", "ollama", "datasets", "tqdm"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg.replace("_", "-"))
    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print(f"Install: pip install {' '.join(missing)}")
        exit(1)

check_deps()

import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import ollama as ollama_client

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT = {
    "embed_model":  "all-MiniLM-L6-v2",
    "llm_model":    "llama3.2",
    "db_sizes":     [100_000, 500_000, 1_000_000],
    "n_queries":    200,
    "top_k":        10,
    "hnsw_m":       32,
    "hnsw_ef":      64,
    "concurrent":   [1, 2, 4, 8],
    "warmup":       20,
    "rag_queries":  50,
    "runs":         1,          # number of runs to average
}

# ── Utils ─────────────────────────────────────────────────────────────────────

def get_cpu_info():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":")[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"

def get_gpu_info():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"], text=True
        ).strip()
    except Exception:
        return "N/A"

def pct(data, p):
    s = sorted(data)
    return s[max(0, int(len(s) * p / 100) - 1)]

def ms(t): return round(t * 1000, 3)

def mean(lst): return float(np.mean(lst))
def std(lst):  return float(np.std(lst))

# ── Vector Search (single run) ────────────────────────────────────────────────

def _vector_search_once(db_size, cfg, idx, q_vecs):
    """Run one pass of vector search benchmark, return latency list."""
    bench_q = q_vecs[cfg["warmup"]:]
    latencies = []
    t_total = time.perf_counter()
    for q in bench_q:
        t0 = time.perf_counter()
        idx.search(q.reshape(1, -1), cfg["top_k"])
        latencies.append(time.perf_counter() - t0)
    total_sec = time.perf_counter() - t_total
    qps = len(latencies) / total_sec
    return qps, latencies


def run_vector_search(db_size, cfg):
    """FAISS HNSW benchmark — repeated cfg['runs'] times, results averaged."""
    dim = 384
    n_runs = cfg["runs"]

    print(f"\n  DB size: {db_size:,} vectors  |  runs: {n_runs}")
    print(f"  Est. memory: ~{db_size * dim * 4 / 1024**3:.2f} GB")

    print("  Building vectors...", end=" ", flush=True)
    db_vecs = np.random.rand(db_size, dim).astype(np.float32)
    faiss.normalize_L2(db_vecs)
    print("done")

    print("  Building HNSW index...", end=" ", flush=True)
    t0 = time.perf_counter()
    idx = faiss.IndexHNSWFlat(dim, cfg["hnsw_m"])
    idx.hnsw.efConstruction = 200
    idx.add(db_vecs)
    build_sec = time.perf_counter() - t0
    idx.hnsw.efSearch = cfg["hnsw_ef"]
    print(f"{build_sec:.1f}s")

    # collect per-run QPS
    all_qps  = []
    all_p50  = []
    all_p95  = []
    all_p99  = []
    all_mean = []

    for run_i in range(n_runs):
        # fresh query vectors each run to avoid cache warming bias
        q_vecs = np.random.rand(cfg["n_queries"] + cfg["warmup"], dim).astype(np.float32)
        faiss.normalize_L2(q_vecs)

        # warmup
        for q in q_vecs[:cfg["warmup"]]:
            idx.search(q.reshape(1, -1), cfg["top_k"])

        qps, lats = _vector_search_once(db_size, cfg, idx, q_vecs)
        all_qps.append(qps)
        all_p50.append(ms(pct(lats, 50)))
        all_p95.append(ms(pct(lats, 95)))
        all_p99.append(ms(pct(lats, 99)))
        all_mean.append(ms(np.mean(lats)))

        if n_runs > 1:
            print(f"    run {run_i+1}/{n_runs}  QPS: {qps:.1f}  "
                  f"P50: {ms(pct(lats,50)):.3f}ms  P99: {ms(pct(lats,99)):.3f}ms")

    result = {
        "db_size":         db_size,
        "build_time_s":    round(build_sec, 2),
        "runs":            n_runs,
        # averaged metrics
        "qps":             round(mean(all_qps), 1),
        "qps_stddev":      round(std(all_qps), 1),
        "qps_runs":        [round(v, 1) for v in all_qps],
        "latency_mean_ms": round(mean(all_mean), 3),
        "latency_p50_ms":  round(mean(all_p50), 3),
        "latency_p50_stddev": round(std(all_p50), 3),
        "latency_p95_ms":  round(mean(all_p95), 3),
        "latency_p99_ms":  round(mean(all_p99), 3),
        "latency_p99_stddev": round(std(all_p99), 3),
    }

    print(f"  avg QPS: {result['qps']:.1f} ±{result['qps_stddev']:.1f}  |  "
          f"P50: {result['latency_p50_ms']:.3f}ms  |  "
          f"P99: {result['latency_p99_ms']:.3f}ms ±{result['latency_p99_stddev']:.3f}")

    del db_vecs, idx
    return result


# ── Concurrent Search ─────────────────────────────────────────────────────────

def run_concurrent_search(db_size, cfg):
    """Concurrent query benchmark — repeated cfg['runs'] times."""
    dim = 384
    n_runs = cfg["runs"]

    db_vecs = np.random.rand(db_size, dim).astype(np.float32)
    faiss.normalize_L2(db_vecs)
    idx = faiss.IndexHNSWFlat(dim, cfg["hnsw_m"])
    idx.hnsw.efConstruction = 200
    idx.add(db_vecs)
    idx.hnsw.efSearch = cfg["hnsw_ef"]

    results = {}
    for n_concurrent in cfg["concurrent"]:
        run_qps  = []
        run_p99  = []

        for _ in range(n_runs):
            q_batch = np.random.rand(n_concurrent * 20, dim).astype(np.float32)
            faiss.normalize_L2(q_batch)

            def single_search(q):
                t0 = time.perf_counter()
                idx.search(q.reshape(1, -1), cfg["top_k"])
                return time.perf_counter() - t0

            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=n_concurrent) as ex:
                lats = list(ex.map(single_search, q_batch))
            total = time.perf_counter() - t0

            run_qps.append(len(lats) / total)
            run_p99.append(ms(pct(lats, 99)))

        results[n_concurrent] = {
            "qps":            round(mean(run_qps), 1),
            "qps_stddev":     round(std(run_qps), 1),
            "latency_p50_ms": round(mean([ms(pct([], 50)) if not run_qps else 0]), 3),
            "latency_p99_ms": round(mean(run_p99), 3),
            "latency_p99_stddev": round(std(run_p99), 3),
        }
        print(f"  {n_concurrent:>2} concurrent  "
              f"QPS: {results[n_concurrent]['qps']:>8.1f} ±{results[n_concurrent]['qps_stddev']:.1f}  "
              f"P99: {results[n_concurrent]['latency_p99_ms']:.3f}ms "
              f"±{results[n_concurrent]['latency_p99_stddev']:.3f}")

    del db_vecs, idx
    return results


# ── RAG TTFT ──────────────────────────────────────────────────────────────────

def check_ollama_model(model):
    try:
        models = ollama_client.list()
        return any(model in m.model for m in models.models)
    except Exception:
        return False


def run_rag_ttft(cfg, embed_model):
    """Full RAG pipeline TTFT: embed(GPU) → search(CPU) → LLM(GPU)"""
    if not check_ollama_model(cfg["llm_model"]):
        print(f"  [WARNING] ollama model '{cfg['llm_model']}' not found.")
        print(f"  Install: ollama pull {cfg['llm_model']}")
        return None

    print("  Loading dataset...", end=" ", flush=True)
    try:
        ds = load_dataset("wikipedia", "20220301.simple",
                          split="train[:2000]", trust_remote_code=True)
        passages = [r["text"][:300] for r in ds]
    except Exception:
        passages = [f"This is passage number {i} about topic {i % 50}."
                    for i in range(2000)]
    print(f"{len(passages)} passages")

    print("  Generating embeddings...", end=" ", flush=True)
    t0 = time.perf_counter()
    embeddings = embed_model.encode(passages, batch_size=64,
                                    show_progress_bar=False,
                                    convert_to_numpy=True)
    embed_sec = time.perf_counter() - t0
    print(f"{embed_sec:.1f}s")

    dim = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    idx = faiss.IndexHNSWFlat(dim, 32)
    idx.hnsw.efConstruction = 200
    idx.add(embeddings)
    idx.hnsw.efSearch = 64

    queries = ([
        "What is machine learning?",
        "How does the human brain work?",
        "What is the history of the internet?",
        "Explain quantum computing",
        "What causes climate change?",
    ] * (cfg["rag_queries"] // 5 + 1))[:cfg["rag_queries"]]

    # warmup
    for q in queries[:5]:
        q_emb = embed_model.encode([q], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        idx.search(q_emb, cfg["top_k"])

    ttft_list        = []
    search_lat_list  = []

    print(f"  Measuring RAG TTFT ({len(queries)} queries, {cfg['runs']} run(s))...")
    for _ in range(cfg["runs"]):
        for query in tqdm(queries, ncols=60, leave=False):
            q_emb = embed_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(q_emb)

            t_search = time.perf_counter()
            D, I = idx.search(q_emb, cfg["top_k"])
            search_lat_list.append(time.perf_counter() - t_search)

            context = " ".join([passages[i] for i in I[0] if i < len(passages)])[:500]
            prompt  = f"Context: {context}\n\nQuestion: {query}\nAnswer briefly:"

            t_start = time.perf_counter()
            try:
                stream = ollama_client.chat(
                    model=cfg["llm_model"],
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                )
                first = False
                for chunk in stream:
                    if not first:
                        ttft_list.append(time.perf_counter() - t_start)
                        first = True
                        break
            except Exception as e:
                print(f"\n  [LLM ERROR] {e}")
                return None

    if not ttft_list:
        return None

    result = {
        "n_passages":   len(passages),
        "embed_time_s": round(embed_sec, 2),
        "runs":         cfg["runs"],
        "vector_search": {
            "mean_ms":   ms(np.mean(search_lat_list)),
            "p50_ms":    ms(pct(search_lat_list, 50)),
            "p95_ms":    ms(pct(search_lat_list, 95)),
            "p99_ms":    ms(pct(search_lat_list, 99)),
            "stddev_ms": ms(np.std(search_lat_list)),
        },
        "ttft": {
            "mean_ms":   ms(np.mean(ttft_list)),
            "p50_ms":    ms(pct(ttft_list, 50)),
            "p95_ms":    ms(pct(ttft_list, 95)),
            "p99_ms":    ms(pct(ttft_list, 99)),
            "stddev_ms": ms(np.std(ttft_list)),
        },
    }

    print(f"  [Vector Search]  P50: {result['vector_search']['p50_ms']:.3f}ms  "
          f"P99: {result['vector_search']['p99_ms']:.3f}ms  "
          f"stddev: {result['vector_search']['stddev_ms']:.3f}ms")
    print(f"  [RAG TTFT]       P50: {result['ttft']['p50_ms']:.1f}ms  "
          f"P99: {result['ttft']['p99_ms']:.1f}ms  "
          f"stddev: {result['ttft']['stddev_ms']:.1f}ms")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="x3d-rag-benchmark")
    parser.add_argument("--model",    default=DEFAULT["llm_model"],
                        help=f"ollama LLM model (default: {DEFAULT['llm_model']})")
    parser.add_argument("--db-size",  type=int, default=None,
                        help="single DB size override")
    parser.add_argument("--queries",  type=int, default=DEFAULT["n_queries"],
                        help=f"queries per run (default: {DEFAULT['n_queries']})")
    parser.add_argument("--runs",     type=int, default=DEFAULT["runs"],
                        help="number of runs to average (default: 1, recommended: 3)")
    parser.add_argument("--skip-rag", action="store_true",
                        help="skip RAG TTFT (vector search only)")
    parser.add_argument("--quick",    action="store_true",
                        help="quick test — small DB only (~3 min)")
    parser.add_argument("--output",   default=None,
                        help="output JSON path (default: auto-generated)")
    args = parser.parse_args()

    cfg = DEFAULT.copy()
    cfg["llm_model"] = args.model
    cfg["n_queries"] = args.queries
    cfg["runs"]      = max(1, args.runs)

    if args.db_size:
        cfg["db_sizes"] = [args.db_size]
    elif args.quick:
        cfg["db_sizes"] = [100_000]

    cpu = get_cpu_info()
    gpu = get_gpu_info()
    now = datetime.now()

    print("\n" + "="*60)
    print("  x3d-rag-benchmark")
    print("  github.com/sorrymannn/x3d-rag-benchmark")
    print("="*60)
    print(f"  CPU:  {cpu}")
    print(f"  GPU:  {gpu}")
    print(f"  Runs: {cfg['runs']}  (results averaged)")
    print(f"  Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    output = {
        "meta": {
            "cpu":       cpu,
            "gpu":       gpu,
            "timestamp": now.isoformat(),
            "config":    cfg,
        },
        "vector_search":     [],
        "concurrent_search": {},
        "rag_ttft":          None,
    }

    # 1. Vector Search
    print("\n[1/3] Vector Search Benchmark (FAISS HNSW)")
    print("      Core CPU task in RAG pipelines")
    print("-"*60)
    for db_size in cfg["db_sizes"]:
        output["vector_search"].append(run_vector_search(db_size, cfg))

    # 2. Concurrent Search
    print("\n[2/3] Concurrent Search Benchmark")
    print("      Multi-query throughput — X3D cache effect under load")
    print("-"*60)
    mid = cfg["db_sizes"][len(cfg["db_sizes"]) // 2]
    print(f"  DB size: {mid:,} vectors")
    output["concurrent_search"] = run_concurrent_search(mid, cfg)

    # 3. RAG TTFT
    if not args.skip_rag:
        print("\n[3/3] RAG End-to-End TTFT Benchmark")
        print("      embed(GPU) → search(CPU) → LLM(GPU)")
        print("-"*60)
        print("  Loading embedding model...", end=" ", flush=True)
        embed_model = SentenceTransformer(cfg["embed_model"])
        print("done")
        output["rag_ttft"] = run_rag_ttft(cfg, embed_model)
    else:
        print("\n[3/3] RAG TTFT — skipped (--skip-rag)")

    # Save
    out_path = args.output or f"result_{now.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print(f"  Done! Results saved: {out_path}")
    print(f"  Compare: python3 compare.py result_a.json result_b.json")
    print("="*60)


if __name__ == "__main__":
    main()
