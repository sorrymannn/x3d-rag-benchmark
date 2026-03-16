"""
x3d-rag-benchmark
=================
AMD Ryzen X3D V-Cache vs non-X3D CPU performance benchmark
for RAG AI pipelines

Measures:
  1. Vector Search Latency  - FAISS HNSW search latency (real embeddings)
  2. RAG TTFT               - Time to first token (full pipeline)
  3. Concurrent Search      - Multi-query throughput

Key design:
  - Uses REAL text embeddings (Wikipedia) instead of random vectors
  - Real embeddings cluster by topic → stable, reproducible results
  - Embeddings cached to disk → generated once, reused across runs

Libraries:
  - FAISS              (Meta AI)       : vector search engine
  - sentence-transformers (HuggingFace): embedding model
  - ollama                             : local LLM server
  - datasets           (HuggingFace)  : public dataset

Install:
  pip install faiss-cpu sentence-transformers ollama datasets numpy tqdm

Run:
  python3 benchmark.py
  python3 benchmark.py --runs 3            # average over 3 runs (recommended)
  python3 benchmark.py --skip-rag          # vector search only
  python3 benchmark.py --quick --skip-rag  # fast test (~5 min)
"""

import argparse
import json
import os
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

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
    "embed_model":   "all-MiniLM-L6-v2",
    "llm_model":     "llama3.2",
    "db_sizes":      [100_000, 500_000, 1_000_000],
    "n_queries":     200,
    "top_k":         10,
    "hnsw_m":        32,
    "hnsw_ef":       64,
    "concurrent":    [1, 2, 4, 8],
    "warmup":        20,
    "rag_queries":   50,
    "runs":          1,
    "cache_dir":     "./embedding_cache",   # embedding cache directory
    "n_passages":    1_100_000,             # total passages to embed (>= max db_size + queries)
}

CACHE_DB_FILE    = "wiki_db_embeddings.npy"
CACHE_Q_FILE     = "wiki_query_embeddings.npy"
CACHE_TEXT_FILE  = "wiki_passages.json"

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

# ── Embedding Cache ───────────────────────────────────────────────────────────

def load_or_build_embeddings(cfg, embed_model):
    """
    Load real Wikipedia embeddings from cache,
    or build them from scratch and save for reuse.

    Why real embeddings?
    - Random vectors are uniformly distributed → HNSW traversal paths vary wildly
    - Real text embeddings cluster by topic   → stable, reproducible search paths
    - Reflects actual RAG workload accurately
    """
    cache_dir  = Path(cfg["cache_dir"])
    cache_dir.mkdir(exist_ok=True)

    db_path   = cache_dir / CACHE_DB_FILE
    q_path    = cache_dir / CACHE_Q_FILE
    txt_path  = cache_dir / CACHE_TEXT_FILE

    if db_path.exists() and q_path.exists() and txt_path.exists():
        print("  Loading embeddings from cache...", end=" ", flush=True)
        db_embeddings = np.load(str(db_path))
        q_embeddings  = np.load(str(q_path))
        with open(txt_path) as f:
            passages = json.load(f)
        print(f"done  ({len(db_embeddings):,} DB + {len(q_embeddings):,} query vectors)")
        return db_embeddings, q_embeddings, passages

    # ── Build from scratch ──
    print("  [First run] Building real embeddings from Wikipedia...")
    print("  This takes ~10-20 min (GPU) or ~60 min (CPU).")
    print("  Results will be cached → subsequent runs load instantly.\n")

    n_total   = cfg["n_passages"]
    n_queries = cfg["n_queries"] + cfg["warmup"] + 500   # extra buffer

    # Load Wikipedia (Simple English — smaller, faster)
    print("  Downloading Wikipedia dataset...", end=" ", flush=True)
    try:
        ds = load_dataset(
            "wikipedia", "20220301.simple",
            split="train",
            streaming=False,
        )
        all_texts = [row["text"][:512] for row in ds]
    except Exception as e:
        print(f"\n  [WARNING] Wikipedia load failed ({e}), using fallback dataset...")
        ds = load_dataset("ag_news", split="train")
        all_texts = [row["text"] for row in ds]

    print(f"{len(all_texts):,} articles loaded")

    # Repeat if not enough passages
    while len(all_texts) < n_total + n_queries:
        all_texts = all_texts * 2
    all_texts = all_texts[:n_total + n_queries]

    db_texts    = all_texts[:n_total]
    query_texts = all_texts[n_total:n_total + n_queries]

    # Embed DB passages
    print(f"  Embedding {len(db_texts):,} DB passages...")
    db_embeddings = embed_model.encode(
        db_texts,
        batch_size=512,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Embed query passages
    print(f"  Embedding {len(query_texts):,} query vectors...")
    q_embeddings = embed_model.encode(
        query_texts,
        batch_size=512,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Save cache
    print("  Saving to cache...", end=" ", flush=True)
    np.save(str(db_path), db_embeddings)
    np.save(str(q_path),  q_embeddings)
    with open(txt_path, "w") as f:
        json.dump(db_texts, f)
    print("done")

    return db_embeddings, q_embeddings, db_texts


# ── Vector Search ─────────────────────────────────────────────────────────────

def build_hnsw_index(db_embeddings, db_size, cfg):
    """Build HNSW index from real embeddings, subsampled to db_size."""
    dim = db_embeddings.shape[1]
    vecs = db_embeddings[:db_size].copy()
    faiss.normalize_L2(vecs)

    idx = faiss.IndexHNSWFlat(dim, cfg["hnsw_m"])
    idx.hnsw.efConstruction = 200
    idx.add(vecs)
    idx.hnsw.efSearch = cfg["hnsw_ef"]
    return idx


def _search_once(idx, q_vecs, cfg):
    bench_q = q_vecs[cfg["warmup"]:]
    latencies = []
    t_total = time.perf_counter()
    for q in bench_q:
        t0 = time.perf_counter()
        idx.search(q.reshape(1, -1), cfg["top_k"])
        latencies.append(time.perf_counter() - t0)
    total_sec = time.perf_counter() - t_total
    return len(latencies) / total_sec, latencies


def run_vector_search(db_size, cfg, db_embeddings, q_embeddings):
    """FAISS HNSW benchmark using real embeddings — repeated cfg['runs'] times."""
    dim    = db_embeddings.shape[1]
    n_runs = cfg["runs"]
    n_q    = cfg["n_queries"] + cfg["warmup"]

    print(f"\n  DB size: {db_size:,} vectors  |  dim: {dim}  |  runs: {n_runs}")
    print(f"  Est. memory: ~{db_size * dim * 4 / 1024**3:.2f} GB")

    print("  Building HNSW index...", end=" ", flush=True)
    t0  = time.perf_counter()
    idx = build_hnsw_index(db_embeddings, db_size, cfg)
    build_sec = time.perf_counter() - t0
    print(f"{build_sec:.1f}s")

    # Use real query embeddings — shuffled differently each run
    all_q = q_embeddings[:n_q * n_runs].copy()
    faiss.normalize_L2(all_q)

    all_qps  = []
    all_p50  = []
    all_p95  = []
    all_p99  = []
    all_mean = []

    for run_i in range(n_runs):
        # slice a different window of real query embeddings per run
        start = run_i * cfg["n_queries"]
        q_slice = np.vstack([
            all_q[start: start + cfg["warmup"]],           # warmup
            all_q[start: start + cfg["n_queries"]],        # bench
        ])

        # warmup
        for q in q_slice[:cfg["warmup"]]:
            idx.search(q.reshape(1, -1), cfg["top_k"])

        qps, lats = _search_once(idx, q_slice, cfg)
        all_qps.append(qps)
        all_p50.append(ms(pct(lats, 50)))
        all_p95.append(ms(pct(lats, 95)))
        all_p99.append(ms(pct(lats, 99)))
        all_mean.append(ms(np.mean(lats)))

        if n_runs > 1:
            print(f"    run {run_i+1}/{n_runs}  "
                  f"QPS: {qps:.1f}  "
                  f"P50: {ms(pct(lats,50)):.3f}ms  "
                  f"P99: {ms(pct(lats,99)):.3f}ms")

    result = {
        "db_size":              db_size,
        "build_time_s":         round(build_sec, 2),
        "runs":                 n_runs,
        "embedding_type":       "real (Wikipedia)",
        "qps":                  round(mean(all_qps), 1),
        "qps_stddev":           round(std(all_qps), 1),
        "qps_runs":             [round(v, 1) for v in all_qps],
        "latency_mean_ms":      round(mean(all_mean), 3),
        "latency_p50_ms":       round(mean(all_p50), 3),
        "latency_p50_stddev":   round(std(all_p50), 3),
        "latency_p95_ms":       round(mean(all_p95), 3),
        "latency_p99_ms":       round(mean(all_p99), 3),
        "latency_p99_stddev":   round(std(all_p99), 3),
    }

    print(f"  avg QPS: {result['qps']:.1f} ±{result['qps_stddev']:.1f}  |  "
          f"P50: {result['latency_p50_ms']:.3f}ms  |  "
          f"P99: {result['latency_p99_ms']:.3f}ms ±{result['latency_p99_stddev']:.3f}")

    del idx
    return result


# ── Concurrent Search ─────────────────────────────────────────────────────────

def run_concurrent_search(db_size, cfg, db_embeddings, q_embeddings):
    """Concurrent query benchmark with real embeddings."""
    n_runs = cfg["runs"]

    print(f"  DB size: {db_size:,} vectors")
    idx = build_hnsw_index(db_embeddings, db_size, cfg)

    results = {}
    for n_concurrent in cfg["concurrent"]:
        n_total_q = n_concurrent * 20
        q_batch = q_embeddings[:n_total_q].copy()
        faiss.normalize_L2(q_batch)

        run_qps = []
        run_p99 = []

        for _ in range(n_runs):
            # shuffle query order each run (realistic)
            np.random.shuffle(q_batch)

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
            "qps":                  round(mean(run_qps), 1),
            "qps_stddev":           round(std(run_qps), 1),
            "latency_p99_ms":       round(mean(run_p99), 3),
            "latency_p99_stddev":   round(std(run_p99), 3),
        }
        print(f"  {n_concurrent:>2} concurrent  "
              f"QPS: {results[n_concurrent]['qps']:>8.1f} "
              f"±{results[n_concurrent]['qps_stddev']:.1f}  "
              f"P99: {results[n_concurrent]['latency_p99_ms']:.3f}ms "
              f"±{results[n_concurrent]['latency_p99_stddev']:.3f}")

    del idx
    return results


# ── RAG TTFT ──────────────────────────────────────────────────────────────────

def check_ollama_model(model):
    try:
        models = ollama_client.list()
        return any(model in m.model for m in models.models)
    except Exception:
        return False


def run_rag_ttft(cfg, embed_model, db_embeddings, q_embeddings, passages):
    """Full RAG pipeline TTFT using real embeddings."""
    if not check_ollama_model(cfg["llm_model"]):
        print(f"  [WARNING] ollama model '{cfg['llm_model']}' not found.")
        print(f"  Install: ollama pull {cfg['llm_model']}")
        return None

    rag_db_size = min(10_000, len(db_embeddings))
    vecs = db_embeddings[:rag_db_size].copy()
    faiss.normalize_L2(vecs)

    dim = vecs.shape[1]
    idx = faiss.IndexHNSWFlat(dim, 32)
    idx.hnsw.efConstruction = 200
    idx.add(vecs)
    idx.hnsw.efSearch = 64

    n_q = cfg["rag_queries"]
    query_vecs = q_embeddings[:n_q + 10].copy()
    faiss.normalize_L2(query_vecs)

    # Warmup — include ollama warmup to exclude model load time
    print("  Warming up ollama...", end=" ", flush=True)
    for i in range(10):
        qv = query_vecs[i].reshape(1, -1)
        _, I = idx.search(qv, cfg["top_k"])
        ctx = " ".join([passages[j] for j in I[0] if j < len(passages)])[:300]
        try:
            stream = ollama_client.chat(
                model=cfg["llm_model"],
                messages=[{"role": "user",
                           "content": f"Context: {ctx}\nAnswer in one word:"}],
                stream=True,
            )
            for chunk in stream:
                break
        except Exception:
            pass
    print("done")

    ttft_list       = []
    search_lat_list = []

    print(f"  Measuring RAG TTFT ({n_q} queries, {cfg['runs']} run(s))...")
    for run_i in range(cfg["runs"]):
        run_q = query_vecs[run_i % len(query_vecs): run_i % len(query_vecs) + n_q]
        if len(run_q) < n_q:
            run_q = query_vecs[:n_q]

        for qv in tqdm(run_q, ncols=60, leave=False,
                       desc=f"  run {run_i+1}/{cfg['runs']}"):
            t_search = time.perf_counter()
            _, I = idx.search(qv.reshape(1, -1), cfg["top_k"])
            search_lat_list.append(time.perf_counter() - t_search)

            ctx    = " ".join([passages[j] for j in I[0] if j < len(passages)])[:500]
            prompt = f"Context: {ctx}\n\nQuestion: What is this about?\nAnswer briefly:"

            t_start = time.perf_counter()
            try:
                stream = ollama_client.chat(
                    model=cfg["llm_model"],
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                )
                for chunk in stream:
                    ttft_list.append(time.perf_counter() - t_start)
                    break
            except Exception as e:
                print(f"\n  [LLM ERROR] {e}")
                return None

    if not ttft_list:
        return None

    result = {
        "n_passages":    rag_db_size,
        "runs":          cfg["runs"],
        "embedding_type": "real (Wikipedia)",
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

    print(f"  [Vector Search]  "
          f"P50: {result['vector_search']['p50_ms']:.3f}ms  "
          f"P99: {result['vector_search']['p99_ms']:.3f}ms  "
          f"stddev: {result['vector_search']['stddev_ms']:.3f}ms")
    print(f"  [RAG TTFT]       "
          f"P50: {result['ttft']['p50_ms']:.1f}ms  "
          f"P99: {result['ttft']['p99_ms']:.1f}ms  "
          f"stddev: {result['ttft']['stddev_ms']:.1f}ms")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="x3d-rag-benchmark")
    parser.add_argument("--model",     default=DEFAULT["llm_model"],
                        help=f"ollama LLM model (default: {DEFAULT['llm_model']})")
    parser.add_argument("--db-size",   type=int, default=None,
                        help="single DB size override")
    parser.add_argument("--queries",   type=int, default=DEFAULT["n_queries"],
                        help=f"queries per run (default: {DEFAULT['n_queries']})")
    parser.add_argument("--runs",      type=int, default=DEFAULT["runs"],
                        help="number of runs to average (default: 1, recommended: 3)")
    parser.add_argument("--skip-rag",  action="store_true",
                        help="skip RAG TTFT (vector search only)")
    parser.add_argument("--quick",     action="store_true",
                        help="quick test — small DB only")
    parser.add_argument("--output",    default=None,
                        help="output JSON path (default: auto-generated)")
    parser.add_argument("--cache-dir", default=DEFAULT["cache_dir"],
                        help=f"embedding cache directory (default: {DEFAULT['cache_dir']})")
    parser.add_argument("--rebuild",   action="store_true",
                        help="force rebuild embedding cache")
    args = parser.parse_args()

    cfg = DEFAULT.copy()
    cfg["llm_model"]  = args.model
    cfg["n_queries"]  = args.queries
    cfg["runs"]       = max(1, args.runs)
    cfg["cache_dir"]  = args.cache_dir

    if args.db_size:
        cfg["db_sizes"] = [args.db_size]
    elif args.quick:
        cfg["db_sizes"] = [100_000]

    # Rebuild cache if requested
    if args.rebuild:
        for f in [CACHE_DB_FILE, CACHE_Q_FILE, CACHE_TEXT_FILE]:
            p = Path(cfg["cache_dir"]) / f
            if p.exists():
                p.unlink()
        print("  Cache cleared. Will rebuild on next run.")

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

    # ── Load embedding model & build/load cache ──
    print("\n[Setup] Loading embedding model & real embeddings")
    print("-"*60)
    print("  Loading embedding model...", end=" ", flush=True)
    embed_model = SentenceTransformer(cfg["embed_model"])
    print("done")

    db_embeddings, q_embeddings, passages = load_or_build_embeddings(cfg, embed_model)

    output = {
        "meta": {
            "cpu":            cpu,
            "gpu":            gpu,
            "timestamp":      now.isoformat(),
            "config":         cfg,
            "embedding_type": "real (Wikipedia)",
        },
        "vector_search":     [],
        "concurrent_search": {},
        "rag_ttft":          None,
    }

    # 1. Vector Search
    print("\n[1/3] Vector Search Benchmark (FAISS HNSW)")
    print("      Core CPU task in RAG — real Wikipedia embeddings")
    print("-"*60)
    for db_size in cfg["db_sizes"]:
        output["vector_search"].append(
            run_vector_search(db_size, cfg, db_embeddings, q_embeddings)
        )

    # 2. Concurrent Search
    print("\n[2/3] Concurrent Search Benchmark")
    print("      Multi-query throughput — X3D cache effect under load")
    print("-"*60)
    mid = cfg["db_sizes"][len(cfg["db_sizes"]) // 2]
    output["concurrent_search"] = run_concurrent_search(
        mid, cfg, db_embeddings, q_embeddings
    )

    # 3. RAG TTFT
    if not args.skip_rag:
        print("\n[3/3] RAG End-to-End TTFT Benchmark")
        print("      embed(GPU) → search(CPU) → LLM(GPU)")
        print("-"*60)
        output["rag_ttft"] = run_rag_ttft(
            cfg, embed_model, db_embeddings, q_embeddings, passages
        )
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
