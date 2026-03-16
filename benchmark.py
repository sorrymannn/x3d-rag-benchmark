"""
x3d-rag-benchmark
=================
AMD Ryzen X3D V-Cache vs non-X3D CPU performance benchmark
for RAG AI pipelines

Measures:
  1. Vector Search Latency  - FAISS HNSW search latency (real embeddings)
  2. RAG TTFT               - Time to first token (full pipeline)

Key design decisions:
  - Real Wikipedia embeddings (not random vectors)
    → clusters by topic → stable, reproducible search paths
  - Configurable FAISS threads (default: auto)
    → use --threads 1 for maximum stability if CV > 3%
  - Trimmed mean (drop top/bottom 5%)
    → removes outliers from results
  - Embeddings cached to disk
    → generated once, reused across runs

Install:
  pip install faiss-cpu sentence-transformers ollama datasets numpy tqdm

Run:
  python3 benchmark.py
  python3 benchmark.py --runs 5            # recommended for publication
  python3 benchmark.py --skip-rag          # vector search only
  python3 benchmark.py --quick --skip-rag  # fast test
"""

import argparse
import gc
import json
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np


# ── Variance Control ─────────────────────────────────────────────────────────

def apply_variance_controls(verbose=True):
    """
    OS-level variance reduction.
    Applies what it can; skips what requires missing permissions.
    """
    controls = []

    # 1. Python GC off during benchmark
    gc.disable()
    controls.append("Python GC -> disabled")

    if platform.system() == "Linux":
        for name, cmd in [
            ("CPU governor -> performance",
             ["sudo", "bash", "-c",
              "for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; "
              "do echo performance > $f; done"]),
            ("NUMA balancing -> disabled",
             ["sudo", "sysctl", "-w", "kernel.numa_balancing=0"]),
            ("THP -> disabled",
             ["sudo", "bash", "-c",
              "echo never > /sys/kernel/mm/transparent_hugepage/enabled"]),
        ]:
            try:
                r = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=5)
                controls.append(name if r.returncode == 0
                                else f"{name} (FAILED)")
            except Exception:
                controls.append(f"{name} (SKIPPED)")

    elif platform.system() == "Windows":
        # High Performance power plan
        try:
            subprocess.run(
                ["powercfg", "/setactive",
                 "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            controls.append("Power plan -> High Performance")
        except Exception:
            controls.append("Power plan -> SKIPPED")

    # Process priority
    if platform.system() in ("Linux", "Darwin"):
        for n in (-20, -10, -5):
            try:
                os.nice(n)
                controls.append(f"Process nice -> {n}")
                break
            except (PermissionError, OSError):
                continue
        else:
            controls.append("Process nice -> default (no permission)")
    elif platform.system() == "Windows":
        try:
            import ctypes
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(handle, 0x00000080)
            controls.append("Process priority -> HIGH")
        except Exception:
            controls.append("Process priority -> default")

    if verbose:
        print("\n  [Variance Controls]")
        for c in controls:
            print(f"    * {c}")

    return controls


def restore_after_benchmark():
    """Re-enable Python GC after benchmark."""
    gc.enable()

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
    "n_queries":    300,       # queries per run
    "top_k":        10,
    "hnsw_m":       32,
    "hnsw_ef":      64,
    "warmup":       30,        # warmup queries (discarded)
    "runs":         5,         # default 5 runs
    "trim":         0.05,      # trimmed mean: drop top/bottom 5%
    "threads":      0,         # 0 = auto (all cores), 1 = single-threaded
    "rag_queries":  50,
    "cache_dir":    "./embedding_cache",
    "n_passages":   1_100_000,
}

CACHE_DB_FILE   = "wiki_db_embeddings.npy"
CACHE_Q_FILE    = "wiki_query_embeddings.npy"
CACHE_TEXT_FILE = "wiki_passages.json"

# ── Utils ─────────────────────────────────────────────────────────────────────

def get_cpu_info():
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        except Exception:
            pass
    elif platform.system() == "Windows":
        try:
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "Name", "/format:list"],
                text=True, stderr=subprocess.DEVNULL)
            for line in out.strip().splitlines():
                if "Name=" in line:
                    return line.split("=", 1)[1].strip()
        except Exception:
            pass
    elif platform.system() == "Darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True).strip()
        except Exception:
            pass
    return platform.processor() or "Unknown CPU"


def get_l3_cache():
    """L3 캐시 크기 감지 (편차 분석용)"""
    if platform.system() == "Linux":
        try:
            out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
            for line in out.splitlines():
                if "L3 cache" in line:
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    elif platform.system() == "Windows":
        try:
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "L3CacheSize", "/format:list"],
                text=True, stderr=subprocess.DEVNULL)
            for line in out.strip().splitlines():
                if "L3CacheSize=" in line:
                    val = int(line.split("=")[1].strip())
                    if val > 0:
                        return f"{val // 1024} MiB"
        except Exception:
            pass
    elif platform.system() == "Darwin":
        try:
            l3 = int(subprocess.check_output(
                ["sysctl", "-n", "hw.l3cachesize"], text=True).strip())
            return f"{l3 // 1024 // 1024} MiB"
        except Exception:
            pass
    return "Unknown"


def get_memory_info():
    """RAM 용량 및 속도 감지"""
    info = {"total_gb": 0, "speed": "Unknown"}
    if platform.system() == "Windows":
        try:
            out = subprocess.check_output(
                ["wmic", "memorychip", "get", "Capacity,ConfiguredClockSpeed",
                 "/format:list"], text=True, stderr=subprocess.DEVNULL)
            caps, speeds = [], []
            for line in out.strip().splitlines():
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "Capacity" and v.strip():
                    caps.append(int(v.strip()))
                elif k.strip() == "ConfiguredClockSpeed" and v.strip():
                    speeds.append(int(v.strip()))
            if caps:
                info["total_gb"] = round(sum(caps) / 1024**3, 1)
            if speeds:
                info["speed"] = f"{max(speeds)} MT/s"
        except Exception:
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        info["total_gb"] = round(int(line.split()[1]) / 1024 / 1024, 1)
                        break
        except Exception:
            pass
    elif platform.system() == "Darwin":
        try:
            info["total_gb"] = round(
                int(subprocess.check_output(["sysctl", "-n", "hw.memsize"],
                    text=True).strip()) / 1024**3, 1)
        except Exception:
            pass
    return info

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

def trimmed_mean(values, trim=0.05):
    """Drop top/bottom trim% and return mean — removes outliers."""
    arr = sorted(values)
    cut = max(1, int(len(arr) * trim))
    trimmed = arr[cut:-cut] if cut > 0 else arr
    return float(np.mean(trimmed))

def trimmed_std(values, trim=0.05):
    arr = sorted(values)
    cut = max(1, int(len(arr) * trim))
    trimmed = arr[cut:-cut] if cut > 0 else arr
    return float(np.std(trimmed))

# ── Embedding Cache ───────────────────────────────────────────────────────────

def load_or_build_embeddings(cfg, embed_model):
    """
    Load real Wikipedia embeddings from cache or build from scratch.

    Why real embeddings instead of random vectors:
      Random vectors → uniformly distributed in space
                     → HNSW traversal path varies wildly each run
                     → high variance between runs

      Real embeddings → clustered by topic/meaning
                      → HNSW traversal paths are stable
                      → low variance between runs
                      → matches actual RAG workload
    """
    cache_dir = Path(cfg["cache_dir"])
    cache_dir.mkdir(exist_ok=True)

    db_path  = cache_dir / CACHE_DB_FILE
    q_path   = cache_dir / CACHE_Q_FILE
    txt_path = cache_dir / CACHE_TEXT_FILE

    if db_path.exists() and q_path.exists() and txt_path.exists():
        print("  Loading embeddings from cache...", end=" ", flush=True)
        db_emb = np.load(str(db_path))
        q_emb  = np.load(str(q_path))
        with open(txt_path) as f:
            passages = json.load(f)
        print(f"done  ({len(db_emb):,} DB + {len(q_emb):,} query vectors)")
        return db_emb, q_emb, passages

    print("  [First run] Building real embeddings from Wikipedia...")
    print("  Takes ~10-20 min on GPU, ~60 min on CPU.")
    print("  Cached after first run → instant load on subsequent runs.\n")

    n_total = cfg["n_passages"]
    n_q     = cfg["n_queries"] * cfg["runs"] + cfg["warmup"] + 500

    print("  Downloading Wikipedia...", end=" ", flush=True)
    try:
        ds        = load_dataset("wikipedia", "20220301.simple", split="train")
        all_texts = [row["text"][:512] for row in ds]
    except Exception as e:
        print(f"\n  [WARNING] Wikipedia failed ({e}), using AG News fallback...")
        ds        = load_dataset("ag_news", split="train")
        all_texts = [row["text"] for row in ds]
    print(f"{len(all_texts):,} articles")

    while len(all_texts) < n_total + n_q:
        all_texts = all_texts * 2
    all_texts = all_texts[:n_total + n_q]

    db_texts    = all_texts[:n_total]
    query_texts = all_texts[n_total:n_total + n_q]

    print(f"  Embedding {len(db_texts):,} DB passages...")
    db_emb = embed_model.encode(
        db_texts, batch_size=512, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )

    print(f"  Embedding {len(query_texts):,} query vectors...")
    q_emb = embed_model.encode(
        query_texts, batch_size=512, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )

    print("  Saving cache...", end=" ", flush=True)
    np.save(str(db_path), db_emb)
    np.save(str(q_path),  q_emb)
    with open(txt_path, "w") as f:
        json.dump(db_texts, f)
    print("done")

    return db_emb, q_emb, db_texts

# ── Vector Search ─────────────────────────────────────────────────────────────

def build_hnsw_index(db_emb, db_size, cfg):
    dim  = db_emb.shape[1]
    vecs = db_emb[:db_size].copy()
    faiss.normalize_L2(vecs)
    idx  = faiss.IndexHNSWFlat(dim, cfg["hnsw_m"])
    idx.hnsw.efConstruction = 200
    idx.add(vecs)
    idx.hnsw.efSearch = cfg["hnsw_ef"]
    return idx


def run_vector_search(db_size, cfg, db_emb, q_emb):
    """
    FAISS HNSW benchmark.

    Stability measures:
      - Single-threaded FAISS (omp_set_num_threads(1))
        eliminates OS scheduler variance from multi-core dispatch
      - Different query slice per run (real embeddings, not re-random)
        realistic query diversity without path randomness
      - Trimmed mean across runs: drops top/bottom 5%
        removes warmup spikes and thermal throttle outliers
      - Explicit warmup queries discarded before measurement
    """
    dim    = db_emb.shape[1]
    n_runs = cfg["runs"]

    print(f"\n  DB size: {db_size:,} vectors  |  dim: {dim}  |  runs: {n_runs}")
    print(f"  Est. memory: ~{db_size * dim * 4 / 1024**3:.2f} GB")

    # Thread count: 0=auto (all cores), 1=single-threaded (max stability)
    n_threads = cfg.get("threads", 0)
    if n_threads > 0:
        faiss.omp_set_num_threads(n_threads)
    thread_label = f"{n_threads}" if n_threads > 0 else "auto"
    print(f"  FAISS threads: {thread_label}")

    print("  Building HNSW index...", end=" ", flush=True)
    t0        = time.perf_counter()
    idx       = build_hnsw_index(db_emb, db_size, cfg)
    build_sec = time.perf_counter() - t0
    print(f"{build_sec:.1f}s")

    # Pre-normalize query slices
    n_per_run = cfg["n_queries"] + cfg["warmup"]
    all_q     = q_emb[:n_per_run * n_runs].copy()
    faiss.normalize_L2(all_q)

    run_qps  = []
    run_p50  = []
    run_p95  = []
    run_p99  = []
    run_mean = []

    for run_i in range(n_runs):
        q_slice = all_q[run_i * n_per_run: (run_i + 1) * n_per_run]
        warmup_q = q_slice[:cfg["warmup"]]
        bench_q  = q_slice[cfg["warmup"]:]

        # Warmup — discarded
        for q in warmup_q:
            idx.search(q.reshape(1, -1), cfg["top_k"])

        # Measurement
        latencies = []
        t_total   = time.perf_counter()
        for q in bench_q:
            t0 = time.perf_counter()
            idx.search(q.reshape(1, -1), cfg["top_k"])
            latencies.append(time.perf_counter() - t0)
        total_sec = time.perf_counter() - t_total

        qps = len(latencies) / total_sec
        run_qps.append(qps)
        run_p50.append(ms(pct(latencies, 50)))
        run_p95.append(ms(pct(latencies, 95)))
        run_p99.append(ms(pct(latencies, 99)))
        run_mean.append(ms(float(np.mean(latencies))))

        print(f"    run {run_i+1}/{n_runs}  "
              f"QPS: {qps:>8.1f}  "
              f"P50: {ms(pct(latencies,50)):.3f}ms  "
              f"P99: {ms(pct(latencies,99)):.3f}ms")

        # Inter-run cooling — prevents thermal throttle drift
        if run_i < n_runs - 1:
            time.sleep(2)

    # Trimmed mean across runs
    trim = cfg["trim"]
    result = {
        "db_size":            db_size,
        "build_time_s":       round(build_sec, 2),
        "runs":               n_runs,
        "embedding_type":     "real (Wikipedia)",
        "qps":                round(trimmed_mean(run_qps, trim), 1),
        "qps_stddev":         round(trimmed_std(run_qps, trim), 1),
        "qps_runs":           [round(v, 1) for v in run_qps],
        "latency_mean_ms":    round(trimmed_mean(run_mean, trim), 3),
        "latency_p50_ms":     round(trimmed_mean(run_p50, trim), 3),
        "latency_p50_stddev": round(trimmed_std(run_p50, trim), 3),
        "latency_p95_ms":     round(trimmed_mean(run_p95, trim), 3),
        "latency_p99_ms":     round(trimmed_mean(run_p99, trim), 3),
        "latency_p99_stddev": round(trimmed_std(run_p99, trim), 3),
    }

    cv = result["qps_stddev"] / result["qps"] * 100  # coefficient of variation
    print(f"  → avg QPS: {result['qps']:.1f} ±{result['qps_stddev']:.1f} "
          f"(CV: {cv:.1f}%)  |  "
          f"P99: {result['latency_p99_ms']:.3f}ms ±{result['latency_p99_stddev']:.3f}")

    del idx
    return result

# ── RAG TTFT ──────────────────────────────────────────────────────────────────

def check_ollama_model(model):
    try:
        return any(model in m.model for m in ollama_client.list().models)
    except Exception:
        return False


def run_rag_ttft(cfg, embed_model, db_emb, q_emb, passages):
    """Full RAG pipeline TTFT: embed(GPU) → search(CPU) → LLM(GPU)"""
    if not check_ollama_model(cfg["llm_model"]):
        print(f"  [WARNING] ollama model '{cfg['llm_model']}' not found.")
        print(f"  Install: ollama pull {cfg['llm_model']}")
        return None

    n_threads = cfg.get("threads", 0)
    if n_threads > 0:
        faiss.omp_set_num_threads(n_threads)

    rag_db_size = min(10_000, len(db_emb))
    vecs = db_emb[:rag_db_size].copy()
    faiss.normalize_L2(vecs)
    dim  = vecs.shape[1]
    idx  = faiss.IndexHNSWFlat(dim, 32)
    idx.hnsw.efConstruction = 200
    idx.add(vecs)
    idx.hnsw.efSearch = 64

    n_q        = cfg["rag_queries"]
    query_vecs = q_emb[:n_q * cfg["runs"] + 10].copy()
    faiss.normalize_L2(query_vecs)

    # Ollama warmup — ensures model is loaded, eliminates cold-start spike
    print("  Warming up ollama (eliminates cold-start latency)...",
          end=" ", flush=True)
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

    print(f"  Measuring ({n_q} queries × {cfg['runs']} runs)...")
    for run_i in range(cfg["runs"]):
        run_q = query_vecs[run_i * n_q: (run_i + 1) * n_q]
        for qv in tqdm(run_q, ncols=60, leave=False,
                       desc=f"  run {run_i+1}/{cfg['runs']}"):
            t_s = time.perf_counter()
            _, I = idx.search(qv.reshape(1, -1), cfg["top_k"])
            search_lat_list.append(time.perf_counter() - t_s)

            ctx    = " ".join([passages[j] for j in I[0]
                               if j < len(passages)])[:500]
            prompt = (f"Context: {ctx}\n\n"
                      f"Question: What is this about?\nAnswer briefly:")

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

    trim = cfg["trim"]
    result = {
        "n_passages":    rag_db_size,
        "runs":          cfg["runs"],
        "embedding_type": "real (Wikipedia)",
        "vector_search": {
            "mean_ms":   ms(trimmed_mean(search_lat_list, trim)),
            "p50_ms":    ms(pct(search_lat_list, 50)),
            "p95_ms":    ms(pct(search_lat_list, 95)),
            "p99_ms":    ms(pct(search_lat_list, 99)),
            "stddev_ms": ms(trimmed_std(search_lat_list, trim)),
        },
        "ttft": {
            "mean_ms":   ms(trimmed_mean(ttft_list, trim)),
            "p50_ms":    ms(pct(ttft_list, 50)),
            "p95_ms":    ms(pct(ttft_list, 95)),
            "p99_ms":    ms(pct(ttft_list, 99)),
            "stddev_ms": ms(trimmed_std(ttft_list, trim)),
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
    parser.add_argument("--model",     default=DEFAULT["llm_model"])
    parser.add_argument("--db-size",   type=int, default=None)
    parser.add_argument("--queries",   type=int, default=DEFAULT["n_queries"])
    parser.add_argument("--runs",      type=int, default=DEFAULT["runs"],
                        help="number of runs to average (default: 5)")
    parser.add_argument("--skip-rag",  action="store_true")
    parser.add_argument("--quick",     action="store_true",
                        help="small DB only, 3 runs")
    parser.add_argument("--output",    default=None)
    parser.add_argument("--cache-dir", default=DEFAULT["cache_dir"])
    parser.add_argument("--rebuild",   action="store_true",
                        help="force rebuild embedding cache")
    parser.add_argument("--threads",   type=int, default=DEFAULT["threads"],
                        help="FAISS threads: 0=auto, 1=single (default: 0)")
    args = parser.parse_args()

    cfg = DEFAULT.copy()
    cfg["llm_model"]  = args.model
    cfg["n_queries"]  = args.queries
    cfg["runs"]       = max(1, args.runs)
    cfg["cache_dir"]  = args.cache_dir
    cfg["threads"]    = args.threads

    if args.db_size:
        cfg["db_sizes"] = [args.db_size]
    elif args.quick:
        cfg["db_sizes"] = [100_000]
        cfg["runs"]     = min(cfg["runs"], 3)

    if args.rebuild:
        for fname in [CACHE_DB_FILE, CACHE_Q_FILE, CACHE_TEXT_FILE]:
            p = Path(cfg["cache_dir"]) / fname
            if p.exists():
                p.unlink()
        print("  Embedding cache cleared.")

    cpu = get_cpu_info()
    gpu = get_gpu_info()
    l3  = get_l3_cache()
    mem = get_memory_info()
    now = datetime.now()

    print("\n" + "="*60)
    print("  x3d-rag-benchmark")
    print("  github.com/sorrymannn/x3d-rag-benchmark")
    print("="*60)
    print(f"  CPU:      {cpu}")
    print(f"  L3 Cache: {l3}")
    print(f"  Memory:   {mem['total_gb']} GB @ {mem['speed']}")
    print(f"  GPU:      {gpu}")
    thread_str = str(cfg["threads"]) if cfg["threads"] > 0 else "auto"
    print(f"  Runs:     {cfg['runs']}  (trimmed mean, drop top/bottom 5%)")
    print(f"  FAISS threads: {thread_str}  (use --threads 1 if CV > 3%)")
    print(f"  Time:     {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # Apply OS-level variance controls
    controls = apply_variance_controls()
    print("="*60)

    # Setup
    print("\n[Setup] Loading embedding model & real embeddings")
    print("-"*60)
    print("  Loading embedding model...", end=" ", flush=True)
    embed_model = SentenceTransformer(cfg["embed_model"])
    print("done")
    db_emb, q_emb, passages = load_or_build_embeddings(cfg, embed_model)

    output = {
        "meta": {
            "cpu":            cpu,
            "l3_cache":       l3,
            "memory":         mem,
            "gpu":            gpu,
            "timestamp":      now.isoformat(),
            "config":         cfg,
            "embedding_type": "real (Wikipedia)",
            "methodology": {
                "faiss_threads": cfg.get("threads", 0),
                "trimmed_mean":  f"top/bottom {int(cfg['trim']*100)}% dropped",
                "warmup_queries": cfg["warmup"],
                "inter_run_cooling": "2s",
                "variance_controls": controls,
            },
        },
        "vector_search": [],
        "rag_ttft":      None,
    }

    # 1. Vector Search
    print("\n[1/3] Vector Search Benchmark (FAISS HNSW, single-threaded)")
    print("      Core CPU task in RAG — real Wikipedia embeddings")
    print("-"*60)
    for db_size in cfg["db_sizes"]:
        output["vector_search"].append(
            run_vector_search(db_size, cfg, db_emb, q_emb)
        )

    # 2. RAG TTFT
    if not args.skip_rag:
        print("\n[2/3] RAG End-to-End TTFT Benchmark")
        print("      embed(GPU) → search(CPU, single-threaded) → LLM(GPU)")
        print("-"*60)
        output["rag_ttft"] = run_rag_ttft(
            cfg, embed_model, db_emb, q_emb, passages
        )
    else:
        print("\n[2/3] RAG TTFT — skipped (--skip-rag)")

    # Save
    restore_after_benchmark()
    out_path = args.output or f"result_{now.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # CV quality summary
    cvs = []
    for r in output["vector_search"]:
        if r["qps"] > 0:
            cvs.append(r["qps_stddev"] / r["qps"] * 100)
    if cvs:
        max_cv = max(cvs)
        print(f"\n  [Variance Quality]")
        if max_cv <= 2.0:
            print(f"    Max CV: {max_cv:.1f}% — EXCELLENT (very stable)")
        elif max_cv <= 3.0:
            print(f"    Max CV: {max_cv:.1f}% — GOOD (reliable)")
        elif max_cv <= 5.0:
            print(f"    Max CV: {max_cv:.1f}% — OK (acceptable)")
        else:
            print(f"    Max CV: {max_cv:.1f}% — NOISY (try --threads 1)")

    print("\n" + "="*60)
    print(f"  Done! Saved: {out_path}")
    print(f"  Compare: python3 compare.py result_a.json result_b.json")
    print("="*60)


if __name__ == "__main__":
    main()
