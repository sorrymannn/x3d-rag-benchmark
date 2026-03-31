"""
x3d-rag-benchmark - Full CPU Benchmark
=======================================
Measures whole-CPU performance in GPU-centric AI inference environments.

While benchmark.py isolates single-core L3 cache effects,
this script measures how the ENTIRE CPU performs under realistic
multi-core AI workloads where L3 cache contention matters.

Tests:
  1. Batch Vector Search      - FAISS HNSW with all cores (OMP threads)
     > L3 cache contention across CCDs
  2. Concurrent RAG Pipeline  - Multiple RAG requests in parallel
     > L3 cache competition between independent workers
  3. Data Feeding Throughput  - Tokenizer > GPU tensor transfer rate
     > CPU preparation speed that determines GPU utilization
  4. Index Build Time         - HNSW index construction from embeddings
     > One-time setup cost users actually wait for

Shares embedding cache with benchmark.py (./embedding_cache/)

Install:
  pip install faiss-cpu sentence-transformers ollama datasets numpy tqdm transformers torch

Run:
  python benchmark_full.py
  python benchmark_full.py --skip-rag              # skip concurrent RAG
  python benchmark_full.py --skip-feeding           # skip data feeding
  python benchmark_full.py --skip-build             # skip index build test
  python benchmark_full.py --quick                  # small DB, fewer runs
  python benchmark_full.py --output 9950x3d.json
"""

import argparse
import gc
import json
import multiprocessing as mp
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np


# == Variance Control =========================================================

def apply_variance_controls(verbose=True):
    controls = []
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
        try:
            subprocess.run(
                ["powercfg", "/setactive",
                 "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            controls.append("Power plan -> High Performance")
        except Exception:
            controls.append("Power plan -> SKIPPED")

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
    gc.enable()


# == Dependency check =========================================================

def check_deps():
    missing = []
    for pkg in ["faiss", "sentence_transformers", "ollama", "datasets",
                "tqdm", "transformers", "torch"]:
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
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer
import ollama as ollama_client


# == Config ===================================================================

DEFAULT = {
    "embed_model": "all-MiniLM-L6-v2",
    "llm_model": "llama3.2",
    "db_sizes": [100_000, 200_000],
    "batch_queries": 3000,        # fixed for fair comparison across CPUs
    "top_k": 10,
    "hnsw_m": 32,
    "hnsw_ef": 64,
    "warmup_batches": 5,
    "runs": 10,
    "trim": 0.05,
    "rag_workers": None,
    "rag_queries_per_worker": 20,
    "rag_runs": 5,
    "feeding_batch_size": 32,
    "feeding_n_batches": 200,
    "feeding_runs": 5,
    "build_runs": 5,
    "cache_dir": "./embedding_cache",
    "n_passages": 1_100_000,
}

CACHE_DB_FILE = "wiki_db_embeddings.npy"
CACHE_Q_FILE = "wiki_query_embeddings.npy"
CACHE_TEXT_FILE = "wiki_passages.json"


# == Utils ====================================================================

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
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
            return name.strip()
        except Exception:
            pass
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                text=True, stderr=subprocess.DEVNULL, timeout=10)
            if out.strip():
                return out.strip()
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


def get_cpu_count():
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


def get_l3_cache():
    if platform.system() == "Linux":
        try:
            out = subprocess.check_output(["lscpu"], text=True,
                                          stderr=subprocess.DEVNULL)
            for line in out.splitlines():
                if "L3 cache" in line:
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    elif platform.system() == "Windows":
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).L3CacheSize"],
                text=True, stderr=subprocess.DEVNULL, timeout=10)
            val = int(out.strip())
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
    info = {"total_gb": 0, "speed": "Unknown"}
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        info["total_gb"] = round(
                            int(line.split()[1]) / 1024 / 1024, 1)
                        break
        except Exception:
            pass
    elif platform.system() == "Windows":
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_PhysicalMemory | "
                 "Measure-Object -Property Capacity -Sum).Sum"],
                text=True, stderr=subprocess.DEVNULL, timeout=10)
            total = int(out.strip())
            if total > 0:
                info["total_gb"] = round(total / 1024**3, 1)
        except Exception:
            pass
    elif platform.system() == "Darwin":
        try:
            info["total_gb"] = round(
                int(subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"],
                    text=True).strip()) / 1024**3, 1)
        except Exception:
            pass
    return info


def get_gpu_info():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"], text=True).strip()
    except Exception:
        return "N/A"


def pct(data, p):
    return float(np.percentile(data, p, method="linear"))


def ms(t):
    return round(t * 1000, 3)


def trimmed_mean(values, trim=0.05):
    arr = sorted(values)
    cut = max(1, int(len(arr) * trim))
    trimmed = arr[cut:-cut] if cut > 0 and len(arr) > 2 * cut else arr
    return float(np.mean(trimmed))


def trimmed_std(values, trim=0.05):
    arr = sorted(values)
    cut = max(1, int(len(arr) * trim))
    trimmed = arr[cut:-cut] if cut > 0 and len(arr) > 2 * cut else arr
    return float(np.std(trimmed))


# == Embedding Cache (shared with benchmark.py) ===============================

def load_or_build_embeddings(cfg, embed_model):
    cache_dir = Path(cfg["cache_dir"])
    cache_dir.mkdir(exist_ok=True)

    db_path = cache_dir / CACHE_DB_FILE
    q_path = cache_dir / CACHE_Q_FILE
    txt_path = cache_dir / CACHE_TEXT_FILE

    if db_path.exists() and q_path.exists() and txt_path.exists():
        print("  Loading embeddings from cache...", end=" ", flush=True)
        db_emb = np.load(str(db_path))
        q_emb = np.load(str(q_path))
        with open(txt_path) as f:
            passages = json.load(f)
        print(f"done ({len(db_emb):,} DB + {len(q_emb):,} query vectors)")
        return db_emb, q_emb, passages

    print("  [First run] Building real embeddings from Wikipedia...")
    print("  Run benchmark.py first to build the cache, or wait ~10-20 min.")

    n_total = cfg["n_passages"]
    n_q = 5000

    print("  Downloading Wikipedia...", end=" ", flush=True)
    try:
        ds = load_dataset("wikipedia", "20220301.simple", split="train")
        all_texts = [row["text"][:512] for row in ds]
    except Exception as e:
        print(f"\n  [WARNING] Wikipedia failed ({e}), using AG News...")
        ds = load_dataset("ag_news", split="train")
        all_texts = [row["text"] for row in ds]
    print(f"{len(all_texts):,} articles")

    while len(all_texts) < n_total + n_q:
        all_texts = all_texts * 2
    all_texts = all_texts[:n_total + n_q]

    db_texts = all_texts[:n_total]
    query_texts = all_texts[n_total:n_total + n_q]

    print(f"  Embedding {len(db_texts):,} DB passages...")
    db_emb = embed_model.encode(
        db_texts, batch_size=512, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True)

    print(f"  Embedding {len(query_texts):,} query vectors...")
    q_emb = embed_model.encode(
        query_texts, batch_size=512, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True)

    print("  Saving cache...", end=" ", flush=True)
    np.save(str(db_path), db_emb)
    np.save(str(q_path), q_emb)
    with open(txt_path, "w") as f:
        json.dump(db_texts, f)
    print("done")
    return db_emb, q_emb, db_texts


# == Test 1: Batch Vector Search ==============================================

def build_hnsw_index(db_emb, db_size, cfg):
    dim = db_emb.shape[1]
    vecs = db_emb[:db_size].copy()
    faiss.normalize_L2(vecs)
    idx = faiss.IndexHNSWFlat(dim, cfg["hnsw_m"])
    idx.hnsw.efConstruction = 200
    idx.add(vecs)
    idx.hnsw.efSearch = cfg["hnsw_ef"]
    return idx


def run_batch_vector_search(db_size, cfg, db_emb, q_emb):
    """
    Batch vector search: all CPU cores search simultaneously.

    FAISS distributes N queries across OMP threads.
    Each thread traverses HNSW independently, accessing its own
    region of the index in L3 cache.

    With asymmetric CCD (e.g. 4585PX: 96MB + 32MB),
    threads on the smaller-cache CCD suffer more cache misses.
    This is THE test that shows L3 contention across CCDs.
    """
    dim = db_emb.shape[1]
    n_runs = cfg["runs"]
    n_cores = get_cpu_count()

    # Fixed batch size for fair cross-CPU comparison
    n_batch = cfg["batch_queries"]
    # Cap at available queries (will tile if needed anyway)
    print(f"\n  DB size: {db_size:,} | batch: {n_batch} queries | "
          f"cores: {n_cores} | runs: {n_runs}")

    faiss.omp_set_num_threads(n_cores)

    print("  Building HNSW index...", end=" ", flush=True)
    t0 = time.perf_counter()
    idx = build_hnsw_index(db_emb, db_size, cfg)
    build_sec = time.perf_counter() - t0
    print(f"{build_sec:.1f}s")

    n_needed = n_batch * (n_runs + cfg["warmup_batches"])
    if n_needed <= len(q_emb):
        all_q = q_emb[:n_needed].copy()
    else:
        repeats = (n_needed // len(q_emb)) + 1
        all_q = np.tile(q_emb, (repeats, 1))[:n_needed].copy()
    faiss.normalize_L2(all_q)

    print("  Warming up...", end=" ", flush=True)
    for i in range(cfg["warmup_batches"]):
        batch = all_q[i * n_batch:(i + 1) * n_batch]
        idx.search(batch, cfg["top_k"])
    print("done")

    offset = cfg["warmup_batches"] * n_batch
    run_qps = []
    run_latency = []

    for run_i in range(n_runs):
        batch = all_q[offset + run_i * n_batch:
                      offset + (run_i + 1) * n_batch]

        t0 = time.perf_counter()
        idx.search(batch, cfg["top_k"])
        elapsed = time.perf_counter() - t0

        qps = n_batch / elapsed
        avg_lat = elapsed / n_batch
        run_qps.append(qps)
        run_latency.append(ms(avg_lat))

        print(f"    run {run_i + 1}/{n_runs}  "
              f"QPS: {qps:>10,.1f}  "
              f"avg latency: {ms(avg_lat):.3f}ms  "
              f"batch time: {elapsed:.3f}s")

        if run_i < n_runs - 1:
            time.sleep(2)

    trim = cfg["trim"]
    result = {
        "db_size": db_size,
        "build_time_s": round(build_sec, 2),
        "batch_size": n_batch,
        "cores_used": n_cores,
        "runs": n_runs,
        "qps": round(trimmed_mean(run_qps, trim), 1),
        "qps_stddev": round(trimmed_std(run_qps, trim), 1),
        "qps_runs": [round(v, 1) for v in run_qps],
        "avg_latency_ms": round(trimmed_mean(run_latency, trim), 3),
        "avg_latency_stddev_ms": round(trimmed_std(run_latency, trim), 3),
    }

    cv = result["qps_stddev"] / result["qps"] * 100 if result["qps"] > 0 else 0
    print(f"  -> batch QPS: {result['qps']:,.1f} +/-{result['qps_stddev']:.1f} "
          f"(CV: {cv:.1f}%) | avg lat: {result['avg_latency_ms']:.3f}ms")

    del idx
    return result


# == Test 2: Concurrent RAG Pipeline =========================================

def _rag_worker(args):
    """
    Single RAG worker process.
    Each worker builds its own small FAISS index, runs queries, returns timing.
    Workers compete for L3 cache and memory bandwidth.
    """
    worker_id, db_emb_path, q_emb_path, passages_path, cfg = args

    import faiss as faiss_local
    import ollama as ollama_local

    rag_db_size = 10_000

    # Memory-efficient: mmap full file, copy only 10K slice
    db_mmap = np.load(db_emb_path, mmap_mode='r')
    vecs = db_mmap[:rag_db_size].copy()
    del db_mmap

    q_mmap = np.load(q_emb_path, mmap_mode='r')
    q_emb = q_mmap[:cfg["rag_queries_per_worker"]].copy()
    del q_mmap

    # Load pre-sliced small passages file (only 10K entries)
    with open(passages_path) as f:
        passages = json.load(f)

    faiss_local.normalize_L2(vecs)
    dim = vecs.shape[1]

    faiss_local.omp_set_num_threads(1)

    idx = faiss_local.IndexHNSWFlat(dim, 32)
    idx.hnsw.efConstruction = 200
    idx.add(vecs)
    idx.hnsw.efSearch = 64

    n_q = cfg["rag_queries_per_worker"]
    query_vecs = q_emb[:n_q].copy()
    faiss_local.normalize_L2(query_vecs)

    ttft_list = []
    search_list = []

    for qv in query_vecs:
        t_s = time.perf_counter()
        _, I = idx.search(qv.reshape(1, -1), cfg["top_k"])
        search_list.append(time.perf_counter() - t_s)

        ctx = " ".join([passages[j] for j in I[0]
                        if j < len(passages)])[:500]
        prompt = (f"Context: {ctx}\n\n"
                  f"Question: What is this about?\nAnswer briefly:")

        t_start = time.perf_counter()
        try:
            stream = ollama_local.chat(
                model=cfg["llm_model"],
                messages=[{"role": "user", "content": prompt}],
                stream=True)
            for chunk in stream:
                ttft_list.append(time.perf_counter() - t_start)
                break
        except Exception:
            pass

    return {
        "worker_id": worker_id,
        "queries": n_q,
        "ttft_list": ttft_list,
        "search_list": search_list,
    }


def run_concurrent_rag(cfg, db_emb_path, q_emb_path, passages_path):
    """
    Concurrent RAG: multiple workers run full RAG pipelines in parallel.

    Key difference from batch search:
    - Batch search = one process, many OMP threads, shared index
    - Concurrent RAG = many processes, each with own index + LLM call
    - Tests L3 cache competition between independent processes
    """
    n_workers = cfg["rag_workers"] or get_cpu_count()
    n_workers = min(n_workers, 8)

    # Cap based on available memory
    mem = get_memory_info()
    if mem["total_gb"] > 0:
        mem_cap = max(2, int(mem["total_gb"] / 2))
        n_workers = min(n_workers, mem_cap)

    n_runs = cfg["rag_runs"]

    print(f"\n  Workers: {n_workers} | "
          f"queries/worker: {cfg['rag_queries_per_worker']} | "
          f"runs: {n_runs}")

    # Pre-slice passages to small temp file to save worker memory
    import tempfile
    with open(passages_path) as f:
        all_passages = json.load(f)
    small_passages = all_passages[:10_000]
    del all_passages

    small_passages_path = os.path.join(
        tempfile.gettempdir(), "rag_passages_10k.json")
    with open(small_passages_path, "w") as f:
        json.dump(small_passages, f)
    del small_passages

    # Warmup ollama
    print("  Warming up ollama...", end=" ", flush=True)
    try:
        stream = ollama_client.chat(
            model=cfg["llm_model"],
            messages=[{"role": "user", "content": "Say hi"}],
            stream=True)
        for chunk in stream:
            break
    except Exception:
        pass
    print("done")

    all_run_results = []

    for run_i in range(n_runs):
        worker_args = [
            (w, db_emb_path, q_emb_path, small_passages_path, cfg)
            for w in range(n_workers)
        ]

        t0 = time.perf_counter()
        with mp.Pool(n_workers) as pool:
            results = pool.map(_rag_worker, worker_args)
        wall_time = time.perf_counter() - t0

        total_queries = sum(r["queries"] for r in results)
        all_ttft = []
        all_search = []
        for r in results:
            all_ttft.extend(r["ttft_list"])
            all_search.extend(r["search_list"])

        throughput = total_queries / wall_time
        all_run_results.append({
            "wall_time_s": round(wall_time, 2),
            "throughput_qps": round(throughput, 2),
            "total_queries": total_queries,
            "avg_ttft_ms": ms(float(np.mean(all_ttft))) if all_ttft else 0,
            "avg_search_ms": ms(float(np.mean(all_search))) if all_search else 0,
        })

        print(f"    run {run_i + 1}/{n_runs}  "
              f"throughput: {throughput:.1f} req/s  "
              f"wall: {wall_time:.1f}s  "
              f"avg TTFT: {ms(float(np.mean(all_ttft))):.1f}ms")

        if run_i < n_runs - 1:
            time.sleep(3)

    trim = cfg["trim"]
    throughputs = [r["throughput_qps"] for r in all_run_results]
    ttfts = [r["avg_ttft_ms"] for r in all_run_results]

    result = {
        "n_workers": n_workers,
        "queries_per_worker": cfg["rag_queries_per_worker"],
        "runs": n_runs,
        "throughput_qps": round(trimmed_mean(throughputs, trim), 2),
        "throughput_stddev": round(trimmed_std(throughputs, trim), 2),
        "avg_ttft_ms": round(trimmed_mean(ttfts, trim), 1),
        "run_details": all_run_results,
    }

    print(f"  -> concurrent throughput: {result['throughput_qps']:.1f} req/s "
          f"+/-{result['throughput_stddev']:.1f} | "
          f"avg TTFT: {result['avg_ttft_ms']:.1f}ms")

    # Cleanup temp file
    try:
        os.remove(small_passages_path)
    except Exception:
        pass

    return result


# == Test 3: Data Feeding Throughput ==========================================

def run_data_feeding(cfg, texts):
    """
    CPU -> GPU data feeding pipeline.

    Tokenize text on CPU -> convert to tensors -> transfer to GPU.
    In LLM inference, slow tokenization = GPU starved = wasted GPU cycles.

    L3 cache affects tokenizer vocabulary lookups and tensor allocation.
    """
    has_gpu = torch.cuda.is_available()
    device = "cuda" if has_gpu else "cpu"

    batch_size = cfg["feeding_batch_size"]
    n_batches = cfg["feeding_n_batches"]
    n_runs = cfg["feeding_runs"]
    n_texts = batch_size * n_batches

    if n_texts > len(texts):
        test_texts = (texts * ((n_texts // len(texts)) + 1))[:n_texts]
    else:
        test_texts = texts[:n_texts]

    print(f"\n  Batches: {n_batches} x {batch_size} = {n_texts:,} texts | "
          f"target: {device} | runs: {n_runs}")

    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2")

    # Warmup
    print("  Warming up...", end=" ", flush=True)
    for i in range(10):
        batch = test_texts[i * batch_size:(i + 1) * batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True,
                           max_length=128, return_tensors="pt")
        if has_gpu:
            _ = {k: v.to(device) for k, v in tokens.items()}
    if has_gpu:
        torch.cuda.synchronize()
    print("done")

    run_throughput = []

    for run_i in range(n_runs):
        t0 = time.perf_counter()
        for b_i in range(n_batches):
            batch = test_texts[b_i * batch_size:(b_i + 1) * batch_size]
            tokens = tokenizer(batch, padding=True, truncation=True,
                               max_length=128, return_tensors="pt")
            if has_gpu:
                tokens = {k: v.to(device) for k, v in tokens.items()}
        if has_gpu:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        tps = n_texts / elapsed
        run_throughput.append(tps)

        print(f"    run {run_i + 1}/{n_runs}  "
              f"{tps:,.0f} texts/s  ({elapsed:.1f}s)")

        if run_i < n_runs - 1:
            time.sleep(2)

    trim = cfg["trim"]
    result = {
        "n_texts": n_texts,
        "batch_size": batch_size,
        "n_batches": n_batches,
        "target_device": device,
        "gpu_available": has_gpu,
        "runs": n_runs,
        "throughput_texts_per_s": round(trimmed_mean(run_throughput, trim), 1),
        "throughput_stddev": round(trimmed_std(run_throughput, trim), 1),
        "throughput_runs": [round(v, 1) for v in run_throughput],
    }

    cv = (result["throughput_stddev"] / result["throughput_texts_per_s"] * 100
          if result["throughput_texts_per_s"] > 0 else 0)
    print(f"  -> {result['throughput_texts_per_s']:,.0f} texts/s "
          f"+/-{result['throughput_stddev']:.0f} (CV: {cv:.1f}%)")

    return result


# == Test 4: Index Build Time =================================================

def run_index_build(db_size, cfg, db_emb):
    """
    HNSW index construction benchmark.

    This is the one-time cost a personal PC user actually waits for
    when setting up a local RAG system. Building an HNSW graph requires
    repeated nearest-neighbor searches during insertion, making it
    heavily dependent on L3 cache capacity:
    - Small index (100K): graph fits in L3 -> fast build
    - Large index (1M): graph spills to DRAM -> build slows down
    - More L3 cache -> larger index fits -> less slowdown

    Unlike search QPS (which matters for concurrent users),
    build time is directly felt by every single user.
    """
    n_runs = cfg["build_runs"]
    n_cores = get_cpu_count()
    dim = db_emb.shape[1]

    print(f"\n  DB size: {db_size:,} | cores: {n_cores} | runs: {n_runs}")
    faiss.omp_set_num_threads(n_cores)

    build_times = []

    for run_i in range(n_runs):
        vecs = db_emb[:db_size].copy()
        faiss.normalize_L2(vecs)

        t0 = time.perf_counter()
        idx = faiss.IndexHNSWFlat(dim, cfg["hnsw_m"])
        idx.hnsw.efConstruction = 200
        idx.add(vecs)
        elapsed = time.perf_counter() - t0

        build_times.append(elapsed)
        print(f"    run {run_i + 1}/{n_runs}  {elapsed:.2f}s")

        del idx
        del vecs

        if run_i < n_runs - 1:
            time.sleep(2)

    trim = cfg["trim"]
    result = {
        "db_size": db_size,
        "cores_used": n_cores,
        "hnsw_m": cfg["hnsw_m"],
        "ef_construction": 200,
        "runs": n_runs,
        "build_time_s": round(trimmed_mean(build_times, trim), 2),
        "build_time_stddev_s": round(trimmed_std(build_times, trim), 2),
        "build_time_runs_s": [round(v, 2) for v in build_times],
        "vectors_per_s": round(db_size / trimmed_mean(build_times, trim), 0),
    }

    cv = (result["build_time_stddev_s"] / result["build_time_s"] * 100
          if result["build_time_s"] > 0 else 0)
    print(f"  -> {result['build_time_s']:.2f}s "
          f"+/-{result['build_time_stddev_s']:.2f} (CV: {cv:.1f}%) | "
          f"{result['vectors_per_s']:,.0f} vectors/s")

    return result


# == Main =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="x3d-rag-benchmark - Full CPU Benchmark")
    parser.add_argument("--model", default=DEFAULT["llm_model"])
    parser.add_argument("--db-size", type=int, default=None)
    parser.add_argument("--runs", type=int, default=DEFAULT["runs"])
    parser.add_argument("--batch-queries", type=int,
                        default=DEFAULT["batch_queries"])
    parser.add_argument("--skip-rag", action="store_true",
                        help="skip concurrent RAG test")
    parser.add_argument("--skip-feeding", action="store_true",
                        help="skip data feeding test")
    parser.add_argument("--skip-build", action="store_true",
                        help="skip index build time test")
    parser.add_argument("--quick", action="store_true",
                        help="small DB, fewer runs")
    parser.add_argument("--output", default=None)
    parser.add_argument("--cache-dir", default=DEFAULT["cache_dir"])
    args = parser.parse_args()

    cfg = DEFAULT.copy()
    cfg["llm_model"] = args.model
    cfg["batch_queries"] = args.batch_queries
    cfg["runs"] = max(1, args.runs)
    cfg["cache_dir"] = args.cache_dir

    if args.db_size:
        cfg["db_sizes"] = [args.db_size]
    elif args.quick:
        cfg["db_sizes"] = [100_000]
        cfg["runs"] = min(cfg["runs"], 3)
        cfg["rag_runs"] = 2
        cfg["feeding_runs"] = 2
        cfg["feeding_n_batches"] = 50
        cfg["build_runs"] = 2

    cpu = get_cpu_info()
    gpu = get_gpu_info()
    l3 = get_l3_cache()
    mem = get_memory_info()
    n_cores = get_cpu_count()
    now = datetime.now()

    print("\n" + "=" * 60)
    print("  x3d-rag-benchmark - Full CPU Benchmark")
    print("  github.com/sorrymannn/x3d-rag-benchmark")
    print("=" * 60)
    print(f"  CPU:     {cpu}")
    print(f"  Cores:   {n_cores}")
    print(f"  L3:      {l3}")
    print(f"  Memory:  {mem['total_gb']} GB @ {mem['speed']}")
    print(f"  GPU:     {gpu}")
    print(f"  Runs:    {cfg['runs']} (batch) / "
          f"{cfg['rag_runs']} (RAG) / "
          f"{cfg['feeding_runs']} (feeding) / "
          f"{cfg['build_runs']} (build)")
    print(f"  Time:    {now.strftime('%Y-%m-%d %H:%M:%S')}")

    controls = apply_variance_controls()
    print("=" * 60)

    # Setup
    print("\n[Setup] Loading embedding model & cached embeddings")
    print("-" * 60)
    print("  Loading embedding model...", end=" ", flush=True)
    embed_model = SentenceTransformer(cfg["embed_model"])
    print("done")

    db_emb, q_emb, passages = load_or_build_embeddings(cfg, embed_model)

    cache_dir = Path(cfg["cache_dir"])
    db_emb_path = str(cache_dir / CACHE_DB_FILE)
    q_emb_path = str(cache_dir / CACHE_Q_FILE)
    passages_path = str(cache_dir / CACHE_TEXT_FILE)

    output = {
        "meta": {
            "cpu": cpu,
            "cores": n_cores,
            "l3_cache": l3,
            "memory": mem,
            "gpu": gpu,
            "timestamp": now.isoformat(),
            "config": cfg,
            "benchmark_type": "full-cpu",
            "methodology": {
                "batch_search": "FAISS OMP all-core parallel search",
                "concurrent_rag": "multiprocessing, 1 FAISS thread/worker",
                "data_feeding": "tokenizer + GPU transfer throughput",
                "index_build": "HNSW construction time (one-time setup cost)",
                "trimmed_mean": f"top/bottom {int(cfg['trim']*100)}% dropped",
                "inter_run_cooling": "2-3s",
                "variance_controls": controls,
            },
        },
        "batch_vector_search": [],
        "concurrent_rag": None,
        "data_feeding": None,
        "index_build": [],
    }

    # == Test 1: Batch Vector Search ==========================================
    print(f"\n[1/4] Batch Vector Search (all {n_cores} cores)")
    print("  FAISS HNSW - all OMP threads active")
    print("  CCD-level L3 cache contention visible here")
    print("-" * 60)

    for db_size in cfg["db_sizes"]:
        output["batch_vector_search"].append(
            run_batch_vector_search(db_size, cfg, db_emb, q_emb))

    # == Test 2: Concurrent RAG ===============================================
    if not args.skip_rag:
        print(f"\n[2/4] Concurrent RAG Pipeline")
        print("  Multiple workers: search(CPU) -> LLM(GPU) in parallel")
        print("-" * 60)

        try:
            models = ollama_client.list().models
            has_model = any(cfg["llm_model"] in m.model for m in models)
        except Exception:
            has_model = False

        if has_model:
            output["concurrent_rag"] = run_concurrent_rag(
                cfg, db_emb_path, q_emb_path, passages_path)
        else:
            print(f"  [SKIP] ollama model '{cfg['llm_model']}' not found")
            print(f"  Install: ollama pull {cfg['llm_model']}")
    else:
        print("\n[2/4] Concurrent RAG - skipped (--skip-rag)")

    # == Test 3: Data Feeding =================================================
    if not args.skip_feeding:
        print(f"\n[3/4] Data Feeding Throughput")
        print("  Tokenize (CPU) -> tensor transfer (GPU)")
        print("-" * 60)
        output["data_feeding"] = run_data_feeding(cfg, passages)
    else:
        print("\n[3/4] Data Feeding - skipped (--skip-feeding)")

    # == Test 4: Index Build Time =============================================
    if not args.skip_build:
        print(f"\n[4/4] Index Build Time")
        print("  HNSW construction - one-time setup cost")
        print("  L3 cache capacity determines build speed at scale")
        print("-" * 60)

        for db_size in cfg["db_sizes"]:
            output["index_build"].append(
                run_index_build(db_size, cfg, db_emb))
    else:
        print("\n[4/4] Index Build - skipped (--skip-build)")

    # == Save =================================================================
    restore_after_benchmark()

    out_path = (args.output or
                f"fullcpu_{now.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)

    if output["batch_vector_search"]:
        for r in output["batch_vector_search"]:
            cv = (r["qps_stddev"] / r["qps"] * 100
                  if r["qps"] > 0 else 0)
            print(f"  Batch Search {r['db_size']//1000}K: "
                  f"{r['qps']:>10,.0f} QPS (CV: {cv:.1f}%)")

    if output["concurrent_rag"]:
        r = output["concurrent_rag"]
        print(f"  Concurrent RAG:     "
              f"{r['throughput_qps']:>10.1f} req/s "
              f"({r['n_workers']} workers)")

    if output["data_feeding"]:
        r = output["data_feeding"]
        print(f"  Data Feeding:       "
              f"{r['throughput_texts_per_s']:>10,.0f} texts/s "
              f"(-> {r['target_device']})")

    if output["index_build"]:
        for r in output["index_build"]:
            cv = (r["build_time_stddev_s"] / r["build_time_s"] * 100
                  if r["build_time_s"] > 0 else 0)
            print(f"  Index Build {r['db_size']//1000}K:  "
                  f"{r['build_time_s']:>10.2f}s  "
                  f"({r['vectors_per_s']:,.0f} vec/s, CV: {cv:.1f}%)")

    print(f"\n  Saved: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
