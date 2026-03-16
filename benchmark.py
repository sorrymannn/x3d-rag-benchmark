"""
x3d-rag-benchmark
=================
AMD Ryzen X3D V-Cache vs non-X3D CPU 성능 비교
RAG 파이프라인에서 CPU가 담당하는 작업의 실측 벤치마크

측정 항목:
  1. Vector Search Latency  - FAISS HNSW 검색 레이턴시
  2. RAG TTFT               - 전체 파이프라인 첫 토큰까지 시간
  3. Concurrent RAG         - 동시 요청 처리 성능

사용 라이브러리:
  - FAISS     (Meta)         : 벡터 검색 엔진
  - sentence-transformers    : 임베딩 모델 (HuggingFace)
  - ollama                   : 로컬 LLM 서버
  - datasets  (HuggingFace)  : 공개 데이터셋

설치:
  pip install faiss-cpu sentence-transformers ollama datasets numpy tqdm

실행:
  python3 benchmark.py
  python3 benchmark.py --model llama3.2 --db-size 500000
  python3 benchmark.py --skip-rag   # Vector Search만 측정
"""

import argparse
import json
import os
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np

# ── 의존성 체크 ────────────────────────────────────────────────────────────────

def check_deps():
    missing = []
    for pkg in ["faiss", "sentence_transformers", "ollama", "datasets", "tqdm"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg.replace("_", "-"))
    if missing:
        print(f"[오류] 누락된 패키지: {', '.join(missing)}")
        print(f"설치: pip install {' '.join(missing)}")
        exit(1)

check_deps()

import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import ollama as ollama_client

# ── 설정 ──────────────────────────────────────────────────────────────────────

DEFAULT = {
    "embed_model":   "all-MiniLM-L6-v2",   # HuggingFace 임베딩 모델
    "llm_model":     "llama3.2",            # ollama 모델
    "db_sizes":      [100_000, 500_000, 1_000_000],
    "n_queries":     200,
    "top_k":         10,
    "hnsw_m":        32,
    "hnsw_ef":       64,
    "concurrent":    [1, 2, 4, 8],          # 동시 요청 수
    "warmup":        20,
    "rag_queries":   50,                    # RAG TTFT 측정 쿼리 수
}

# ── 유틸 ──────────────────────────────────────────────────────────────────────

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
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"], text=True
        ).strip()
        return out
    except Exception:
        return "N/A"

def percentile(data, p):
    s = sorted(data)
    return s[int(len(s) * p / 100)]

def ms(t): return round(t * 1000, 3)

# ── 1단계: Vector Search Benchmark ────────────────────────────────────────────

def run_vector_search(db_size, cfg):
    """FAISS HNSW 벡터 검색 벤치마크 — X3D V-Cache 효과가 가장 잘 나오는 구간"""
    dim = 384  # all-MiniLM-L6-v2 출력 차원

    print(f"\n  DB 크기: {db_size:,} 벡터")
    print(f"  예상 메모리: ~{db_size * dim * 4 / 1024**3:.2f} GB")

    # 벡터 생성
    print("  벡터 DB 생성 중...", end=" ", flush=True)
    db_vecs = np.random.rand(db_size, dim).astype(np.float32)
    faiss.normalize_L2(db_vecs)
    q_vecs = np.random.rand(cfg["n_queries"] + cfg["warmup"], dim).astype(np.float32)
    faiss.normalize_L2(q_vecs)
    print("완료")

    # HNSW 인덱스 빌드
    print("  HNSW 인덱스 빌드 중...", end=" ", flush=True)
    t0 = time.perf_counter()
    idx = faiss.IndexHNSWFlat(dim, cfg["hnsw_m"])
    idx.hnsw.efConstruction = 200
    idx.add(db_vecs)
    build_sec = time.perf_counter() - t0
    print(f"{build_sec:.1f}s")

    idx.hnsw.efSearch = cfg["hnsw_ef"]

    # 워밍업
    for q in q_vecs[:cfg["warmup"]]:
        idx.search(q.reshape(1, -1), cfg["top_k"])

    # 본 측정
    bench_q = q_vecs[cfg["warmup"]:]
    latencies = []
    t_total = time.perf_counter()
    for q in bench_q:
        t0 = time.perf_counter()
        idx.search(q.reshape(1, -1), cfg["top_k"])
        latencies.append(time.perf_counter() - t0)
    total_sec = time.perf_counter() - t_total

    qps = len(latencies) / total_sec
    result = {
        "db_size":        db_size,
        "build_time_s":   round(build_sec, 2),
        "qps":            round(qps, 1),
        "latency_mean_ms": ms(np.mean(latencies)),
        "latency_p50_ms":  ms(percentile(latencies, 50)),
        "latency_p95_ms":  ms(percentile(latencies, 95)),
        "latency_p99_ms":  ms(percentile(latencies, 99)),
    }

    print(f"  QPS: {qps:.1f}  |  P50: {ms(percentile(latencies,50)):.3f}ms  "
          f"|  P99: {ms(percentile(latencies,99)):.3f}ms")

    del db_vecs, idx
    return result


def run_concurrent_search(db_size, cfg):
    """동시 요청 벤치마크 — 멀티쿼리 시나리오"""
    dim = 384
    db_vecs = np.random.rand(db_size, dim).astype(np.float32)
    faiss.normalize_L2(db_vecs)

    idx = faiss.IndexHNSWFlat(dim, cfg["hnsw_m"])
    idx.hnsw.efConstruction = 200
    idx.add(db_vecs)
    idx.hnsw.efSearch = cfg["hnsw_ef"]

    results = {}
    for n_concurrent in cfg["concurrent"]:
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

        results[n_concurrent] = {
            "qps":            round(len(lats) / total, 1),
            "latency_p50_ms": ms(percentile(lats, 50)),
            "latency_p99_ms": ms(percentile(lats, 99)),
        }
        print(f"  동시 {n_concurrent:>2}req  QPS: {results[n_concurrent]['qps']:>8.1f}  "
              f"P99: {results[n_concurrent]['latency_p99_ms']:.3f}ms")

    del db_vecs, idx
    return results


# ── 2단계: RAG TTFT Benchmark ──────────────────────────────────────────────────

def check_ollama_model(model):
    try:
        models = ollama_client.list()
        names = [m.model for m in models.models]
        return any(model in n for n in names)
    except Exception:
        return False


def run_rag_ttft(cfg, embed_model):
    """
    실제 RAG 파이프라인 TTFT 측정
    임베딩(GPU/CPU) → FAISS 검색(CPU) → ollama LLM(GPU)
    """
    print(f"\n  임베딩 모델: {cfg['embed_model']}")
    print(f"  LLM 모델:   {cfg['llm_model']}")

    if not check_ollama_model(cfg["llm_model"]):
        print(f"  [경고] ollama에 {cfg['llm_model']} 모델이 없습니다.")
        print(f"  설치: ollama pull {cfg['llm_model']}")
        return None

    # 공개 데이터셋 로드 (Wikipedia 일부)
    print("  데이터셋 로드 중...", end=" ", flush=True)
    try:
        ds = load_dataset("wikipedia", "20220301.simple",
                          split="train[:2000]", trust_remote_code=True)
        passages = [r["text"][:300] for r in ds]
    except Exception:
        # fallback: 랜덤 텍스트
        passages = [f"This is passage number {i} about topic {i % 50}."
                    for i in range(2000)]
    print(f"{len(passages)}개 문서")

    # 임베딩 생성
    print("  임베딩 생성 중...", end=" ", flush=True)
    t0 = time.perf_counter()
    embeddings = embed_model.encode(passages, batch_size=64,
                                    show_progress_bar=False,
                                    convert_to_numpy=True)
    embed_sec = time.perf_counter() - t0
    print(f"{embed_sec:.1f}s")

    # FAISS 인덱스 빌드
    dim = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    idx = faiss.IndexHNSWFlat(dim, 32)
    idx.hnsw.efConstruction = 200
    idx.add(embeddings)
    idx.hnsw.efSearch = 64

    # 테스트 쿼리
    queries = [
        "What is machine learning?",
        "How does the human brain work?",
        "What is the history of the internet?",
        "Explain quantum computing",
        "What causes climate change?",
    ] * (cfg["rag_queries"] // 5 + 1)
    queries = queries[:cfg["rag_queries"]]

    # 워밍업
    for q in queries[:5]:
        q_emb = embed_model.encode([q], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        idx.search(q_emb, cfg["top_k"])

    # 본 측정
    ttft_list = []
    search_lat_list = []

    print(f"  RAG TTFT 측정 중 ({len(queries)}회)...")
    for query in tqdm(queries, ncols=60):
        # 1. 쿼리 임베딩
        q_emb = embed_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)

        # 2. Vector Search (CPU — 측정 핵심)
        t_search = time.perf_counter()
        D, I = idx.search(q_emb, cfg["top_k"])
        search_lat = time.perf_counter() - t_search
        search_lat_list.append(search_lat)

        # 3. Context 구성
        context = " ".join([passages[i] for i in I[0] if i < len(passages)])[:500]
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer briefly:"

        # 4. LLM TTFT (첫 토큰까지)
        t_start = time.perf_counter()
        try:
            stream = ollama_client.chat(
                model=cfg["llm_model"],
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            first_token = False
            for chunk in stream:
                if not first_token:
                    ttft = time.perf_counter() - t_start
                    ttft_list.append(ttft)
                    first_token = True
                    break
        except Exception as e:
            print(f"\n  [LLM 오류] {e}")
            break

    if not ttft_list:
        return None

    result = {
        "n_passages":         len(passages),
        "embed_time_s":       round(embed_sec, 2),
        "vector_search": {
            "mean_ms": ms(np.mean(search_lat_list)),
            "p50_ms":  ms(percentile(search_lat_list, 50)),
            "p95_ms":  ms(percentile(search_lat_list, 95)),
            "p99_ms":  ms(percentile(search_lat_list, 99)),
        },
        "ttft": {
            "mean_ms": ms(np.mean(ttft_list)),
            "p50_ms":  ms(percentile(ttft_list, 50)),
            "p95_ms":  ms(percentile(ttft_list, 95)),
            "p99_ms":  ms(percentile(ttft_list, 99)),
        },
    }

    print(f"\n  [Vector Search]  P50: {result['vector_search']['p50_ms']:.3f}ms  "
          f"P99: {result['vector_search']['p99_ms']:.3f}ms")
    print(f"  [RAG TTFT]       P50: {result['ttft']['p50_ms']:.1f}ms  "
          f"P99: {result['ttft']['p99_ms']:.1f}ms")

    return result


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="x3d-rag-benchmark")
    parser.add_argument("--model",    default=DEFAULT["llm_model"],
                        help=f"ollama LLM 모델 (기본: {DEFAULT['llm_model']})")
    parser.add_argument("--db-size",  type=int, default=None,
                        help="단일 DB 크기 지정")
    parser.add_argument("--queries",  type=int, default=DEFAULT["n_queries"],
                        help=f"쿼리 반복 수 (기본: {DEFAULT['n_queries']})")
    parser.add_argument("--skip-rag", action="store_true",
                        help="RAG TTFT 측정 건너뜀 (Vector Search만)")
    parser.add_argument("--quick",    action="store_true",
                        help="빠른 테스트 (작은 DB만)")
    parser.add_argument("--output",   default=None,
                        help="결과 저장 경로 (기본: 자동 생성)")
    args = parser.parse_args()

    cfg = DEFAULT.copy()
    cfg["llm_model"] = args.model
    cfg["n_queries"] = args.queries

    if args.db_size:
        cfg["db_sizes"] = [args.db_size]
    elif args.quick:
        cfg["db_sizes"] = [100_000]

    cpu  = get_cpu_info()
    gpu  = get_gpu_info()
    now  = datetime.now()

    print("\n" + "="*60)
    print("  x3d-rag-benchmark")
    print("  github.com/YOUR_ID/x3d-rag-benchmark")
    print("="*60)
    print(f"  CPU:  {cpu}")
    print(f"  GPU:  {gpu}")
    print(f"  시각: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    output = {
        "meta": {
            "cpu":       cpu,
            "gpu":       gpu,
            "timestamp": now.isoformat(),
            "config":    cfg,
        },
        "vector_search":    [],
        "concurrent_search": {},
        "rag_ttft":         None,
    }

    # ── 1. Vector Search ──
    print("\n[1/3] Vector Search Benchmark (FAISS HNSW)")
    print("      RAG에서 CPU가 담당하는 핵심 작업")
    print("-"*60)
    for db_size in cfg["db_sizes"]:
        r = run_vector_search(db_size, cfg)
        output["vector_search"].append(r)

    # ── 2. Concurrent Search ──
    print("\n[2/3] Concurrent Search Benchmark")
    print("      동시 요청 증가 시 X3D 캐시 효과 측정")
    print("-"*60)
    mid_size = cfg["db_sizes"][len(cfg["db_sizes"])//2]
    print(f"  DB 크기: {mid_size:,} 벡터")
    output["concurrent_search"] = run_concurrent_search(mid_size, cfg)

    # ── 3. RAG TTFT ──
    if not args.skip_rag:
        print("\n[3/3] RAG End-to-End TTFT Benchmark")
        print("      임베딩(GPU) → 검색(CPU) → LLM(GPU)")
        print("-"*60)
        print("  임베딩 모델 로딩 중...", end=" ", flush=True)
        embed_model = SentenceTransformer(cfg["embed_model"])
        print("완료")
        output["rag_ttft"] = run_rag_ttft(cfg, embed_model)
    else:
        print("\n[3/3] RAG TTFT — 건너뜀 (--skip-rag)")

    # ── 결과 저장 ──
    out_path = args.output or f"result_{now.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print(f"  완료! 결과: {out_path}")
    print(f"  비교: python3 compare.py 9700x.json 9800x3d.json")
    print("="*60)


if __name__ == "__main__":
    main()
