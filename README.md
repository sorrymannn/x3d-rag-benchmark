# x3d-rag-benchmark

**AMD Ryzen X3D V-Cache vs non-X3D — RAG AI 파이프라인 CPU 성능 벤치마크**

AMD 슬라이드에서 "CPU가 RAG에서 담당하는 작업은 Latency-Sensitive + Random Memory Access Pattern"
이라고 명시했지만, 실제로 이를 측정한 공개 벤치마크는 없었습니다.

이 프로젝트는 **GPU를 함께 사용하는 실제 AI PC 환경**에서 CPU(X3D vs non-X3D)가
RAG 파이프라인에 미치는 영향을 측정합니다.

---

## 왜 X3D가 RAG에서 유리한가

```
RAG 파이프라인에서 CPU가 하는 일:
  Tokenize / Detokenize       → 레이턴시 민감
  RAG Vector Searching        → 랜덤 메모리 접근 ← X3D 핵심 강점
  GPU data feeding            → 레이턴시 민감

X3D V-Cache (96MB L3)의 강점:
  Random Memory Access Pattern = 게임에서 X3D가 강한 이유와 동일한 메커니즘
  HNSW 그래프 탐색 = 랜덤 노드 접근의 연속 → 캐시 히트율이 성능을 결정
```

---

## 측정 항목

| 항목 | 설명 | X3D 영향 |
|---|---|---|
| Vector Search QPS | FAISS HNSW 초당 쿼리 수 | **직접적** |
| Vector Search P99 Latency | 최악의 경우 검색 레이턴시 | **직접적** |
| Concurrent Search | 동시 요청 처리 성능 | **직접적** |
| RAG TTFT | 첫 토큰까지 전체 시간 | 간접적 |

---

## 설치

```bash
git clone https://github.com/YOUR_ID/x3d-rag-benchmark
cd x3d-rag-benchmark
pip install -r requirements.txt

# LLM 모델 (RAG TTFT 측정 시)
ollama pull llama3.2
```

---

## 실행

```bash
# 전체 벤치마크
python3 benchmark.py

# Vector Search만 (ollama 불필요)
python3 benchmark.py --skip-rag

# 빠른 테스트
python3 benchmark.py --quick --skip-rag

# 결과 저장
python3 benchmark.py --output 9700x.json
python3 benchmark.py --output 9800x3d.json
```

---

## 결과 비교

```bash
python3 compare.py 9700x.json 9800x3d.json
# → comparison.png 생성
```

---

## 사용 라이브러리

| 라이브러리 | 출처 | 용도 |
|---|---|---|
| FAISS | Meta AI | 벡터 검색 엔진 |
| sentence-transformers | HuggingFace | 임베딩 모델 |
| ollama | Ollama | 로컬 LLM 서버 |
| datasets | HuggingFace | 공개 데이터셋 |

---

## 측정 환경 조건

재현 가능한 결과를 위해:
- 백그라운드 프로세스 최소화
- 벤치마크 전 시스템 재시작 권장
- 동일한 RAM 용량 / 속도 사용
- 동일한 GPU 사용

---

## 기여

PR과 이슈 환영합니다.
다른 CPU로 측정한 결과는 `results/` 폴더에 PR 주시면 추가합니다.
