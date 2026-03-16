# x3d-rag-benchmark

**AMD Ryzen X3D V-Cache vs non-X3D — RAG AI 파이프라인 CPU 성능 벤치마크**

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![FAISS](https://img.shields.io/badge/Meta-FAISS-blue.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-sentence--transformers-orange.svg)

GPU를 함께 사용하는 실제 AI PC 환경에서 CPU(X3D vs non-X3D)가
RAG 파이프라인에 미치는 영향을 측정하는 오픈소스 벤치마크입니다.

---

## 왜 X3D가 RAG에서 유리한가

RAG 파이프라인에서 CPU가 담당하는 Vector Search는
Random Memory Access Pattern이 핵심 병목입니다.
X3D V-Cache의 대용량 L3 캐시(96MB)는 이 workload에서
게임과 동일한 메커니즘으로 성능 이점을 제공합니다.

```
RAG 파이프라인 구조:

  사용자 질문
      ↓
  임베딩 생성        (GPU)
      ↓
  Vector Search      (CPU) ← X3D V-Cache 효과 발생 구간
      ↓
  LLM 생성           (GPU)
      ↓
  응답

Vector Search 특성:
  - HNSW 그래프를 랜덤하게 탐색
  - 매 탐색마다 다른 메모리 주소 접근 (Random Access)
  - L3 캐시가 클수록 캐시 히트율 증가 → 레이턴시 감소
  - 96MB V-Cache = 게임에서 X3D가 강한 이유와 동일한 메커니즘
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

### 1. 레포 클론
```bash
git clone https://github.com/sorrymannn/x3d-rag-benchmark
cd x3d-rag-benchmark
```

### 2. Python 패키지 설치
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. ollama 설치 및 모델 다운로드
```bash
# ollama 설치 (없는 경우)
curl -fsSL https://ollama.com/install.sh | sh

# LLM 모델 다운로드
ollama pull llama3.2
```

> RAG TTFT 측정이 필요 없다면 ollama 없이도 실행 가능합니다.

---

## 실행

```bash
# 전체 벤치마크 (Vector Search + RAG TTFT)
python3 benchmark.py

# Vector Search만 (ollama 불필요, 약 30~45분)
python3 benchmark.py --skip-rag

# 빠른 테스트 (약 3분)
python3 benchmark.py --quick --skip-rag

# 결과 파일 지정
python3 benchmark.py --output 9700x.json
python3 benchmark.py --output 9800x3d.json
```

---

## 결과 비교 그래프

두 CPU의 결과 JSON을 비교해서 그래프로 출력합니다.

```bash
python3 compare.py 9700x.json 9800x3d.json
# → comparison.png 자동 생성
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

## 재현 가능한 결과를 위한 측정 조건

- 백그라운드 프로세스 최소화
- 벤치마크 전 시스템 재시작 권장
- 동일한 RAM 용량 / 속도 사용
- 동일한 GPU 사용
- 동일한 운영체제 환경

---

## 기여

PR과 이슈 환영합니다.
다른 CPU로 측정한 결과는 `results/` 폴더에 PR 주시면 추가합니다.
