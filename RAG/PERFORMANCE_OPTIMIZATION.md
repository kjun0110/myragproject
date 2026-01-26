# 성능 최적화 전략

## 현재 상황 분석

### 발견된 성능 병목 지점

1. **모델 로딩**
   - 여러 곳에서 중복 로딩 가능성
   - Lazy loading은 있지만 최적화 부족
   - 전역 캐싱만 있고 메모리 관리 부족

2. **추론 속도**
   - 배치 처리 미적용
   - KV 캐시 미활용
   - 양자화 설정 최적화 필요

3. **RAG 체인**
   - 벡터 검색 최적화 필요
   - 문서 임베딩 캐싱 부족

---

## 최적화 전략 (우선순위별)

### 🔥 1단계: 즉시 적용 가능 (High Impact, Low Effort)

#### 1.1 모델 프리로딩 및 싱글톤 강화

**현재 문제:**
- 모델이 첫 요청 시 로드되어 지연 발생
- 여러 요청이 동시에 모델 로딩 시도 가능

**해결책:**
```python
# app/domains/v1/chat/agents/model_loader.py 개선
import threading
from functools import lru_cache

_model_lock = threading.Lock()

@lru_cache(maxsize=1)
def load_exaone_model_for_service(...):
    """싱글톤 + 스레드 안전 + LRU 캐시"""
    global _exaone_llm
    if _exaone_llm is not None:
        return _exaone_llm
    
    with _model_lock:
        if _exaone_llm is None:
            # 로딩 로직
            ...
    return _exaone_llm
```

**예상 효과:** 첫 요청 지연 제거, 중복 로딩 방지

---

#### 1.2 애플리케이션 시작 시 모델 프리로딩

**현재:** Lazy loading으로 첫 요청 시 느림

**해결책:**
```python
# app/main.py 또는 startup 이벤트
@app.on_event("startup")
async def preload_models():
    """서버 시작 시 모델 미리 로드"""
    import asyncio
    from app.domains.v1.chat.agents.model_loader import load_exaone_model_for_service
    
    # 백그라운드에서 비동기 로딩
    asyncio.create_task(
        asyncio.to_thread(load_exaone_model_for_service)
    )
```

**예상 효과:** 첫 요청 지연 0초 (이미 로드됨)

---

#### 1.3 KV 캐시 활성화

**현재:** 매번 전체 컨텍스트 재계산

**해결책:**
```python
# model_loader.py
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device_map,
    trust_remote_code=trust_remote_code,
    use_cache=True,  # KV 캐시 활성화
)

# 추론 시
outputs = model.generate(
    **inputs,
    past_key_values=past_key_values,  # 이전 대화 캐시 재사용
    use_cache=True,
)
```

**예상 효과:** 대화 연속성 시 30-50% 속도 향상

---

### ⚡ 2단계: 중기 최적화 (High Impact, Medium Effort)

#### 2.1 배치 처리 구현

**현재:** 요청별 개별 처리

**해결책:**
```python
# app/domains/v1/chat/agents/chat_service.py
from collections import deque
import asyncio

class BatchProcessor:
    def __init__(self, batch_size=4, max_wait=0.1):
        self.batch_size = batch_size
        self.max_wait = max_wait
        self.queue = deque()
        self.lock = asyncio.Lock()
    
    async def process(self, inputs):
        """배치로 묶어서 처리"""
        async with self.lock:
            self.queue.append(inputs)
            if len(self.queue) >= self.batch_size:
                batch = [self.queue.popleft() for _ in range(self.batch_size)]
                return await self._process_batch(batch)
        
        # 타임아웃 대기
        await asyncio.sleep(self.max_wait)
        # 배치 처리
```

**예상 효과:** 동시 요청 시 2-4배 처리량 증가

---

#### 2.2 양자화 최적화

**현재:** 4-bit 양자화 사용 중

**최적화 옵션:**
```python
# 더 빠른 양자화 설정
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # bfloat16 → float16 (더 빠름)
    bnb_4bit_use_double_quant=True,
)

# 또는 8-bit로 변경 (속도 vs 메모리 트레이드오프)
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # 4-bit보다 빠름
)
```

**예상 효과:** 추론 속도 20-30% 향상

---

#### 2.3 Flash Attention 2 사용

**현재:** 표준 Attention 사용

**해결책:**
```python
# model_loader.py
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2",  # Flash Attention 2
    device_map=device_map,
    trust_remote_code=trust_remote_code,
)
```

**필요 패키지:**
```bash
pip install flash-attn --no-build-isolation
```

**예상 효과:** 긴 컨텍스트에서 2-3배 속도 향상

---

#### 2.4 텍스트 생성 최적화

**현재:** 기본 생성 설정

**최적화:**
```python
# 더 빠른 생성 파라미터
outputs = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    do_sample=False,  # Greedy decoding (더 빠름)
    num_beams=1,  # Beam search 비활성화
    use_cache=True,
    pad_token_id=tokenizer.eos_token_id,
)
```

**예상 효과:** 생성 속도 20-40% 향상

---

### 🚀 3단계: 장기 최적화 (High Impact, High Effort)

#### 3.1 모델 서빙 프레임워크 도입

**옵션 1: vLLM**
```python
# vLLM은 배치 처리, KV 캐시 최적화 자동 제공
from vllm import LLM

llm = LLM(
    model="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
    quantization="awq",  # 또는 "gptq"
    max_model_len=32768,
    gpu_memory_utilization=0.9,
)
```

**옵션 2: TensorRT-LLM**
- NVIDIA GPU 전용
- 최고 성능 (2-5배 향상)

**예상 효과:** 전체 처리량 3-5배 증가

---

#### 3.2 비동기 추론 파이프라인

**현재:** 동기 처리

**해결책:**
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncModelInference:
    def __init__(self, model, tokenizer, max_workers=2):
        self.model = model
        self.tokenizer = tokenizer
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def generate_async(self, prompt, **kwargs):
        """비동기 추론"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._generate_sync,
            prompt,
            **kwargs
        )
```

**예상 효과:** 동시 요청 처리 능력 향상

---

#### 3.3 벡터 검색 최적화

**현재:** PGVector 사용

**최적화:**
```python
# HNSW 인덱스 사용 (더 빠른 검색)
vector_store = PGVector(
    connection_string=connection_string,
    collection_name=collection_name,
    embedding_function=embeddings,
    use_jsonb=True,
    pre_delete_collection=False,
)

# 인덱스 생성
CREATE INDEX ON vector_store USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

**예상 효과:** 검색 속도 5-10배 향상

---

#### 3.4 문서 임베딩 캐싱

**현재:** 매번 재임베딩

**해결책:**
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=10000)
def get_cached_embedding(text: str, model_name: str):
    """임베딩 캐싱"""
    return embeddings.embed_query(text)

# 또는 Redis 캐시
import redis
r = redis.Redis()

def get_embedding(text):
    key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
    cached = r.get(key)
    if cached:
        return pickle.loads(cached)
    embedding = embeddings.embed_query(text)
    r.setex(key, 3600, pickle.dumps(embedding))  # 1시간 캐시
    return embedding
```

**예상 효과:** 반복 문서 검색 시 10-100배 빠름

---

## 구현 우선순위

### Phase 1 (1주일 내)
1. ✅ 모델 프리로딩 (startup 이벤트)
2. ✅ KV 캐시 활성화
3. ✅ 싱글톤 + 스레드 안전 강화
4. ✅ 생성 파라미터 최적화

**예상 효과:** 첫 요청 지연 제거, 20-30% 속도 향상

### Phase 2 (2-3주 내)
1. ⚡ Flash Attention 2
2. ⚡ 양자화 최적화 (float16)
3. ⚡ 배치 처리 기본 구현
4. ⚡ 벡터 검색 인덱스 최적화

**예상 효과:** 추가 30-50% 속도 향상

### Phase 3 (1-2개월 내)
1. 🚀 vLLM 또는 TensorRT-LLM 도입
2. 🚀 비동기 추론 파이프라인
3. 🚀 문서 임베딩 캐싱 (Redis)
4. 🚀 모니터링 및 프로파일링

**예상 효과:** 전체 3-5배 성능 향상

---

## 모니터링 지표

### 측정할 메트릭

1. **모델 로딩 시간**
   - 첫 로딩: 목표 < 30초
   - 캐시된 로딩: 목표 < 1초

2. **추론 속도**
   - 토큰/초: 목표 > 20 tokens/s
   - 첫 토큰 지연: 목표 < 500ms
   - 전체 생성 시간: 목표 < 5초 (100 토큰 기준)

3. **동시 처리**
   - 동시 요청 처리 수: 목표 > 4
   - 평균 응답 시간: 목표 < 3초

4. **메모리 사용량**
   - GPU 메모리: 목표 < 80%
   - 시스템 메모리: 목표 < 70%

---

## 하드웨어 최적화

### GPU 설정
```python
# CUDA 최적화
import torch
torch.backends.cudnn.benchmark = True  # cuDNN 자동 튜닝
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 활성화
```

### 환경 변수
```bash
# .env
CUDA_LAUNCH_BLOCKING=0  # 비동기 실행
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # 메모리 단편화 방지
```

---

## 벤치마크 테스트

### 테스트 시나리오

```python
# benchmarks/test_performance.py
import time
import asyncio

async def benchmark_model():
    """성능 벤치마크"""
    from app.common.loaders import load_exaone_with_spam_adapter
    
    # 로딩 시간
    start = time.time()
    model, tokenizer = load_exaone_with_spam_adapter()
    load_time = time.time() - start
    print(f"로딩 시간: {load_time:.2f}초")
    
    # 추론 속도
    prompt = "안녕하세요" * 10
    start = time.time()
    outputs = model.generate(**tokenizer(prompt, return_tensors="pt"))
    inference_time = time.time() - start
    tokens = len(outputs[0])
    print(f"추론 속도: {tokens/inference_time:.2f} tokens/s")
    
    # 동시 요청
    async def concurrent_request(i):
        start = time.time()
        outputs = model.generate(**tokenizer(f"질문 {i}", return_tensors="pt"))
        return time.time() - start
    
    times = await asyncio.gather(*[concurrent_request(i) for i in range(5)])
    print(f"동시 요청 평균: {sum(times)/len(times):.2f}초")
```

---

## 예상 성능 개선

| 단계 | 현재 | Phase 1 | Phase 2 | Phase 3 |
|------|------|---------|---------|---------|
| 첫 요청 지연 | 30-60초 | 0초 | 0초 | 0초 |
| 추론 속도 | 10-15 tokens/s | 15-20 tokens/s | 25-35 tokens/s | 50-100 tokens/s |
| 동시 처리 | 1-2 요청 | 2-4 요청 | 4-8 요청 | 10-20 요청 |
| 메모리 사용 | 100% | 90% | 85% | 80% |

---

## 다음 단계

1. **즉시 시작:** Phase 1 구현 (프리로딩, KV 캐시)
2. **모니터링 설정:** 성능 메트릭 수집 시작
3. **점진적 개선:** Phase 2, 3 순차 구현
4. **지속적 최적화:** 벤치마크 기반 튜닝
