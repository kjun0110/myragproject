# 채팅 스트리밍 구현 전략 문서 (Chat Stream Strategy)

## 📋 목차
1. [개요](#개요)
2. [아키텍처 개요](#아키텍처-개요)
3. [백엔드 스트리밍 전략](#백엔드-스트리밍-전략)
4. [프론트엔드 스트리밍 처리](#프론트엔드-스트리밍-처리)
5. [토큰 단위 증분 업데이트](#토큰-단위-증분-업데이트)
6. [Next.js API 라우트 프록시 패턴](#nextjs-api-라우트-프록시-패턴)
7. [구현 세부사항](#구현-세부사항)
8. [성능 최적화](#성능-최적화)
9. [문제 해결 및 트러블슈팅](#문제-해결-및-트러블슈팅)

---

## 개요

본 문서는 LangChain/LangGraph 기반 챗봇 애플리케이션에서 **실시간 토큰 단위 스트리밍**을 구현하기 위한 전략을 설명합니다.

### 핵심 목표
- ✅ **토큰 단위 실시간 스트리밍**: LLM이 생성하는 각 토큰을 즉시 프론트엔드에 전달
- ✅ **증분 업데이트**: 전체 메시지가 아닌 새로 생성된 토큰(delta)만 전송
- ✅ **깔끔한 응답**: 시스템 프롬프트, 내부 태그, 과거 대화 내용 제거
- ✅ **부드러운 UX**: 한 글자씩 타이핑되는 듯한 자연스러운 사용자 경험

---

## 아키텍처 개요

```
┌─────────────────┐
│   Frontend      │
│  (Next.js)      │
│  page.tsx       │
└────────┬────────┘
         │ HTTP POST
         │ /api/chat or /api/graph
         ▼
┌─────────────────┐
│  Next.js API    │
│  Route Proxy    │
│  route.ts       │
└────────┬────────┘
         │ HTTP POST
         │ text/plain stream
         ▼
┌─────────────────┐
│   Backend       │
│  (FastAPI)      │
│  router.py     │
└────────┬────────┘
         │
         ├─► LangGraph (astream_events)
         │   └─► on_chat_model_stream
         │
         └─► RAG Chain (astream)
             └─► answer chunks
```

### 데이터 흐름
1. **프론트엔드** → Next.js API 라우트로 요청
2. **Next.js API 라우트** → FastAPI 백엔드로 프록시
3. **FastAPI 백엔드** → LangGraph/RAG Chain에서 스트리밍
4. **스트림** → 토큰 단위로 `text/plain` 전송
5. **프론트엔드** → `ReadableStream`으로 수신하여 실시간 렌더링

---

## 백엔드 스트리밍 전략

### 1. LangGraph 스트리밍 (`graph_router.py`)

#### 전략: `astream_events` 사용
LangGraph의 `astream_events` API를 사용하여 **이벤트 기반 토큰 스트리밍**을 구현합니다.

```python
async def stream_generator():
    """스트리밍 제너레이터 - 토큰 단위 증분 업데이트."""
    # astream_events를 사용하여 토큰 단위 스트리밍
    async for event in graph.astream_events(state, version="v1"):
        event_type = event.get("event")

        # on_chat_model_stream 이벤트만 처리 (토큰 단위)
        if event_type == "on_chat_model_stream":
            chunk_data = event.get("data", {})
            chunk = chunk_data.get("chunk")

            if chunk and hasattr(chunk, "content"):
                content = chunk.content
                if content:
                    # 태그 제거 (실시간으로)
                    if not any(tag in content for tag in ["[[system]]", "[[endofturn]]"]):
                        yield content  # 순수 텍스트만 yield
                        await asyncio.sleep(0.01)  # 10ms 지연
```

#### 핵심 포인트
- ✅ **이벤트 필터링**: `on_chat_model_stream` 이벤트만 처리
- ✅ **토큰 단위 전송**: `chunk.content`만 yield (전체 메시지 아님)
- ✅ **실시간 필터링**: 내부 태그를 스트리밍 중에 제거
- ✅ **지연 제어**: `asyncio.sleep(0.01)`로 부드러운 스트리밍 효과

#### FastAPI 응답
```python
return StreamingResponse(
    stream_generator(),
    media_type="text/plain; charset=utf-8",  # JSON이 아닌 순수 텍스트
)
```

---

### 2. RAG Chain 스트리밍 (`chat_router.py`)

#### 전략: `astream()` 사용
LangChain RAG Chain의 `astream()` 메서드를 사용하여 **증분 업데이트**를 구현합니다.

```python
async def stream_response():
    accumulated_text = ""

    async for chunk in current_rag_chain.astream({
        "input": request.message,
        "chat_history": chat_history,
    }):
        # chunk에서 answer 추출
        if isinstance(chunk, dict):
            answer = chunk.get("answer", "")
            if answer:
                # 새로 추가된 부분만 추출 (증분 업데이트)
                if len(answer) > len(accumulated_text):
                    delta = answer[len(accumulated_text):]
                    accumulated_text = answer

                    # 한 글자씩 스트리밍
                    for char in delta:
                        yield char
                        await asyncio.sleep(0.01)
```

#### 핵심 포인트
- ✅ **증분 추출**: `delta = answer[len(accumulated_text):]`로 새로 추가된 부분만 추출
- ✅ **문자 단위 스트리밍**: 한 글자씩 yield하여 타이핑 효과 구현
- ✅ **누적 추적**: `accumulated_text`로 전체 길이 추적

---

## 프론트엔드 스트리밍 처리

### 1. Next.js API 라우트 프록시 (`api/chat/route.ts`, `api/graph/route.ts`)

#### 전략: 직접 스트림 전달
백엔드에서 받은 `text/plain` 스트림을 **그대로 프론트엔드에 전달**합니다.

```typescript
// Content-Type 확인
const contentType = response.headers.get("content-type");

if (contentType && contentType.includes("text/plain")) {
  // 백엔드의 text/plain 스트림을 그대로 프론트엔드에 전달
  return new Response(response.body, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    },
  });
}
```

#### 핵심 포인트
- ✅ **스트림 직접 전달**: JSON 변환 없이 `response.body`를 그대로 전달
- ✅ **Content-Type 확인**: `text/plain`일 때만 스트리밍 처리
- ✅ **헤더 설정**: `Cache-Control: no-cache`, `Connection: keep-alive`

#### ❌ 피해야 할 실수
```typescript
// 잘못된 예: JSON으로 감싸기
controller.enqueue(
  new TextEncoder().encode(
    JSON.stringify({ delta: chunk }) + '\n'  // ❌ 이렇게 하면 프론트엔드에 JSON 문자열이 표시됨
  )
);
```

---

### 2. 프론트엔드 컴포넌트 (`page.tsx`)

#### 전략: ReadableStream 처리
`ReadableStream`을 사용하여 **실시간으로 메시지 상태 업데이트**합니다.

```typescript
// 스트리밍 응답 처리
if (isStreaming || modelType === "graph") {
  // 스트리밍 메시지 생성 (초기 상태)
  const streamingMessageId = (Date.now() + 1).toString();
  const streamingMessage: Message = {
    id: streamingMessageId,
    role: "assistant",
    content: "",  // 빈 문자열로 시작
    timestamp: new Date(),
  };
  setMessages((prev) => [...prev, streamingMessage]);

  // ReadableStream 처리
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let accumulatedText = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    // 청크 디코딩 (순수 텍스트)
    const chunk = decoder.decode(value, { stream: true });

    if (chunk) {
      // 순수 텍스트 누적
      accumulatedText += chunk;

      // 실시간 업데이트 (마지막 메시지만 수정)
      setMessages((prev) => {
        const updated = [...prev];
        const msgIndex = updated.findIndex((m) => m.id === streamingMessageId);
        if (msgIndex !== -1) {
          updated[msgIndex] = {
            ...updated[msgIndex],
            content: accumulatedText,  // 누적된 텍스트로 업데이트
          };
        }
        return updated;
      });
    }
  }
}
```

#### 핵심 포인트
- ✅ **초기 빈 메시지 생성**: 스트리밍 시작 시 빈 메시지를 리스트에 추가
- ✅ **텍스트 누적**: `accumulatedText += chunk`로 청크를 누적
- ✅ **증분 업데이트**: `setMessages`로 마지막 메시지만 업데이트
- ✅ **TextDecoder 사용**: `{ stream: true }` 옵션으로 멀티바이트 문자 처리

---

## 토큰 단위 증분 업데이트

### 문제: 전체 메시지 중복 전송

#### 기존 문제점
```python
# ❌ 잘못된 방식: 전체 메시지를 매번 전송
for event in graph.stream(state):
    messages = event.get("messages", [])
    last_message = messages[-1].content  # 전체 메시지
    yield last_message  # 중복 전송!
```

#### 해결: 증분 업데이트
```python
# ✅ 올바른 방식: 새로 생성된 토큰만 전송
async for event in graph.astream_events(state, version="v1"):
    if event["event"] == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        yield chunk.content  # 새 토큰만 yield
```

### LangGraph: `astream_events` 사용

#### 이벤트 타입
- `on_chat_model_start`: 모델 호출 시작
- `on_chat_model_stream`: **토큰 생성 이벤트** (이것만 사용)
- `on_chat_model_end`: 모델 호출 종료

#### 구현 예시
```python
async for event in graph.astream_events(state, version="v1"):
    event_type = event.get("event")

    if event_type == "on_chat_model_stream":
        chunk = event.get("data", {}).get("chunk")
        if chunk and hasattr(chunk, "content"):
            content = chunk.content
            if content:
                yield content  # 토큰 단위 전송
```

### RAG Chain: 증분 추출

#### 구현 예시
```python
accumulated_text = ""

async for chunk in chain.astream(input_data):
    answer = chunk.get("answer", "")
    if answer and len(answer) > len(accumulated_text):
        # 새로 추가된 부분만 추출
        delta = answer[len(accumulated_text):]
        accumulated_text = answer

        # 한 글자씩 스트리밍
        for char in delta:
            yield char
            await asyncio.sleep(0.01)
```

---

## Next.js API 라우트 프록시 패턴

### 아키텍처 이유

#### 왜 프록시가 필요한가?
1. **CORS 문제 해결**: 백엔드와 프론트엔드가 다른 포트에서 실행
2. **환경 변수 관리**: `NEXT_PUBLIC_*`로 프론트엔드에서 백엔드 URL 접근
3. **에러 처리 통합**: Next.js에서 통일된 에러 응답 형식
4. **타임아웃 관리**: 로컬 모델의 긴 응답 시간 처리

### 구현 패턴

#### 1. 요청 전달
```typescript
const response = await fetch(`${backendUrl}/api/graph`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ message, history }),
});
```

#### 2. 스트리밍 응답 처리
```typescript
if (contentType && contentType.includes("text/plain")) {
  // 스트림을 그대로 전달
  return new Response(response.body, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    },
  });
}
```

#### 3. 에러 처리
```typescript
if (!response.ok) {
  const errorData = await response.json().catch(() => ({}));
  return NextResponse.json(
    { error: errorData.detail || "서버 오류" },
    { status: response.status }
  );
}
```

---

## 구현 세부사항

### 1. 응답 정리 (Response Cleaning)

#### 문제: 시스템 프롬프트 및 태그 노출
LLM 응답에 `[[system]]`, `[[endofturn]]`, `[[assistant]]` 같은 내부 태그가 포함될 수 있습니다.

#### 해결: 정규식 기반 필터링

**백엔드 (`chat_service.py`, `graph.py`)**
```python
import re

# 1. 태그 제거
response_text = re.sub(r'\[\[system\]\].*?\[\[endofturn\]\]\s*', '', response_text, flags=re.DOTALL)
response_text = re.sub(r'\[\[assistant\]\]\s*', '', response_text, flags=re.IGNORECASE)
response_text = re.sub(r'\[\[endofturn\]\]\s*', '', response_text, flags=re.IGNORECASE)
response_text = re.sub(r'\[\[user\]\]\s*', '', response_text, flags=re.IGNORECASE)

# 2. 과거 대화 형식 제거 (Human:, Assistant:)
if "Human:" in response_text or "Assistant:" in response_text:
    assistant_match = re.search(r"Assistant:\s*(.+?)(?:\nHuman:|$)", response_text, re.DOTALL)
    if assistant_match:
        response_text = assistant_match.group(1).strip()

# 3. 간단한 인사에 대한 응답 정리
if any(greeting in message.lower() for greeting in ["안녕", "안녕하세요", "hi", "hello"]):
    lines = response_text.split('\n')
    clean_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("Human:") or line.startswith("Assistant:"):
            continue
        clean_lines.append(line)

    if clean_lines:
        response_text = '\n'.join(clean_lines)
    else:
        response_text = "안녕하세요! 어떻게 도와드릴 수 있을까요? 궁금한 점이 있거나 도움이 필요한 사항이 있으면 말씀해 주세요. 😊"
```

**스트리밍 중 필터링 (`graph_router.py`)**
```python
# 실시간으로 태그 제거
if not any(tag in content for tag in ["[[system]]", "[[endofturn]]", "[[user]]", "[[assistant]]"]):
    yield content
```

---

### 2. 지연 제어 (Delay Control)

#### 목적
- 부드러운 타이핑 효과
- 네트워크 부하 분산
- 사용자 경험 개선

#### 구현
```python
await asyncio.sleep(0.01)  # 10ms 지연
```

#### 최적화 고려사항
- **너무 짧으면**: 네트워크 부하 증가, 렌더링 비용 증가
- **너무 길면**: 느린 사용자 경험
- **권장값**: 10ms (0.01초)

---

### 3. 에러 처리

#### 백엔드 에러 처리
```python
async def stream_generator():
    try:
        async for event in graph.astream_events(state, version="v1"):
            # ... 스트리밍 로직 ...
    except Exception as e:
        print(f"[ERROR] 스트리밍 중 오류: {e}")
        yield f"\n\n[오류 발생: {str(e)}]"
```

#### 프론트엔드 에러 처리
```typescript
try {
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    // ... 처리 ...
  }
} catch (streamError) {
  console.error("[ERROR] 스트리밍 처리 실패:", streamError);
  setMessages((prev) => {
    // 에러 메시지로 업데이트
    const updated = [...prev];
    const msgIndex = updated.findIndex((m) => m.id === streamingMessageId);
    if (msgIndex !== -1) {
      updated[msgIndex] = {
        ...updated[msgIndex],
        content: accumulatedText || "⚠️ 스트리밍 중 오류가 발생했습니다.",
      };
    }
    return updated;
  });
}
```

---

## 성능 최적화

### 1. 메모리 최적화

#### 문제: 전체 메시지 히스토리 누적
과거 대화가 계속 누적되면 메모리 사용량이 증가합니다.

#### 해결: 히스토리 제한
```python
# 최근 N개 메시지만 유지
if len(chat_history) > MAX_HISTORY_LENGTH:
    chat_history = chat_history[-MAX_HISTORY_LENGTH:]
```

### 2. 네트워크 최적화

#### 청크 크기 조절
```python
# 한 글자씩 전송 (부드러운 UX)
for char in delta:
    yield char
    await asyncio.sleep(0.01)

# 또는 여러 글자씩 전송 (빠른 전송)
chunk_size = 5
for i in range(0, len(delta), chunk_size):
    chunk = delta[i:i + chunk_size]
    yield chunk
    await asyncio.sleep(0.01)
```

### 3. 렌더링 최적화

#### React 상태 업데이트 최적화
```typescript
// ✅ 좋은 예: 마지막 메시지만 업데이트
setMessages((prev) => {
  const updated = [...prev];
  const msgIndex = updated.findIndex((m) => m.id === streamingMessageId);
  if (msgIndex !== -1) {
    updated[msgIndex] = { ...updated[msgIndex], content: accumulatedText };
  }
  return updated;
});

// ❌ 나쁜 예: 전체 메시지 리스트 재생성
setMessages([...messages, { ...streamingMessage, content: accumulatedText }]);
```

---

## 문제 해결 및 트러블슈팅

### 문제 1: JSON 객체가 화면에 표시됨

#### 증상
```
{"delta":"안녕하세요"}
{"delta":" 어떻게"}
{"delta":" 도와드릴까요?"}
```

#### 원인
Next.js API 라우트에서 백엔드의 순수 텍스트를 JSON으로 감싸서 전송

#### 해결
```typescript
// ❌ 잘못된 방식
controller.enqueue(
  new TextEncoder().encode(JSON.stringify({ delta: chunk }) + '\n')
);

// ✅ 올바른 방식
return new Response(response.body, {
  headers: { "Content-Type": "text/plain; charset=utf-8" },
});
```

---

### 문제 2: 전체 메시지가 중복 전송됨

#### 증상
스트리밍 중에 이전에 보낸 텍스트가 계속 반복됨

#### 원인
`graph.stream()`을 사용하여 전체 상태를 매번 전송

#### 해결
```python
# ❌ 잘못된 방식
for event in graph.stream(state):
    messages = event.get("messages", [])
    yield messages[-1].content  # 전체 메시지

# ✅ 올바른 방식
async for event in graph.astream_events(state, version="v1"):
    if event["event"] == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        yield chunk.content  # 새 토큰만
```

---

### 문제 3: 한글 문자가 깨짐

#### 증상
한글이 `` 같은 문자로 표시됨

#### 원인
`TextDecoder`에서 `{ stream: true }` 옵션 미사용

#### 해결
```typescript
// ✅ 올바른 방식
const decoder = new TextDecoder();
const chunk = decoder.decode(value, { stream: true });  // stream: true 필수
```

---

### 문제 4: 스트리밍이 너무 빠르거나 느림

#### 증상
- 너무 빠름: 화면이 깜빡임, CPU 사용량 증가
- 너무 느림: 사용자가 기다림

#### 해결
```python
# 지연 시간 조절
await asyncio.sleep(0.01)  # 10ms (기본값)
await asyncio.sleep(0.005)  # 5ms (더 빠름)
await asyncio.sleep(0.02)   # 20ms (더 느림)
```

---

## 요약

### 핵심 전략
1. **백엔드**: `astream_events` (LangGraph) 또는 `astream` (RAG Chain) 사용
2. **토큰 단위 전송**: 전체 메시지가 아닌 새로 생성된 토큰만 yield
3. **순수 텍스트 스트리밍**: JSON 변환 없이 `text/plain`으로 전송
4. **프록시 패턴**: Next.js API 라우트에서 스트림을 그대로 전달
5. **증분 업데이트**: 프론트엔드에서 텍스트를 누적하여 마지막 메시지만 업데이트

### 파일 구조
```
api/app/routers/
  ├── graph_router.py      # LangGraph 스트리밍 (astream_events)
  └── chat_router.py        # RAG Chain 스트리밍 (astream)

frontend/app/api/
  ├── graph/route.ts        # LangGraph 프록시
  └── chat/route.ts         # RAG Chain 프록시

frontend/app/
  └── page.tsx              # 스트리밍 처리 및 UI 업데이트
```

### 성능 지표
- **스트리밍 지연**: ~10ms per token
- **메모리 사용**: 히스토리 제한으로 최적화
- **네트워크 효율**: 증분 업데이트로 중복 전송 제거

---

## 참고 자료

- [LangGraph Streaming Documentation](https://langchain-ai.github.io/langgraph/how-tos/streaming/)
- [FastAPI StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)
- [Next.js API Routes](https://nextjs.org/docs/api-routes/introduction)
- [ReadableStream API](https://developer.mozilla.org/en-US/docs/Web/API/ReadableStream)

---

**작성일**: 2026-01-21
**버전**: 1.0
**작성자**: AI Assistant
