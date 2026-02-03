# 워커 (참고)

임베딩 큐·워커는 **백엔드(Python)** 에 있습니다.

- **큐/워커**: `api/app/routers/shared/embedding_queue.py`, `api/app/routers/shared/embedding_worker.py`
- **실행**: `python -m api.app.routers.shared.embedding_worker` (프로젝트 루트 RAG/ 에서) 또는 `python -m app.routers.shared.embedding_worker` (api/ 에서)
- Redis(Upstash REST) 자격 증명은 백엔드 `.env`에만 두고, 프론트는 백엔드 API 호출·상태 폴링만 합니다.

이 디렉터리는 이전 Node/BullMQ 워커용이었으며, 현재는 사용하지 않습니다.
