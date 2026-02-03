"""
Upstash Redis(REST) 기반 임베딩 작업 큐.

- BullMQ처럼 job 큐 + 상태만 Redis(REST)로 구현.
- 백엔드에서만 사용. Redis 자격 증명은 서버 env에만 둠.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, Literal, Optional

from app.routers.shared.redis import get_redis

logger = logging.getLogger(__name__)

QueueName = Literal["player"]
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

PENDING_LIST_PREFIX = "embedding:"
JOB_HASH_PREFIX = "embedding:job:"
JOB_TTL_SECONDS = 86400 * 7  # 7일


def _pending_list_key(queue_name: str) -> str:
    return f"{PENDING_LIST_PREFIX}{queue_name}:pending"


def _job_hash_key(job_id: str) -> str:
    return f"{JOB_HASH_PREFIX}{job_id}"


def add_job(
    queue_name: QueueName,
    payload: Optional[Dict[str, Any]] = None,
) -> str:
    """큐에 작업을 넣고 job_id를 반환합니다."""
    job_id = uuid.uuid4().hex
    now = str(int(time.time()))
    payload = payload or {}
    r = get_redis()
    key = _job_hash_key(job_id)
    # HSET: status, created_at, updated_at, payload(JSON)
    r.hset(
        key,
        values={
            "status": STATUS_PENDING,
            "created_at": now,
            "updated_at": now,
            "payload": json.dumps(payload, ensure_ascii=False),
        },
    )
    r.expire(key, JOB_TTL_SECONDS)
    list_key = _pending_list_key(queue_name)
    r.lpush(list_key, job_id)
    logger.info("[EMBED_QUEUE] job 추가 queue=%s job_id=%s", queue_name, job_id)
    return job_id


def get_status(job_id: str) -> Optional[Dict[str, Any]]:
    """job_id의 상태를 반환합니다. 없으면 None."""
    r = get_redis()
    key = _job_hash_key(job_id)
    raw = r.hgetall(key)
    if not raw:
        return None
    # upstash_redis may return bytes or str
    def _s(v: Any) -> str:
        return v.decode() if isinstance(v, bytes) else (v or "")

    status = _s(raw.get("status", ""))
    created_at = _s(raw.get("created_at", ""))
    updated_at = _s(raw.get("updated_at", ""))
    result = _s(raw.get("result", ""))
    error = _s(raw.get("error", ""))
    out: Dict[str, Any] = {
        "job_id": job_id,
        "status": status,
        "created_at": created_at,
        "updated_at": updated_at,
    }
    if result:
        try:
            out["result"] = json.loads(result)
        except Exception:
            out["result"] = result
    if error:
        out["error"] = error
    return out


def update_status(
    job_id: str,
    status: str,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """작업 상태를 갱신합니다."""
    r = get_redis()
    key = _job_hash_key(job_id)
    now = str(int(time.time()))
    mapping: Dict[str, str] = {"status": status, "updated_at": now}
    if result is not None:
        mapping["result"] = json.dumps(result, ensure_ascii=False)
    if error is not None:
        mapping["error"] = error
    r.hset(key, values=mapping)
    r.expire(key, JOB_TTL_SECONDS)
    logger.info("[EMBED_QUEUE] job 상태 갱신 job_id=%s status=%s", job_id, status)


def pop_pending_job(queue_name: QueueName) -> Optional[str]:
    """대기 목록에서 job_id 하나를 꺼냅니다. 없으면 None."""
    r = get_redis()
    list_key = _pending_list_key(queue_name)
    # RPOP: list 오른쪽에서 꺼냄 (FIFO)
    job_id = r.rpop(list_key)
    if job_id is None:
        return None
    if isinstance(job_id, bytes):
        job_id = job_id.decode()
    return job_id
