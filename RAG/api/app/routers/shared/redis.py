"""Upstash Redis(REST) 클라이언트 및 공용 유틸.

요구사항:
- `.env`의 `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN`로 접속
- 로그인(JWT access token 발급) 시 Redis에 토큰을 저장하고
  BullMQ(별도 Node 워커)가 구독할 수 있도록 이벤트를 발행할 수 있게 함

주의:
- 이 모듈은 "설정/연결"만 담당합니다. 비즈니스 로직(권한/인증/큐 처리)은 호출부에서 수행하세요.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

from upstash_redis import Redis  # type: ignore

if TYPE_CHECKING:
    from upstash_redis.asyncio import Redis as AsyncRedis  # type: ignore
else:
    try:
        from upstash_redis.asyncio import Redis as AsyncRedis  # type: ignore
    except ImportError:
        AsyncRedis = None  # type: ignore


@dataclass(frozen=True)
class UpstashRedisConfig:
    url: str
    token: str


_redis: Optional[Redis] = None
_redis_async: Optional["AsyncRedis"] = None


def get_upstash_redis_config() -> UpstashRedisConfig:
    url = (os.getenv("UPSTASH_REDIS_REST_URL") or "").strip().strip("\"'")
    token = (os.getenv("UPSTASH_REDIS_REST_TOKEN") or "").strip().strip("\"'")
    if not url or not token:
        raise RuntimeError(
            "Upstash Redis 설정이 없습니다. "
            "`UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN` 환경변수를 설정하세요."
        )
    return UpstashRedisConfig(url=url, token=token)


def get_redis() -> Redis:
    """싱글톤 Upstash Redis 클라이언트."""
    global _redis
    if _redis is None:
        cfg = get_upstash_redis_config()
        _redis = Redis(url=cfg.url, token=cfg.token)
    return _redis


def get_redis_async() -> "AsyncRedis":
    """싱글톤 Upstash Redis 비동기 클라이언트 (upstash_redis.asyncio)."""
    global _redis_async
    if AsyncRedis is None:
        raise RuntimeError("upstash_redis.asyncio를 사용할 수 없습니다. upstash-redis 패키지를 확인하세요.")
    if _redis_async is None:
        cfg = get_upstash_redis_config()
        _redis_async = AsyncRedis(url=cfg.url, token=cfg.token)
    return _redis_async


def store_access_token(
    *,
    user_id: str,
    jti: str,
    access_token: str,
    ttl_seconds: int,
) -> None:
    """JWT access token을 Redis에 저장합니다.

    - 키 설계:
      - `auth:access:{jti}`: token 본문(블랙리스트/세션 조회용)
      - `auth:user:{user_id}:access_jti`: 마지막 access token의 jti(선택)
    """
    r = get_redis()
    r.set(f"auth:access:{jti}", access_token, ex=ttl_seconds)
    r.set(f"auth:user:{user_id}:access_jti", jti, ex=ttl_seconds)


def publish_bullmq_event(event: str, payload: Dict[str, Any]) -> None:
    """BullMQ 워커(별도 Node)가 구독할 수 있도록 Redis pub/sub 이벤트를 발행합니다.

    BullMQ 자체는 내부적으로 pub/sub를 쓰지 않지만,
    운영에서 "Python API -> Node Worker" 브릿지로 간단히 쓰기 좋습니다.
    """
    r = get_redis()
    channel = os.getenv("BULLMQ_EVENT_CHANNEL", "bullmq:events")
    message = json.dumps({"event": event, "payload": payload}, ensure_ascii=False)
    # upstash-redis는 publish 지원
    r.publish(channel, message)

