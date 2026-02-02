"""간단 Auth 라우터.

요구사항(사용자 요청):
- 로그인 시 JWT access token 발급
- Upstash Redis에 access token 저장
- BullMQ(별도 Node 워커)로 이벤트 발행할 수 있게 연결

주의:
- 현재 레포에 사용자/비밀번호 인증 DB 로직이 없어서,
  MVP로 "user_id만 받는 로그인" 형태로 제공합니다.
  (추후 실제 사용자 테이블/비밀번호 검증 로직으로 교체 권장)
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, Optional

import jwt  # type: ignore
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.routers.shared.redis import publish_bullmq_event, store_access_token


router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])


class LoginRequest(BaseModel):
    user_id: str = Field(..., description="로그인 사용자 ID(임시)")


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


def _get_jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET") or os.getenv("SECRET_KEY") or ""
    secret = secret.strip()
    if not secret:
        # 개발 편의용 fallback (운영에서는 반드시 설정하세요)
        secret = "CHANGE_ME_JWT_SECRET"
    return secret


@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest) -> LoginResponse:
    """로그인(임시) - JWT 발급 + Redis 저장 + BullMQ 이벤트 발행."""
    user_id = body.user_id.strip()
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id가 필요합니다.",
        )

    now = int(time.time())
    expires_in = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES_IN", "3600"))
    jti = uuid.uuid4().hex

    payload: Dict[str, Any] = {
        "sub": user_id,
        "iat": now,
        "exp": now + expires_in,
        "jti": jti,
        "type": "access",
    }

    token = jwt.encode(payload, _get_jwt_secret(), algorithm="HS256")
    if isinstance(token, bytes):
        token = token.decode("utf-8")

    # Redis 저장 (세션/블랙리스트/단일 로그인 등 용도)
    try:
        store_access_token(
            user_id=user_id,
            jti=jti,
            access_token=token,
            ttl_seconds=expires_in,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Redis 저장 실패: {str(e)}",
        )

    # BullMQ 워커가 구독할 수 있게 이벤트 발행 (옵션)
    try:
        publish_bullmq_event(
            "auth.login",
            {"user_id": user_id, "jti": jti, "expires_in": expires_in},
        )
    except Exception:
        # 이벤트 발행 실패는 로그인 자체를 막지 않음
        pass

    return LoginResponse(access_token=token, expires_in=expires_in)

