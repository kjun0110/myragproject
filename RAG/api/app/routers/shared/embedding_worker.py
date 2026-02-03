"""
임베딩 큐 워커: Upstash Redis(REST)에서 job을 꺼내 처리합니다.

실행 (프로젝트 루트 RAG/ 에서):
  python -m api.app.routers.shared.embedding_worker [--queue player] [--poll-interval 2]

실행 (api/ 디렉터리에서):
  python -m app.routers.shared.embedding_worker [--queue player] [--poll-interval 2]

- 큐·워커는 백엔드에만 둠. Redis 자격 증명은 서버 env에만 있음.
- Upstash REST만 사용하므로 TCP(redis://) 없이 동작.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 프로젝트 루트(RAG/)에서 실행 시 api 디렉터리를 path에 추가
_this_file = Path(__file__).resolve()
_api_dir = _this_file.parent.parent.parent  # .../api/app/routers/shared -> .../api
if _api_dir.name == "api" and str(_api_dir) not in sys.path:
    sys.path.insert(0, str(_api_dir))

import time

from app.routers.shared.embedding_queue import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PROCESSING,
    QueueName,
    pop_pending_job,
    update_status,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_player_embedding_batch() -> dict:
    """선수 임베딩 배치 실행. 실제 생성/적재는 추후 hub/services 등에서 구현."""
    logger.info("[WORKER] Player 임베딩 배치 실행 (실제 로직은 추후 구현)")
    # TODO: app.domains.v10.soccer.hub 등에서 임베딩 생성/DB 적재
    return {"success": True, "message": "임베딩 트리거 처리됨 (실제 생성 로직은 추후 구현)"}


def process_job(job_id: str, queue_name: QueueName) -> None:
    """단일 job 처리: 상태를 processing → completed/failed 로 갱신."""
    update_status(job_id, STATUS_PROCESSING)
    try:
        if queue_name == "player":
            result = run_player_embedding_batch()
        else:
            result = {"success": False, "message": f"미지원 큐: {queue_name}"}
        update_status(job_id, STATUS_COMPLETED, result=result)
    except Exception as e:
        logger.exception("[WORKER] job 처리 실패 job_id=%s", job_id)
        update_status(job_id, STATUS_FAILED, error=str(e))


def main() -> None:
    # 프로젝트 루트(RAG/) 또는 api/ 의 .env 로드 (UPSTASH_REDIS_* 등)
    try:
        from dotenv import load_dotenv
        project_root = _this_file.parent.parent.parent.parent  # .../api/app/routers/shared -> RAG/
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Embedding queue worker (Upstash REST)")
    parser.add_argument(
        "--queue",
        type=str,
        default="player",
        choices=["player"],
        help="큐 이름",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="대기 목록 폴링 간격(초)",
    )
    args = parser.parse_args()
    queue_name: QueueName = args.queue  # type: ignore[assignment]

    logger.info("[WORKER] 시작 queue=%s poll_interval=%.1fs", queue_name, args.poll_interval)
    try:
        while True:
            job_id = pop_pending_job(queue_name)
            if job_id:
                process_job(job_id, queue_name)
            else:
                time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        logger.info("[WORKER] 종료")
        sys.exit(0)


if __name__ == "__main__":
    main()
