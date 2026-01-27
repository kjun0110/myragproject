"""Soccer bases 모듈.

축구 관련 데이터베이스 모델을 제공합니다.
"""

from app.domains.v10.shared.models.bases.base import Base
from .stadiums import Stadium
from .teams import Team
from .schedules import Schedule
from .players import Player

__all__ = ["Base", "Stadium", "Team", "Schedule", "Player"]
