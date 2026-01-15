"""애플리케이션 설정 및 환경 변수 초기화."""

import os

# HuggingFace Hub 심볼릭 링크 비활성화 (Windows 권한 문제 해결)
# Windows에서 심볼릭 링크를 생성하려면 관리자 권한이 필요하므로,
# 심볼릭 링크를 사용하지 않도록 설정합니다.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# HuggingFace Hub 원격 코드 신뢰 설정
os.environ.setdefault("HF_HUB_TRUST_REMOTE_CODE", "true")
