"""
KoELECTRA 게이트웨이 모델 로더.

스팸 분류 도메인에서 사용하는 KoELECTRA 모델을 로드하고 캐싱합니다.
"""

import sys
from pathlib import Path
from typing import Tuple

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
orchestrator_dir = current_file.parent  # api/app/domains/spam_classifier/orchestrator/
spam_classifier_dir = orchestrator_dir.parent  # api/app/domains/spam_classifier/
domains_dir = spam_classifier_dir.parent  # api/app/domains/
app_dir = domains_dir.parent  # api/app/
api_dir = app_dir.parent  # api/

sys.path.insert(0, str(api_dir))

# 애플리케이션 공통 프리셋(스팸 아답터 결합)
from app.common.presets.model_presets import load_koelectra_with_spam_adapter

# 전역 캐싱
_koelectra_model = None
_koelectra_tokenizer = None
_koelectra_loading = False
_koelectra_error = None


def load_koelectra_gate() -> Tuple:
    """KoELECTRA 게이트웨이 모델을 로드합니다.

    Returns:
        (model, tokenizer) 튜플

    Raises:
        RuntimeError: 모델 로드 실패 시
    """
    global _koelectra_model, _koelectra_tokenizer, _koelectra_loading, _koelectra_error

    if _koelectra_model is not None and _koelectra_tokenizer is not None:
        return _koelectra_model, _koelectra_tokenizer

    if _koelectra_error is not None:
        raise RuntimeError(f"이전 KoELECTRA 모델 로드 실패: {_koelectra_error}")

    if _koelectra_loading:
        raise RuntimeError("KoELECTRA 모델이 현재 로딩 중입니다.")

    _koelectra_loading = True
    try:
        print("[INFO] KoELECTRA 게이트웨이 모델 로딩 시작...")

        # 공통 모델 로더 사용 (HuggingFace 캐시 + 로컬 아답터)
        _koelectra_model, _koelectra_tokenizer = load_koelectra_with_spam_adapter()

        print("[OK] KoELECTRA 게이트웨이 모델 로드 완료")
        _koelectra_loading = False
        _koelectra_error = None
        return _koelectra_model, _koelectra_tokenizer

    except Exception as e:
        _koelectra_loading = False
        error_msg = f"KoELECTRA 모델 로드 실패: {str(e)}"
        _koelectra_error = error_msg
        print(f"[ERROR] {error_msg}")
        import traceback

        print(f"[ERROR] 상세 오류:\n{traceback.format_exc()}")
        raise RuntimeError(error_msg) from e
