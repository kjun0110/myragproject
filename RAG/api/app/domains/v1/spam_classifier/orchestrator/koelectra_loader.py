"""
KoELECTRA 게이트웨이 모델 로더.

스팸 분류 도메인에서 사용하는 KoELECTRA 모델을 로드하고 캐싱합니다.
"""

import sys
from pathlib import Path
from typing import Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
orchestrator_dir = current_file.parent  # api/app/domains/spam_classifier/orchestrator/
spam_classifier_dir = orchestrator_dir.parent  # api/app/domains/spam_classifier/
domains_dir = spam_classifier_dir.parent  # api/app/domains/
app_dir = domains_dir.parent  # api/app/
api_dir = app_dir.parent  # api/

sys.path.insert(0, str(api_dir))

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

        # Base 모델 경로
        model_dir = (
            api_dir / "artifacts" / "koelectra" / "koelectra-small-v3-discriminator"
        )
        if model_dir.exists() and (model_dir / "config.json").exists():
            base_model_path = str(model_dir)
        else:
            base_model_path = "monologg/koelectra-small-v3-discriminator"

        # LoRA 어댑터 경로
        lora_dir = (
            api_dir
            / "artifacts"
            / "koelectra"
            / "spam_adapter"
            / "koelectra-small-v3-discriminator-spam-lora"
        )
        lora_adapter_path = None
        if lora_dir.exists():
            subdirs = [d for d in lora_dir.iterdir() if d.is_dir()]
            if subdirs:
                lora_adapter_path = str(max(subdirs, key=lambda x: x.stat().st_mtime))
            else:
                lora_adapter_path = str(lora_dir)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] 디바이스: {device}")

        # 토크나이저 로드
        _koelectra_tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        # Base 모델 로드
        base_model = AutoModelForSequenceClassification.from_pretrained(base_model_path)
        base_model.to(device)

        # LoRA 어댑터 로드 (있는 경우)
        if lora_adapter_path:
            _koelectra_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        else:
            _koelectra_model = base_model

        _koelectra_model.eval()
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
