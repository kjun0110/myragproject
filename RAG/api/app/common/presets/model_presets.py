"""애플리케이션 공통 모델 프리셋(편의 함수).

core(인프라)에는 저수준 로딩 구현만 두고,
도메인/서비스에서 재사용하는 프리셋(예: 스팸 어댑터 결합)은 common에 둡니다.
"""

from __future__ import annotations

from typing import Tuple

from app.core.loaders.model_loader import ModelLoader


def load_exaone_with_spam_adapter(**kwargs) -> Tuple:
    """EXAONE + 스팸 분류 아답터를 로드합니다."""
    return ModelLoader.load_exaone_model(
        adapter_name="exaone3.5-2.4b-spam-lora",
        **kwargs,
    )


def load_koelectra_with_spam_adapter(**kwargs) -> Tuple:
    """KoELECTRA + 스팸 분류 아답터를 로드합니다."""
    return ModelLoader.load_koelectra_model(
        adapter_name="koelectra-small-v3-discriminator-spam-lora",
        **kwargs,
    )

