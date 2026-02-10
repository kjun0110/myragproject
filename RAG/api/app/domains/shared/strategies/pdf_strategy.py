"""
PDF 추출 전략 인터페이스 (GoF 전략 패턴).

KoELECTRA가 라벨(0~6)을 선택하면, 해당 라벨에 대응하는
strategy_imples/pdf/ 내 구현체가 이 인터페이스를 구현하여 실행됩니다.
"""

from abc import ABC, abstractmethod


class PDFExtractionStrategy(ABC):
    """PDF에서 텍스트를 추출하는 전략의 추상 인터페이스."""

    @abstractmethod
    def extract(self, file_path: str) -> str:
        """
        PDF 파일에서 텍스트를 추출합니다.

        Args:
            file_path: PDF 파일의 경로 (로컬 경로 또는 URI).

        Returns:
            추출된 전체 텍스트. 실패 시 빈 문자열 또는 예외.
        """
        pass
