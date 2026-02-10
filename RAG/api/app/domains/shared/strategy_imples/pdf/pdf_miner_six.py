"""
[Label 2] pdfminer.six 전략.

텍스트의 정교한 위치 정보(좌표) 및 폰트 분석이 필요할 때 선택.
레이아웃 분석에 강점, 속도는 다소 느린 편.
"""

from pathlib import Path

from app.domains.shared.strategies.pdf_strategy import PDFExtractionStrategy


class PdfMinerSixExtractionStrategy(PDFExtractionStrategy):
    """pdfminer.six 기반 PDF 텍스트 추출."""

    def extract(self, file_path: str) -> str:
        try:
            from pdfminer.high_level import extract_text
        except ImportError:
            raise RuntimeError("pdfminer.six 전략 사용을 위해 pip install pdfminer.six 필요")

        path = Path(file_path)
        if not path.exists():
            return ""

        return extract_text(str(path))
