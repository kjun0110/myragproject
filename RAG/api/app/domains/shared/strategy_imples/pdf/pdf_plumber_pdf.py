"""
[Label 1] pdfplumber 전략.

표(Table)가 포함된 PDF, 선 기반 데이터 추출 필요할 때 선택.
표 구조 인식 및 데이터프레임 변환에 강함.
"""

from pathlib import Path

from app.domains.shared.strategies.pdf_strategy import PDFExtractionStrategy


class PdfPlumberExtractionStrategy(PDFExtractionStrategy):
    """pdfplumber 기반 PDF 텍스트 추출 (표 추출 특화)."""

    def extract(self, file_path: str) -> str:
        try:
            import pdfplumber
        except ImportError:
            raise RuntimeError("pdfplumber 전략 사용을 위해 pip install pdfplumber 필요")

        path = Path(file_path)
        if not path.exists():
            return ""

        parts = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    parts.append(text)
        return "\n".join(parts)
