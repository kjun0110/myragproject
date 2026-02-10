"""
[Label 3] pypdf 전략.

매우 단순한 텍스트 추출 또는 단순 병합/분할 연계 시 선택.
가장 오래되고 안정적이며, PDF 합치기/나누기 등 조작에 가볍고 편리함.
"""

from pathlib import Path

from app.domains.shared.strategies.pdf_strategy import PDFExtractionStrategy


class PyPDFExtractionStrategy(PDFExtractionStrategy):
    """pypdf(PyPDF2) 기반 PDF 텍스트 추출."""

    def extract(self, file_path: str) -> str:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise RuntimeError("pypdf 전략 사용을 위해 pip install pypdf 필요")

        path = Path(file_path)
        if not path.exists():
            return ""

        reader = PdfReader(str(path))
        parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
        return "\n".join(parts)
