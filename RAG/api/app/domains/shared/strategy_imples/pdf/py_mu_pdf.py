"""
[Label 0] PyMuPDF 전략.

텍스트 위주, 빠른 처리 속도 필요, 레이아웃 단순할 때 선택.
속도가 압도적으로 빠르고 텍스트 추출 정확도가 높음.
(기업용 이용 시 AGPL 라이선스 확인 필요)
"""

from pathlib import Path

from app.domains.shared.strategies.pdf_strategy import PDFExtractionStrategy


class PyMuPDFExtractionStrategy(PDFExtractionStrategy):
    """PyMuPDF(fitz) 기반 PDF 텍스트 추출."""

    def extract(self, file_path: str) -> str:
        try:
            import fitz
        except ImportError:
            raise RuntimeError("PyMuPDF 전략 사용을 위해 pip install pymupdf 필요")

        path = Path(file_path)
        if not path.exists():
            return ""

        doc = fitz.open(str(path))
        try:
            parts = []
            for page in doc:
                parts.append(page.get_text())
            return "\n".join(parts)
        finally:
            doc.close()
