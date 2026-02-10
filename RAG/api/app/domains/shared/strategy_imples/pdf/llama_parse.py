"""
[Label 4] LlamaParse 전략.

복잡한 계층 구조, Markdown 변환 필요, RAG 최적화 데이터가 필요할 때 선택.
"""

from pathlib import Path

from app.domains.shared.strategies.pdf_strategy import PDFExtractionStrategy


class LlamaParseExtractionStrategy(PDFExtractionStrategy):
    """LlamaParse API 기반 PDF 추출 (Markdown/RAG 최적화)."""

    def extract(self, file_path: str) -> str:
        try:
            from llama_parse import LlamaParse
        except ImportError:
            raise RuntimeError(
                "LlamaParse 전략 사용을 위해 pip install llama-parse 필요"
            )

        path = Path(file_path)
        if not path.exists():
            return ""

        parser = LlamaParse(result_type="markdown")
        documents = parser.load_data(str(path))
        if not documents:
            return ""
        return "\n\n".join(doc.text for doc in documents if getattr(doc, "text", None))
