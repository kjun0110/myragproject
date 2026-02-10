"""
[Label 6] Google Document AI 전략.

필기체 포함, 다국어 혼합, 구글 클라우드 에코시스템 연동 시 선택.
"""

from pathlib import Path

from app.domains.shared.strategies.pdf_strategy import PDFExtractionStrategy


class GoogleDocumentExtractionStrategy(PDFExtractionStrategy):
    """Google Document AI 기반 PDF 추출."""

    def extract(self, file_path: str) -> str:
        try:
            from google.cloud import documentai_v1 as documentai
        except ImportError:
            raise RuntimeError(
                "Google Document AI 전략 사용을 위해 pip install google-cloud-documentai 필요"
            )

        path = Path(file_path)
        if not path.exists():
            return ""

        with open(path, "rb") as f:
            blob = f.read()

        client = documentai.DocumentProcessorServiceClient()
        # 프로젝트/로케이터/프로세서는 환경변수 또는 설정에서 주입 권장
        name = _processor_name()
        if not name:
            raise ValueError(
                "Google Document AI processor name이 필요합니다. "
                "GOOGLE_DOCUMENT_AI_PROCESSOR 환경변수 설정 또는 코드에서 주입하세요."
            )

        raw_doc = documentai.RawDocument(content=blob, mime_type="application/pdf")
        request = documentai.ProcessRequest(name=name, raw_document=raw_doc)
        result = client.process_document(request=request)
        doc = result.document
        return doc.text or ""


def _processor_name() -> str | None:
    import os

    return os.environ.get("GOOGLE_DOCUMENT_AI_PROCESSOR") or None
