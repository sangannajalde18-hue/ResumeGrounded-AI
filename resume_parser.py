"""
Resume parsing utility supporting PDF, DOCX, and TXT files.
"""
import io
from pypdf import PdfReader
from docx import Document


def extract_resume_text(file_bytes: bytes, filename: str) -> str:
    """Extracts non-empty text content from uploaded resume bytes."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in {"pdf", "docx", "txt"}:
        raise ValueError(f"Unsupported file format '.{ext}'. Please upload a PDF, DOCX, or TXT file.")

    if len(file_bytes) > 10 * 1024 * 1024:
        raise ValueError("File exceeds maximum allowed size of 10 MB.")

    text = ""
    if ext == "txt":
        try:
            text = file_bytes.decode("utf-8").strip()
            if not text:
                text = file_bytes.decode("latin-1").strip()
        except Exception as e:
            raise ValueError(f"Could not read text file: {e}")

    elif ext == "docx":
        try:
            doc = Document(io.BytesIO(file_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            text = "\n".join(paragraphs).strip()
        except Exception as e:
            raise ValueError(f"Could not read DOCX document: {e}")

    elif ext == "pdf":
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    raise ValueError("PDF is password-protected and cannot be read.")
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n\n".join([p.strip() for p in pages if p.strip()]).strip()
        except Exception as e:
            raise ValueError(f"Could not parse PDF file: {e}")

    if not text:
        raise ValueError("No extractable text found in file. Ensure document is not an image-only scan.")

    return text
