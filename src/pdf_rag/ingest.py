from pathlib import Path

from pypdf import PdfReader


def extract_pdf_text(path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        pages.append((i + 1, t))
    return pages


def iter_pdf_documents(pdf_dir: Path) -> list[tuple[Path, list[tuple[int, str]]]]:
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    out: list[tuple[Path, list[tuple[int, str]]]] = []
    for pdf in pdfs:
        out.append((pdf, extract_pdf_text(pdf)))
    return out
