"""
Format-agnostic document loader.

Normalises ANY requirements document (PDF, DOCX, PPTX, HTML, Markdown, plain text,
RTF, CSV, ...) into a single text/Markdown representation so the rest of the
ingestion pipeline never has to care about the source format.

Resolution order for rich formats (first available wins):
    1. MarkItDown   (Microsoft)  - lightweight, broad format coverage
    2. Docling      (IBM)        - SOTA layout-aware PDF/Office parsing
    3. Unstructured              - robust fallback for messy scans
    4. pypdf                     - last-resort plain PDF text

Plain-text formats (.txt/.md/.markdown/.text) are read directly and never need
an optional dependency. If a rich format is supplied but no converter is
installed, a clear, actionable error is raised.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# Extensions we can always read with the stdlib, no optional deps required.
_PLAIN_EXTS = {".txt", ".md", ".markdown", ".text", ".rst"}

# Extensions that need a rich converter.
_RICH_EXTS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    ".html", ".htm", ".rtf", ".odt", ".epub", ".csv",
}


def _have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def available_loaders() -> list[str]:
    """Report which rich converters are installed (for diagnostics)."""
    out = []
    if _have("markitdown"):
        out.append("markitdown")
    if _have("docling"):
        out.append("docling")
    if _have("unstructured"):
        out.append("unstructured")
    if _have("pypdf"):
        out.append("pypdf")
    return out


def _load_with_markitdown(path: Path) -> str:
    from markitdown import MarkItDown

    md = MarkItDown()
    result = md.convert(str(path))
    return (getattr(result, "text_content", "") or "").strip()


def _load_with_docling(path: Path) -> str:
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(path))
    return (result.document.export_to_markdown() or "").strip()


def _load_with_unstructured(path: Path) -> str:
    from unstructured.partition.auto import partition

    elements = partition(filename=str(path))
    return "\n\n".join(str(el) for el in elements).strip()


def _load_with_pypdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()


def _convert_rich(path: Path) -> tuple[str, str]:
    """Try each installed converter in priority order. Returns (text, loader_name)."""
    attempts: list[tuple[str, callable]] = []
    if _have("markitdown"):
        attempts.append(("markitdown", _load_with_markitdown))
    if _have("docling"):
        attempts.append(("docling", _load_with_docling))
    if _have("unstructured"):
        attempts.append(("unstructured", _load_with_unstructured))
    if path.suffix.lower() == ".pdf" and _have("pypdf"):
        attempts.append(("pypdf", _load_with_pypdf))

    if not attempts:
        raise RuntimeError(
            f"Cannot read '{path.name}' ({path.suffix}): no document converter installed. "
            "Install one of: `pip install markitdown` (recommended), `pip install docling`, "
            "or `pip install unstructured`. Plain .txt/.md files need no extra packages."
        )

    errors: list[str] = []
    for name, fn in attempts:
        try:
            text = fn(path)
            if text:
                return text, name
            errors.append(f"{name}: produced empty output")
        except Exception as e:  # noqa: BLE001 - report and continue to next converter
            errors.append(f"{name}: {e}")

    raise RuntimeError(
        f"All available converters failed for '{path.name}'. Details: " + " | ".join(errors)
    )


def load_document(
    source_path: str | None = None,
    raw_text: str | None = None,
) -> dict:
    """
    Load a document from a path (any format) or from inline text.

    Returns a canonical dict:
        {
          "text":   <normalised text / markdown>,
          "format": <file extension without dot, or "inline">,
          "loader": <which loader produced the text>,
          "chars":  <length>,
        }
    """
    if raw_text is not None and raw_text.strip():
        return {
            "text": raw_text,
            "format": "inline",
            "loader": "inline",
            "chars": len(raw_text),
        }

    if not source_path:
        raise ValueError("load_document requires either source_path or raw_text")

    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {source_path}")

    ext = path.suffix.lower()

    if ext in _PLAIN_EXTS or ext == "":
        text = path.read_text(encoding="utf-8", errors="ignore")
        return {
            "text": text,
            "format": ext.lstrip(".") or "txt",
            "loader": "plain",
            "chars": len(text),
        }

    if ext in _RICH_EXTS:
        text, loader = _convert_rich(path)
        return {
            "text": text,
            "format": ext.lstrip("."),
            "loader": loader,
            "chars": len(text),
        }

    # Unknown extension: try plain read first (many configs/specs are just text),
    # then fall back to rich converters if it looks binary.
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
        return {
            "text": text,
            "format": ext.lstrip(".") or "txt",
            "loader": "plain",
            "chars": len(text),
        }
    except (UnicodeDecodeError, ValueError):
        text, loader = _convert_rich(path)
        return {
            "text": text,
            "format": ext.lstrip("."),
            "loader": loader,
            "chars": len(text),
        }
