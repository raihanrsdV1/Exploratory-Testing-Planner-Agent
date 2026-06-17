"""
Format-agnostic ingestion pipeline.

The flow is:  ANY input  ->  Source Adapter  ->  Canonical IR  ->  Graph Builder

- `document_loader`  turns any document format (pdf/docx/html/md/txt/...) into text.
- `extractor`        turns requirements text into a structured entity graph via an LLM.
- `ui_normalizer`    turns a UI design export (Figma JSON, ...) into a canonical UI IR.

Every module degrades gracefully: if an optional dependency or the model backend
is unavailable, it falls back to the previous rule-based behaviour so the system
keeps working.
"""

from . import document_loader, extractor, ui_normalizer

__all__ = ["document_loader", "extractor", "ui_normalizer"]
