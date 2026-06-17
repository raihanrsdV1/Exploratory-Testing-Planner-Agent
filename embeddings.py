"""
Pluggable text-embedding layer for semantic (vector) retrieval.

Backends, selected via EMBEDDING_BACKEND (default "auto"):
    - fastembed             ONNX, no torch, SOTA bge models — recommended on-device
    - sentence_transformers torch-based, if you already have it
    - gemini                Google text-embedding API (no local model)

`auto` picks the first backend that is importable / configured. The model is
loaded lazily on first use and cached for the process lifetime.

Env:
    EMBEDDING_BACKEND   auto | fastembed | sentence_transformers | gemini | none
    EMBEDDING_MODEL     model name (backend-specific; sensible default per backend)
    GEMINI_API_KEY      required for the gemini backend
    GEMINI_EMBED_MODEL  default "models/text-embedding-004" (768 dims)
"""

from __future__ import annotations

import importlib.util
import os

_BACKEND = os.getenv("EMBEDDING_BACKEND", "auto").strip().lower()
_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "").strip()
_GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004").strip()

_DEFAULTS = {
    "fastembed": "BAAI/bge-small-en-v1.5",          # 384 dims
    "sentence_transformers": "all-MiniLM-L6-v2",     # 384 dims
}

# Lazily-populated process-wide state.
_resolved_backend: str | None = None
_model = None
_dim: int | None = None


def _have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _resolve_backend() -> str:
    """Decide which backend to actually use."""
    global _resolved_backend
    if _resolved_backend is not None:
        return _resolved_backend

    if _BACKEND == "none":
        _resolved_backend = "none"
    elif _BACKEND == "auto":
        if _have("fastembed"):
            _resolved_backend = "fastembed"
        elif _have("sentence_transformers"):
            _resolved_backend = "sentence_transformers"
        elif os.getenv("GEMINI_API_KEY") and _have("google.genai"):
            _resolved_backend = "gemini"
        else:
            _resolved_backend = "none"
    else:
        _resolved_backend = _BACKEND
    return _resolved_backend


def is_enabled() -> bool:
    return _resolve_backend() != "none"


def backend_name() -> str:
    return _resolve_backend()


def _load_model():
    global _model, _dim
    if _model is not None:
        return _model

    backend = _resolve_backend()
    if backend == "fastembed":
        from fastembed import TextEmbedding

        name = _MODEL_NAME or _DEFAULTS["fastembed"]
        _model = TextEmbedding(model_name=name)
    elif backend == "sentence_transformers":
        from sentence_transformers import SentenceTransformer

        name = _MODEL_NAME or _DEFAULTS["sentence_transformers"]
        _model = SentenceTransformer(name)
        _dim = int(_model.get_sentence_embedding_dimension())
    elif backend == "gemini":
        from google import genai

        _model = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
    else:
        raise RuntimeError("Embeddings are disabled (EMBEDDING_BACKEND=none / no backend available).")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]] | None:
    """Embed a batch of texts. Returns None when embeddings are disabled."""
    global _dim
    texts = [t if isinstance(t, str) else str(t) for t in texts]
    if not texts or not is_enabled():
        return None

    backend = _resolve_backend()
    model = _load_model()

    if backend == "fastembed":
        vectors = [list(map(float, v)) for v in model.embed(texts)]
    elif backend == "sentence_transformers":
        vectors = [list(map(float, v)) for v in model.encode(texts, normalize_embeddings=True)]
    elif backend == "gemini":
        resp = model.models.embed_content(model=_GEMINI_EMBED_MODEL, contents=texts)
        vectors = [list(map(float, e.values)) for e in resp.embeddings]
    else:
        return None

    if vectors and _dim is None:
        _dim = len(vectors[0])
    return vectors


def embed_query(text: str) -> list[float] | None:
    vecs = embed_texts([text])
    return vecs[0] if vecs else None


def embedding_dim() -> int | None:
    """Return the embedding dimension, loading the model once to discover it if needed."""
    global _dim
    if _dim is not None:
        return _dim
    if not is_enabled():
        return None
    v = embed_query("dimension probe")
    return len(v) if v else None
