import os

from openai import NotFoundError, OpenAI

from pdf_rag import config

_model = None


def _get_local_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(config.local_embed_model())
    return _model


def _embed_texts_local(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _get_local_model()
    out: list[list[float]] = []
    bs = config.local_embed_batch_size()
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        emb = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        if emb.ndim == 1:
            out.append(emb.tolist())
        else:
            out.extend(row.tolist() for row in emb)
    return out


def openai_embed_client() -> OpenAI:
    """Build an OpenAI-compatible client for embeddings.

    Falls back to OPENAI_API_KEY / OPENAI_BASE_URL when PDF_RAG_EMBEDDINGS_* unset.
    """
    key = os.environ.get("PDF_RAG_EMBEDDINGS_API_KEY") or os.environ.get(
        "OPENAI_API_KEY"
    )
    if not key:
        raise SystemExit(
            "Embedding API key missing. Set OPENAI_API_KEY, or PDF_RAG_EMBEDDINGS_API_KEY "
            "to use a different key for embeddings only. "
            "For index when using PDF_RAG_EMBED_BACKEND=openai (default), this is required."
        )
    base = os.environ.get("PDF_RAG_EMBEDDINGS_BASE_URL") or os.environ.get(
        "OPENAI_BASE_URL"
    )
    if base:
        return OpenAI(api_key=key, base_url=base.rstrip("/"))
    return OpenAI(api_key=key)


def embed_texts(
    texts: list[str],
    *,
    backend: str,
    openai_client: OpenAI | None = None,
) -> list[list[float]]:
    if not texts:
        return []
    if backend == "local":
        return _embed_texts_local(texts)
    if openai_client is None:
        openai_client = openai_embed_client()
    try:
        resp = openai_client.embeddings.create(
            model=config.openai_embed_model(),
            input=texts,
        )
    except NotFoundError as e:
        e_base = (
            os.environ.get("PDF_RAG_EMBEDDINGS_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "(default OpenAI API host)"
        )
        raise SystemExit(
            "Embeddings request returned HTTP 404 (not found). Typical causes:\n"
            f"  • Wrong base URL for embeddings (effective: {e_base}). Try "
            "'https://api.openai.com/v1' vs 'https://api.openai.com', or matching "
            "your provider docs.\n"
            f"  • PDF_RAG_EMBED_MODEL ({config.openai_embed_model()!r}) invalid for this host "
            "(set to the embedding model id your provider documents).\n"
            "  • DeepSeek: public docs focus on chat; if /v1/embeddings returns 404, "
            "that deployment may not offer embeddings—in that case use "
            "--embed-backend local or point PDF_RAG_EMBEDDINGS_* to a host that exposes "
            "OpenAI-compatible POST /v1/embeddings.\n"
            "  • Azure OpenAI expects the embedding deployment name as model.\n"
            f"Underlying error: {e}"
        ) from e
    return [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]
