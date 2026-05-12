import os

_model = None


def _model_name() -> str:
    return os.environ.get(
        "PDF_RAG_LOCAL_EMBED_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )


def _batch_size() -> int:
    return max(1, int(os.environ.get("PDF_RAG_LOCAL_EMBED_BATCH", "32")))


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(_model_name())
    return _model


def embed_texts_local(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = get_model()
    out: list[list[float]] = []
    bs = _batch_size()
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
