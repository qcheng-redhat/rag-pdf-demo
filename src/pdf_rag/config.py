import os
from pathlib import Path


def persist_dir(cli_value: Path | None = None) -> Path:
    if cli_value is not None:
        return cli_value.resolve()
    env = os.environ.get("PDF_RAG_DB")
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd() / ".pdf_rag_chroma"


def embedding_backend(cli_override: str | None = None) -> str:
    raw = (cli_override or os.environ.get("PDF_RAG_EMBED_BACKEND", "openai")).lower()
    if raw not in ("local", "openai"):
        raise SystemExit(
            f"Unknown PDF_RAG_EMBED_BACKEND {raw!r}; use 'local' or 'openai'."
        )
    return raw


def openai_embed_model() -> str:
    return os.environ.get("PDF_RAG_EMBED_MODEL", "text-embedding-3-small")


def local_embed_model() -> str:
    return os.environ.get(
        "PDF_RAG_LOCAL_EMBED_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )


def local_embed_batch_size() -> int:
    return max(1, int(os.environ.get("PDF_RAG_LOCAL_EMBED_BATCH", "32")))


def chunk_size() -> int:
    return int(os.environ.get("PDF_RAG_CHUNK_SIZE", "1200"))


def chunk_overlap() -> int:
    return int(os.environ.get("PDF_RAG_CHUNK_OVERLAP", "200"))


def chat_model() -> str:
    return os.environ.get("PDF_RAG_CHAT_MODEL", "gpt-4o-mini")
