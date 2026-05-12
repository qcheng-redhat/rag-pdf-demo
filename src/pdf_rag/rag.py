import os
import uuid
from pathlib import Path

from openai import OpenAI

from pdf_rag.chunking import chunk_text
from pdf_rag.ingest import iter_pdf_documents
from pdf_rag.local_embed import embed_texts_local
from pdf_rag.store import get_collection

CHAT_MODEL = os.environ.get("PDF_RAG_CHAT_MODEL", "gpt-4o-mini")
CHUNK_SIZE = int(os.environ.get("PDF_RAG_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.environ.get("PDF_RAG_CHUNK_OVERLAP", "200"))
OPENAI_EMBED_MODEL = os.environ.get("PDF_RAG_EMBED_MODEL", "text-embedding-3-small")


def resolve_embed_backend(cli_override: str | None) -> str:
    raw = (cli_override or os.environ.get("PDF_RAG_EMBED_BACKEND", "local")).lower()
    if raw not in ("local", "openai"):
        raise SystemExit(
            f"Unknown PDF_RAG_EMBED_BACKEND {raw!r}; use 'local' or 'openai'."
        )
    return raw


def _client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Required for ask (chat), "
            "and for index when PDF_RAG_EMBED_BACKEND=openai."
        )
    base = os.environ.get("OPENAI_BASE_URL")
    if base:
        return OpenAI(api_key=key, base_url=base)
    return OpenAI(api_key=key)


def embed_texts(
    texts: list[str],
    *,
    backend: str,
    openai_client: OpenAI | None,
) -> list[list[float]]:
    if not texts:
        return []
    if backend == "local":
        return embed_texts_local(texts)
    if openai_client is None:
        openai_client = _client()
    resp = openai_client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=texts,
    )
    return [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]


def index_folder(
    pdf_dir: Path,
    persist_dir: Path,
    reset: bool,
    *,
    embed_backend: str | None = None,
) -> int:
    backend = resolve_embed_backend(embed_backend)
    openai_client: OpenAI | None = None
    if backend == "openai":
        openai_client = _client()

    if reset and persist_dir.exists():
        import shutil

        shutil.rmtree(persist_dir)
    col = get_collection(persist_dir, recreate=True)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    embeddings: list[list[float]] = []

    batch_texts: list[str] = []
    batch_meta: list[tuple[str, str, int]] = []

    def flush_batch() -> None:
        nonlocal batch_texts, batch_meta
        if not batch_texts:
            return
        embs = embed_texts(
            batch_texts, backend=backend, openai_client=openai_client
        )
        for j, emb in enumerate(embs):
            fid, src, page = batch_meta[j]
            ids.append(fid)
            documents.append(batch_texts[j])
            metadatas.append({"source": src, "page": page})
            embeddings.append(emb)
        batch_texts = []
        batch_meta = []

    embed_batch = 64
    for pdf_path, pages in iter_pdf_documents(pdf_dir):
        for page_num, page_text in pages:
            for chunk in chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP):
                batch_texts.append(chunk)
                batch_meta.append(
                    (str(uuid.uuid4()), str(pdf_path.resolve()), page_num)
                )
                if len(batch_texts) >= embed_batch:
                    flush_batch()
    flush_batch()

    if not ids:
        return 0

    col.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return len(ids)


def ask(
    question: str,
    persist_dir: Path,
    top_k: int = 5,
    *,
    embed_backend: str | None = None,
) -> str:
    backend = resolve_embed_backend(embed_backend)
    client = _client()
    openai_for_embed: OpenAI | None = client if backend == "openai" else None

    col = get_collection(persist_dir)
    q_emb = embed_texts(
        [question], backend=backend, openai_client=openai_for_embed
    )[0]
    res = col.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    context_blocks: list[str] = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        src = meta.get("source", "?")
        page = meta.get("page", "?")
        context_blocks.append(f"[{src} p.{page}]\n{doc}")

    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no context)"

    system = (
        "You answer using only the provided context. "
        "If the answer is not in the context, say you do not know. "
        "Cite sources as file basename and page when possible."
    )
    user = f"Context:\n{context}\n\nQuestion:\n{question}"

    chat = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    choice = chat.choices[0].message.content
    return choice or ""
