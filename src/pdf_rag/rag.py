import uuid
from pathlib import Path

from pdf_rag import config
from pdf_rag.chunking import chunk_text
from pdf_rag.embedding import embed_texts, openai_embed_client
from pdf_rag.ingest import iter_pdf_documents
from pdf_rag.store import get_collection


def index_folder(
    pdf_dir: Path,
    persist_dir: Path,
    reset: bool,
    *,
    embed_backend: str | None = None,
) -> int:
    backend = config.embedding_backend(embed_backend)
    chunk_sz = config.chunk_size()
    chunk_ov = config.chunk_overlap()
    openai_client = openai_embed_client() if backend == "openai" else None

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
            for chunk in chunk_text(page_text, chunk_sz, chunk_ov):
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
