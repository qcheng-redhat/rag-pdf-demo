import hashlib
import uuid
from pathlib import Path

from pdf_rag import config
from pdf_rag.chunking import chunk_text
from pdf_rag.embedding import embed_texts, openai_embed_client
from pdf_rag.ingest import iter_pdf_documents
from pdf_rag.store import get_collection, load_file_hashes, save_file_hashes


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


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

    col = get_collection(persist_dir, recreate=reset)
    old_hashes = {} if reset else load_file_hashes(persist_dir)

    # Compute current file hashes
    current_hashes: dict[str, str] = {}
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        current_hashes[str(pdf_path.resolve())] = _file_hash(pdf_path)

    # Find changed, new, and deleted files
    changed_paths: set[str] = set()
    for path, hash_val in current_hashes.items():
        if path not in old_hashes or old_hashes[path] != hash_val:
            changed_paths.add(path)

    deleted_paths = [p for p in old_hashes if p not in current_hashes]

    # Nothing to do
    if not changed_paths and not deleted_paths:
        return 0

    # Remove stale chunks for changed and deleted files
    for path in list(changed_paths) + deleted_paths:
        try:
            col.delete(where={"source": path})
        except Exception:
            pass

    # Process only changed/new files
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
        resolved = str(pdf_path.resolve())
        if resolved not in changed_paths:
            continue
        for page_num, page_text in pages:
            for chunk in chunk_text(page_text, chunk_sz, chunk_ov):
                batch_texts.append(chunk)
                batch_meta.append((str(uuid.uuid4()), resolved, page_num))
                if len(batch_texts) >= embed_batch:
                    flush_batch()
    flush_batch()

    if ids:
        col.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    save_file_hashes(persist_dir, current_hashes)
    return len(ids)
