from pathlib import Path

import chromadb
from chromadb.config import Settings


def get_collection(
    persist_dir: Path,
    name: str = "pdf_chunks",
    *,
    recreate: bool = False,
):
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    if recreate:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
