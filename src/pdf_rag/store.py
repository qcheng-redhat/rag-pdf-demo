import json
from pathlib import Path

import chromadb
from chromadb.config import Settings

HASHES_FILE = "file_hashes.json"


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


def load_file_hashes(persist_dir: Path) -> dict[str, str]:
    path = persist_dir / HASHES_FILE
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_file_hashes(persist_dir: Path, hashes: dict[str, str]) -> None:
    path = persist_dir / HASHES_FILE
    with open(path, "w") as f:
        json.dump(hashes, f, indent=2)
