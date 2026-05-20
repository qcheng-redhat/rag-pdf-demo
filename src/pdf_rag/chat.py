import os
from pathlib import Path

from openai import OpenAI

from pdf_rag import config
from pdf_rag.embedding import embed_texts, openai_embed_client
from pdf_rag.store import get_collection


def _chat_client() -> OpenAI:
    """Build an OpenAI-compatible client for chat completions.

    Falls back to OPENAI_API_KEY / OPENAI_BASE_URL when PDF_RAG_CHAT_* unset.
    """
    key = os.environ.get("PDF_RAG_CHAT_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit(
            "Chat API key missing. Set OPENAI_API_KEY or PDF_RAG_CHAT_API_KEY for ask."
        )
    base = os.environ.get("PDF_RAG_CHAT_BASE_URL") or os.environ.get(
        "OPENAI_BASE_URL"
    )
    if base:
        return OpenAI(api_key=key, base_url=base.rstrip("/"))
    return OpenAI(api_key=key)


def ask(
    question: str,
    persist_dir: Path,
    top_k: int = 5,
    *,
    embed_backend: str | None = None,
) -> str:
    backend = config.embedding_backend(embed_backend)
    chat_cli = _chat_client()
    embed_cli = openai_embed_client() if backend == "openai" else None

    col = get_collection(persist_dir)
    q_emb = embed_texts([question], backend=backend, openai_client=embed_cli)[0]
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

    chat = chat_cli.chat.completions.create(
        model=config.chat_model(),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    choice = chat.choices[0].message.content
    return choice or ""
