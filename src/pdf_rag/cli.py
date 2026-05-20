import argparse
from pathlib import Path

from pdf_rag import config
from pdf_rag.chat import ask
from pdf_rag.rag import index_folder


def _shared_parent() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Chroma persist directory (default: ./.pdf_rag_chroma or $PDF_RAG_DB).",
    )
    p.add_argument(
        "--embed-backend",
        choices=("local", "openai"),
        default=None,
        help="Chunk embeddings: openai (default) or local (sentence-transformers). "
        "Override: $PDF_RAG_EMBED_BACKEND. Must match how the DB was built.",
    )
    return p


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Index PDFs and ask questions (Chroma). "
            "Embeddings via OpenAI-compatible API (defaults) or "
            "--embed-backend local. "
            "Override embedding host/key: PDF_RAG_EMBEDDINGS_BASE_URL / "
            "PDF_RAG_EMBEDDINGS_API_KEY; chat in ask(): PDF_RAG_CHAT_* or OPENAI_*."
        )
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    parent = _shared_parent()

    pi = sub.add_parser(
        "index", parents=[parent],
        help="Embed all *.pdf in a folder into the local vector DB.",
    )
    pi.add_argument(
        "--pdf-dir",
        type=Path,
        required=True,
        help="Directory containing PDF files.",
    )
    pi.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing DB at --db before indexing.",
    )

    pa = sub.add_parser(
        "ask", parents=[parent],
        help="Retrieve context and answer with the chat model.",
    )
    pa.add_argument("question", type=str, help="Your question.")
    pa.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve.",
    )

    args = p.parse_args()
    db = config.persist_dir(args.db)

    if args.cmd == "index":
        pdf_dir = args.pdf_dir.expanduser().resolve()
        if not pdf_dir.is_dir():
            raise SystemExit(f"Not a directory: {pdf_dir}")
        n = index_folder(pdf_dir, db, reset=args.reset, embed_backend=args.embed_backend)
        print(f"Indexed {n} chunks into {db}")
    elif args.cmd == "ask":
        print(ask(args.question, db, top_k=args.top_k, embed_backend=args.embed_backend))


if __name__ == "__main__":
    main()
