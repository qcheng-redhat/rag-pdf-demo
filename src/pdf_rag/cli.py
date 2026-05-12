import argparse
import os
from pathlib import Path

from pdf_rag.rag import ask, index_folder


def _embed_backend_arg(p) -> None:
    p.add_argument(
        "--embed-backend",
        choices=("local", "openai"),
        default=None,
        help="Chunk embeddings: local (sentence-transformers, default) or openai. "
        "Override: $PDF_RAG_EMBED_BACKEND. Must match how the DB was built.",
    )


def default_persist_dir() -> Path:
    env = os.environ.get("PDF_RAG_DB")
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd() / ".pdf_rag_chroma"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Index PDFs and ask questions (Chroma; local or OpenAI embeddings, OpenAI chat)."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="Embed all *.pdf in a folder into the local vector DB.")
    pi.add_argument(
        "--pdf-dir",
        type=Path,
        required=True,
        help="Directory containing PDF files.",
    )
    pi.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Chroma persist directory (default: ./.pdf_rag_chroma or $PDF_RAG_DB).",
    )
    pi.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing DB at --db before indexing.",
    )
    _embed_backend_arg(pi)

    pa = sub.add_parser("ask", help="Retrieve context and answer with the chat model.")
    pa.add_argument("question", type=str, help="Your question.")
    pa.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Chroma persist directory (default: ./.pdf_rag_chroma or $PDF_RAG_DB).",
    )
    pa.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve.",
    )
    _embed_backend_arg(pa)

    args = p.parse_args()
    db = (args.db or default_persist_dir()).resolve()

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
