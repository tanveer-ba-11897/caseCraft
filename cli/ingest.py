"""
Ingestion CLI for CaseCraft Knowledge Base.

Commands:
- sitemap: Crawl a sitemap.xml and ingest all pages
- url: Ingest a single URL
- urls: Ingest URLs from a text file (one URL per line)
- docs: Ingest local documents from a directory
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.knowledge.web_loader import (
    load_from_sitemap,
    load_from_url,
    load_from_url_list,
    CrawlConfig,
    validate_url,
)
from core.knowledge.loader import load_documents
from core.knowledge.ingest import ingest_documents
from core.config import load_config


def _validate_db_path(db_path: str) -> str:
    """Validate that the DB path is within the project directory."""
    project_root = Path(__file__).parent.parent.resolve()
    resolved = Path(db_path).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError:
        raise ValueError(f"DB path must be within the project directory. Got: {db_path}")
    return str(resolved)


def ingest_to_index(documents, db_path: str = "knowledge_base/chroma_db"):
    """
    Chunk, embed, and save documents to the ChromaDB vector store.
    """
    _validate_db_path(db_path)

    if not documents:
        print("[WARN] No documents to ingest")
        return

    cfg = load_config()
    kb_chunk_size = cfg.knowledge.kb_chunk_size
    print(f"[INFO] Using knowledge base chunk size: {kb_chunk_size} chars")
    print(f"[INFO] Vector store: {db_path}")

    def _print_progress(stage: str, msg: str) -> None:
        print(f"[INFO] {msg}")

    result = ingest_documents(
        documents,
        persist_dir=db_path,
        kb_chunk_size=kb_chunk_size,
        progress=_print_progress,
    )

    print(f"[SUCCESS] Stored {result.total_index_size} total chunks in {db_path}")


def cmd_ingest_sitemap(args):
    """Handle ingest-sitemap command."""
    config = CrawlConfig(
        delay_between_requests=args.delay,
        max_pages=args.max_pages,
        timeout=args.timeout,
    )
    
    if args.exclude:
        config.exclude_patterns = args.exclude
    
    try:
        validate_url(args.sitemap_url)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return
    
    print(f"[INFO] Starting sitemap crawl: {args.sitemap_url}")
    print(f"[INFO] Config: delay={config.delay_between_requests}s, max_pages={config.max_pages}")
    
    documents = load_from_sitemap(args.sitemap_url, config)
    ingest_to_index(documents, args.db_path)


def cmd_ingest_url(args):
    """Handle ingest-url command."""
    config = CrawlConfig(timeout=args.timeout)
    
    try:
        validate_url(args.url)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return
    
    print(f"[INFO] Loading URL: {args.url}")
    doc = load_from_url(args.url, config)
    ingest_to_index([doc], args.db_path)


def cmd_ingest_docs(args):
    """Handle ingest-docs command."""
    print(f"[INFO] Loading documents from: {args.directory}")
    documents = load_documents(args.directory)
    ingest_to_index(documents, args.db_path)


def cmd_ingest_urls(args):
    """Handle ingest-urls command (URL list from file)."""
    file_path = Path(args.file)
    
    if not file_path.exists():
        print(f"[ERROR] File not found: {args.file}")
        return
    
    # Read URLs from file (one per line, skip empty lines and comments)
    urls = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    validate_url(line)
                    urls.append(line)
                except ValueError as e:
                    print(f"[WARN] Skipping invalid URL '{line}': {e}")
    
    if not urls:
        print("[WARN] No URLs found in file")
        return
    
    print(f"[INFO] Found {len(urls)} URLs in {args.file}")
    
    config = CrawlConfig(
        delay_between_requests=args.delay,
        timeout=args.timeout,
    )
    
    documents = load_from_url_list(urls, config)
    ingest_to_index(documents, args.db_path)


def main():
    parser = argparse.ArgumentParser(
        description="CaseCraft Knowledge Base Ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Ingestion commands")
    
    # Sitemap command
    sitemap_parser = subparsers.add_parser(
        "sitemap",
        help="Crawl a sitemap.xml and ingest all pages"
    )
    sitemap_parser.add_argument(
        "sitemap_url",
        help="URL to sitemap.xml (e.g., https://docs.example.com/sitemap.xml)"
    )
    sitemap_parser.add_argument(
        "--delay", "-d",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)"
    )
    sitemap_parser.add_argument(
        "--max-pages", "-m",
        type=int,
        default=500,
        help="Maximum pages to crawl (default: 500)"
    )
    sitemap_parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30)"
    )
    sitemap_parser.add_argument(
        "--exclude", "-e",
        nargs="+",
        help="URL patterns to exclude (e.g., /api/ /blog/)"
    )
    sitemap_parser.add_argument(
        "--db-path",
        default="knowledge_base/chroma_db",
        help="ChromaDB persistence directory (default: knowledge_base/chroma_db)"
    )
    
    # Single URL command
    url_parser = subparsers.add_parser(
        "url",
        help="Ingest a single URL"
    )
    url_parser.add_argument(
        "url",
        help="URL to ingest"
    )
    url_parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30)"
    )
    url_parser.add_argument(
        "--db-path",
        default="knowledge_base/chroma_db",
        help="ChromaDB persistence directory (default: knowledge_base/chroma_db)"
    )
    
    # Local docs command
    docs_parser = subparsers.add_parser(
        "docs",
        help="Ingest local documents from a directory"
    )
    docs_parser.add_argument(
        "directory",
        help="Directory containing documents (PDF, MD, TXT)"
    )
    docs_parser.add_argument(
        "--db-path",
        default="knowledge_base/chroma_db",
        help="ChromaDB persistence directory (default: knowledge_base/chroma_db)"
    )
    
    # URL list file command
    urls_parser = subparsers.add_parser(
        "urls",
        help="Ingest URLs from a text file (one URL per line)"
    )
    urls_parser.add_argument(
        "file",
        help="Path to text file containing URLs (one per line)"
    )
    urls_parser.add_argument(
        "--delay", "-d",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)"
    )
    urls_parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30)"
    )
    urls_parser.add_argument(
        "--db-path",
        default="knowledge_base/chroma_db",
        help="ChromaDB persistence directory (default: knowledge_base/chroma_db)"
    )
    
    args = parser.parse_args()
    
    if args.command == "sitemap":
        cmd_ingest_sitemap(args)
    elif args.command == "url":
        cmd_ingest_url(args)
    elif args.command == "urls":
        cmd_ingest_urls(args)
    elif args.command == "docs":
        cmd_ingest_docs(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
