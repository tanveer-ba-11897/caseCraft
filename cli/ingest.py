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
from core.knowledge.chunker import chunk_document
from core.knowledge.embedder import Embedder


def _validate_index_path(index_path: str) -> str:
    """Validate that the index path is within the project directory."""
    project_root = Path(__file__).parent.parent.resolve()
    resolved = Path(index_path).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError:
        raise ValueError(f"Index path must be within the project directory. Got: {index_path}")
    return str(resolved)


def ingest_to_index(documents, index_path: str = "knowledge_base/index.json"):
    """
    Chunk, embed, and save documents to the knowledge index.
    """
    # Validate index path to prevent arbitrary file writes
    _validate_index_path(index_path)

    if not documents:
        print("[WARN] No documents to ingest")
        return
    
    print(f"[INFO] Processing {len(documents)} documents...")
    
    # Chunk documents
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"[INFO] Chunked '{doc.source_name[:50]}...' into {len(chunks)} chunks")
    
    print(f"[INFO] Total chunks: {len(all_chunks)}")
    
    # Embed chunks
    print("[INFO] Generating embeddings (this may take a while)...")
    embedder = Embedder()
    embedded_chunks = embedder.embed_chunks(all_chunks)
    
    # Load existing index if exists
    index_file = Path(index_path)
    existing_data = []
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        # Validate index schema
        required_keys = {"id", "text", "metadata", "embedding"}
        if isinstance(raw_data, list):
            existing_data = [
                entry for entry in raw_data
                if isinstance(entry, dict) and required_keys.issubset(entry.keys())
            ]
            skipped = len(raw_data) - len(existing_data)
            if skipped:
                print(f"[WARN] Skipped {skipped} invalid entries in index")
        else:
            print("[WARN] Invalid index format. Starting fresh.")
            existing_data = []
        print(f"[INFO] Loaded {len(existing_data)} existing entries from index")
    
    # Add new chunks
    for chunk in embedded_chunks:
        existing_data.append({
            "id": chunk.id,
            "text": chunk.text,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding,
        })
    
    # Save index
    index_file.parent.mkdir(parents=True, exist_ok=True)
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2)
    
    # SEC-09: Restrict file permissions (owner read/write only)
    import os
    import platform
    try:
        os.chmod(index_file, 0o600)
    except OSError:
        if platform.system() == "Windows":
            print("[WARN] Could not restrict file permissions on Windows. Consider setting permissions manually.")
    
    print(f"[SUCCESS] Saved {len(existing_data)} total entries to {index_path}")


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
    ingest_to_index(documents, args.index)


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
    ingest_to_index([doc], args.index)


def cmd_ingest_docs(args):
    """Handle ingest-docs command."""
    print(f"[INFO] Loading documents from: {args.directory}")
    documents = load_documents(args.directory)
    ingest_to_index(documents, args.index)


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
    ingest_to_index(documents, args.index)


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
        "--index", "-i",
        default="knowledge_base/index.json",
        help="Path to knowledge index file"
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
        "--index", "-i",
        default="knowledge_base/index.json",
        help="Path to knowledge index file"
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
        "--index", "-i",
        default="knowledge_base/index.json",
        help="Path to knowledge index file"
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
        "--index", "-i",
        default="knowledge_base/index.json",
        help="Path to knowledge index file"
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
