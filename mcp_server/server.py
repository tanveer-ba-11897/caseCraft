import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Any
from mcp.server.fastmcp import FastMCP

# Configure server-side logging (stderr only — stdout is reserved for MCP protocol)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("casecraft_mcp")

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─── Security: Path Sandboxing ───────────────────────────────────────────────
ALLOWED_DIRS = [
    PROJECT_ROOT / "features",
    PROJECT_ROOT / "specs",
    PROJECT_ROOT / "docs",
]

VALID_APP_TYPES = {"web", "mobile", "desktop", "api"}

def _validate_file_path(file_path: str) -> str:
    """
    Validate that the requested file is within allowed directories.
    Prevents path traversal attacks.
    """
    resolved = Path(file_path).resolve()
    
    if not resolved.exists():
        # Try resolving relative to PROJECT_ROOT (fixes CWD issues when run from AnythingLLM/Claude)
        potential_path = (PROJECT_ROOT / file_path).resolve()
        if potential_path.exists():
            resolved = potential_path
        else:
            raise ValueError(f"File not found: {file_path}")
    
    for allowed in ALLOWED_DIRS:
        try:
            resolved.relative_to(allowed)
            return str(resolved)
        except ValueError:
            continue
    
    raise PermissionError(
        f"Access denied. Files must be in: {', '.join(d.name for d in ALLOWED_DIRS)}"
    )


# ─── Lazy Import Cache ──────────────────────────────────────────────────────
# Heavy modules (sentence-transformers, torch, etc.) are NOT imported at startup.
# They are loaded on first tool call so the MCP handshake completes instantly.
_generator = None
_config = None
_retriever_cls = None
_exporter = None

def _load_core():
    """Lazily import the heavy CaseCraft modules."""
    global _generator, _config, _retriever_cls
    if _generator is None:
        logger.info("Loading CaseCraft core modules (first call — this takes ~30s)...")
        from core.generator import generate_test_suite
        from core.config import config
        from core.knowledge.retriever import KnowledgeRetriever
        from core import exporter
        _generator = generate_test_suite
        _config = config
        _retriever_cls = KnowledgeRetriever
        _exporter = exporter
        logger.info("Core modules loaded successfully.")


# ─── Security: Rate Limiting ─────────────────────────────────────────────────
_last_generate_call = 0.0
RATE_LIMIT_SECONDS = 5


# 1. Initialize MCP Server (lightweight — no heavy deps)
mcp = FastMCP("CaseCraft")

logger.info("CaseCraft MCP server initialized. Waiting for client connection...")

# 2. Define Tools

@mcp.tool()
def generate_tests(file_path: str, app_type: str = None) -> str:
    """
    Generate a comprehensive test suite from a requirement document.
    
    Args:
        file_path: Path to the file in 'features/', 'specs/', or 'docs/'.
        app_type: Optional explicitly defined app type ('web', 'mobile', etc). 
                  If not provided, uses the default in casecraft.yaml.
    """
    global _last_generate_call
    
    # Validate app_type against whitelist
    if app_type and app_type not in VALID_APP_TYPES:
        return f"Error: Invalid app_type '{app_type}'. Must be one of: {VALID_APP_TYPES}"
    
    # Rate limiting
    elapsed = time.time() - _last_generate_call
    if elapsed < RATE_LIMIT_SECONDS:
        return f"Rate limited. Please wait {RATE_LIMIT_SECONDS - elapsed:.0f} seconds."
    _last_generate_call = time.time()

    # Validate file path
    try:
        validated_path = _validate_file_path(file_path)
    except (ValueError, PermissionError) as e:
        logger.warning(f"Path validation failed for '{file_path}': {e}")
        return f"Error: Invalid or inaccessible file path."

    # Lazy load heavy modules
    _load_core()

    try:
        suite = _generator(
            file_path=validated_path,
            dedup_semantic=True,
            app_type=app_type or _config.generation.app_type
        )
        
        # Save to file (Fix for missing output / context overflow)
        # Default to outputs/ directory with timestamp to avoid collisions
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = Path(file_path).stem
        output_dir = PROJECT_ROOT / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        json_path = output_dir / f"{base_name}_{timestamp}.json"
        excel_path = output_dir / f"{base_name}_{timestamp}.xlsx"
        
        # Export
        _exporter.export(suite, "json", str(json_path))
        _exporter.export(suite, "excel", str(excel_path))
        
        # Unload model from Ollama memory to free resources
        try:
            from core.llm_client import llm_client
            llm_client.unload_model(_config.general.model)
        except Exception:
            pass
        
        return (
            f"Successfully generated {len(suite.test_cases)} test cases.\n"
            f"Saved to:\n"
            f"- {json_path.name}\n"
            f"- {excel_path.name}\n\n"
            f"Summary: {suite.test_cases[:3] if suite.test_cases else 'No cases'}"
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        return "Error: Test generation failed. Check server logs for details."

@mcp.tool()
def query_knowledge(query: str, top_k: int = 3) -> str:
    """
    Search the CaseCraft knowledge base for relevant snippets.
    Useful for answering questions about product requirements.
    """
    # Clamp top_k to prevent resource exhaustion (-1 means retrieve all)
    if top_k != -1:
        top_k = max(1, min(top_k, 20))
    # Truncate query to prevent excessive memory usage
    query = query[:10000]

    try:
        _load_core()
        retriever = _retriever_cls()
        chunks = retriever.retrieve(query, top_k=top_k)
        
        if not chunks:
            return "No relevant information found."
            
        return "\n\n".join([f"[Source: {c.metadata.get('source', 'unknown')}]\n{c.text}" for c in chunks])
    except Exception as e:
        logger.error(f"Knowledge query failed: {e}", exc_info=True)
        return "Error: Knowledge query failed. Check server logs for details."

# 3. Entry Point
def main():
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()
