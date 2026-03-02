#!/usr/bin/env python3
"""
CaseCraft Bridge Server

JSON-RPC bridge between the VS Code extension (TypeScript) and the
CaseCraft Python core. Communicates via line-delimited JSON over stdin/stdout.

Protocol
--------
TypeScript → Python (stdin):
  {"id": "<uuid>", "method": "generate", "params": {...}}
  {"id": "<uuid>", "method": "query",    "params": {...}}
  {"id": "<uuid>", "method": "ingest",   "params": {...}}
  {"id": "<uuid>", "method": "get_config","params": {}}
  {"id": "<uuid>", "type": "llm_response", "data": {"text": "..."}}

Python → TypeScript (stdout):
  {"id": "<uuid>", "type": "progress", "data": {"step": "...", "detail": "..."}}
  {"id": "<uuid>", "type": "result",   "data": {...}}
  {"id": "<uuid>", "type": "error",    "data": {"message": "..."}}
  {"id": "<uuid>", "type": "llm_request","data": {"prompt": "...", "json_mode": true}}
"""

import json
import logging
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

# ── Ensure project root is in path ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
#  I/O Protocol
# ═══════════════════════════════════════════════════════════════════════════════

_output_lock = threading.Lock()

# LLM proxy state – used when llm_provider is "vscode"
_pending_llm_events: dict[str, threading.Event] = {}
_pending_llm_results: dict[str, str] = {}
_pending_llm_errors: dict[str, Optional[str]] = {}
_pending_llm_lock = threading.Lock()

# Thread-local storage for per-request context (request id)
_thread_local = threading.local()

# Active cancellable requests (request_id → threading.Event)
_active_requests: dict[str, threading.Event] = {}
_active_requests_lock = threading.Lock()


def _write_json(obj: dict) -> None:
    """Thread-safe JSON-line write to stdout."""
    line = json.dumps(obj, ensure_ascii=False)
    with _output_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def _send_progress(request_id: str, step: str, detail: str) -> None:
    _write_json({
        "id": request_id,
        "type": "progress",
        "data": {"step": step, "detail": detail},
    })


def _send_result(request_id: str, data: dict) -> None:
    _write_json({
        "id": request_id,
        "type": "result",
        "data": data,
    })


def _send_error(request_id: str, message: str) -> None:
    _write_json({
        "id": request_id,
        "type": "error",
        "data": {"message": message},
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  VS Code LLM Proxy
# ═══════════════════════════════════════════════════════════════════════════════

LLM_PROXY_TIMEOUT = 300  # seconds
MAX_JSON_LINE_SIZE = 10 * 1024 * 1024  # 10 MB max per JSON line from stdin

# Allowed file extensions for generation
_ALLOWED_GENERATE_EXTENSIONS = {".pdf", ".txt", ".md"}


def _sanitize_path_for_display(path_str: str) -> str:
    """Remove absolute path prefix to avoid leaking directory structure in errors."""
    try:
        p = Path(path_str)
        return p.name
    except Exception:
        return "<file>"


def _validate_request_id(raw_id: object) -> str:
    """Validate / sanitize request_id to prevent log injection."""
    if not isinstance(raw_id, str):
        raw_id = str(raw_id)
    # Allow UUIDs and simple alphanumeric IDs up to 128 chars
    clean = raw_id[:128]
    if not all(c.isalnum() or c in "-_" for c in clean):
        return str(uuid.uuid4())
    return clean


def _validate_path_within_root(path: Path, label: str = "path") -> Path:
    """
    Ensure a resolved path is within PROJECT_ROOT.
    Prevents path traversal attacks (e.g. ../../etc/passwd).
    """
    resolved = path.resolve()
    if not str(resolved).startswith(str(PROJECT_ROOT)):
        raise ValueError(
            f"Access denied: {label} must be within the project directory"
        )
    return resolved


def _is_cancelled() -> bool:
    """Check if the current request has been cancelled."""
    cancel_event: Optional[threading.Event] = getattr(_thread_local, "cancel_event", None)
    return cancel_event is not None and cancel_event.is_set()


class RequestCancelledError(Exception):
    """Raised when a request is cancelled by the client."""


def vscode_llm_generate(prompt: str, model: str, json_mode: bool = False) -> str:
    """
    Proxy an LLM call through TypeScript / VS Code's Language Model API.

    Sends an ``llm_request`` message to stdout and blocks until the
    TypeScript extension writes back an ``llm_response`` on stdin.
    """
    # Check for cancellation before sending
    if _is_cancelled():
        raise RequestCancelledError("Request was cancelled")

    req_id = str(uuid.uuid4())
    event = threading.Event()

    with _pending_llm_lock:
        _pending_llm_events[req_id] = event

    _write_json({
        "id": req_id,
        "type": "llm_request",
        "data": {"prompt": prompt, "model": model, "json_mode": json_mode},
    })

    if not event.wait(timeout=LLM_PROXY_TIMEOUT):
        with _pending_llm_lock:
            _pending_llm_events.pop(req_id, None)
            _pending_llm_results.pop(req_id, None)
            _pending_llm_errors.pop(req_id, None)
        raise TimeoutError(
            f"VS Code LLM proxy request timed out after {LLM_PROXY_TIMEOUT}s"
        )

    with _pending_llm_lock:
        result = _pending_llm_results.pop(req_id, "")
        error = _pending_llm_errors.pop(req_id, None)
        _pending_llm_events.pop(req_id, None)

    if error:
        raise RuntimeError(f"VS Code LLM error: {error}")

    return result


def _handle_llm_response(msg: dict) -> None:
    """Dispatch an ``llm_response`` from TypeScript to the waiting thread."""
    req_id = msg.get("id")
    text = msg.get("data", {}).get("text", "")
    error = msg.get("data", {}).get("error")

    # Guard against oversized LLM responses (>5 MB)
    if isinstance(text, str) and len(text) > 5 * 1024 * 1024:
        text = ""
        error = "LLM response exceeded 5 MB limit"

    with _pending_llm_lock:
        if req_id in _pending_llm_events:
            _pending_llm_results[req_id] = text
            _pending_llm_errors[req_id] = error if error else None
            _pending_llm_events[req_id].set()


# ═══════════════════════════════════════════════════════════════════════════════
#  Progress Logging Handler
# ═══════════════════════════════════════════════════════════════════════════════

class BridgeLogHandler(logging.Handler):
    """Forward CaseCraft log messages as progress events to TypeScript."""

    def emit(self, record: logging.LogRecord) -> None:
        rid = getattr(_thread_local, "request_id", None)
        if rid and record.levelno >= logging.INFO:
            try:
                _send_progress(
                    rid,
                    record.name.replace("casecraft.", ""),
                    record.getMessage(),
                )
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Command Handlers
# ═══════════════════════════════════════════════════════════════════════════════

def _handle_generate(request_id: str, params: dict) -> None:
    _thread_local.request_id = request_id
    cancel_event = threading.Event()
    _thread_local.cancel_event = cancel_event
    with _active_requests_lock:
        _active_requests[request_id] = cancel_event

    try:
        from core.generator import generate_test_suite
        from core.config import config
        from core import exporter

        file_path = params.get("file_path")
        if not file_path:
            _send_error(request_id, "file_path is required")
            return

        path = Path(file_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path

        # Path traversal guard
        try:
            path = _validate_path_within_root(path, "file_path")
        except ValueError as e:
            _send_error(request_id, str(e))
            return

        if not path.exists():
            _send_error(request_id, f"File not found: {_sanitize_path_for_display(file_path)}")
            return

        # Extension allowlist
        if path.suffix.lower() not in _ALLOWED_GENERATE_EXTENSIONS:
            _send_error(
                request_id,
                f"Unsupported file type: {path.suffix}. "
                f"Allowed: {', '.join(sorted(_ALLOWED_GENERATE_EXTENSIONS))}"
            )
            return

        app_type = params.get("app_type", config.generation.app_type)
        dedup_semantic = params.get("dedup_semantic", config.quality.semantic_deduplication)
        reviewer_pass = params.get("reviewer_pass", config.quality.reviewer_pass)

        _send_progress(request_id, "starting", f"Processing {path.name}…")

        suite = generate_test_suite(
            file_path=str(path),
            dedup_semantic=dedup_semantic,
            reviewer_pass=reviewer_pass,
            app_type=app_type,
            progress=lambda stage, msg: _send_progress(request_id, stage, msg),
        )

        if _is_cancelled():
            _send_error(request_id, "Request was cancelled")
            return

        # ── Export ────────────────────────────────────────────────────────
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = path.stem
        output_dir = PROJECT_ROOT / "outputs"
        output_dir.mkdir(exist_ok=True)

        json_path = output_dir / f"{base_name}_{timestamp}.json"
        excel_path = output_dir / f"{base_name}_{timestamp}.xlsx"

        exporter.export(suite, "json", str(json_path))
        exporter.export(suite, "excel", str(excel_path))

        # Unload Ollama model if applicable
        try:
            from core.llm_client import llm_client
            llm_client.unload_model(config.general.model)
        except Exception:
            pass

        # Build a preview (first 5 cases)
        suite_dict = json.loads(suite.model_dump_json())
        preview = suite_dict.get("test_cases", [])[:5]

        _send_result(request_id, {
            "test_case_count": len(suite.test_cases),
            "json_path": str(json_path),
            "excel_path": str(excel_path),
            "feature_name": suite.feature_name,
            "test_cases": preview,
        })

    except RequestCancelledError:
        _send_error(request_id, "Request was cancelled")
    except Exception as e:
        logging.getLogger("casecraft.bridge").error("generate failed: %s", e, exc_info=True)
        _send_error(request_id, str(e))
    finally:
        _thread_local.request_id = None
        _thread_local.cancel_event = None
        with _active_requests_lock:
            _active_requests.pop(request_id, None)


def _handle_query(request_id: str, params: dict) -> None:
    _thread_local.request_id = request_id

    try:
        from core.knowledge.retriever import KnowledgeRetriever

        query = params.get("query", "")
        top_k = params.get("top_k", 5)

        if not query:
            _send_error(request_id, "query is required")
            return

        # Validate top_k
        try:
            top_k = int(top_k)
            top_k = max(1, min(top_k, 50))  # Clamp to sensible range
        except (TypeError, ValueError):
            top_k = 5

        retriever = KnowledgeRetriever()
        chunks = retriever.retrieve(query, top_k=top_k)

        results = []
        for chunk in chunks:
            results.append({
                "text": chunk.text,
                "source": chunk.metadata.get("source_name", chunk.metadata.get("source", "unknown")),
            })

        _send_result(request_id, {"chunks": results, "count": len(results)})

    except Exception as e:
        _send_error(request_id, str(e))
    finally:
        _thread_local.request_id = None


def _handle_ingest(request_id: str, params: dict) -> None:
    _thread_local.request_id = request_id

    try:
        source_type = params.get("source_type", "docs")
        source_path = params.get("path", "")

        if not source_path:
            _send_error(request_id, "path is required")
            return

        from core.knowledge.loader import load_documents

        _send_progress(request_id, "loading", f"Loading from {source_path}…")

        if source_type == "docs":
            # Resolve relative paths against project root
            resolved = Path(source_path)
            if not resolved.is_absolute():
                resolved = PROJECT_ROOT / resolved
            # Path traversal guard
            try:
                resolved = _validate_path_within_root(resolved, "path")
            except ValueError as e:
                _send_error(request_id, str(e))
                return
            if not resolved.exists():
                _send_error(request_id, f"Path not found: {_sanitize_path_for_display(source_path)}")
                return
            docs = load_documents(str(resolved))
        elif source_type == "url":
            from core.knowledge.web_loader import load_from_url
            docs = [load_from_url(source_path)]
        elif source_type == "sitemap":
            from core.knowledge.web_loader import load_from_sitemap
            docs = load_from_sitemap(source_path)
        else:
            _send_error(request_id, f"Unknown source_type: {source_type}")
            return

        if not docs:
            _send_error(request_id, f"No documents found at {source_path}")
            return

        from core.knowledge.ingest import ingest_documents

        result = ingest_documents(
            docs,
            progress=lambda stage, msg: _send_progress(request_id, stage, msg),
        )

        _send_result(request_id, {
            "documents": result.documents,
            "chunks": result.chunks,
            "total_index_size": result.total_index_size,
        })

    except Exception as e:
        _send_error(request_id, str(e))
    finally:
        _thread_local.request_id = None


def _handle_get_config(request_id: str, params: dict) -> None:
    try:
        from core.config import config

        _send_result(request_id, {
            "general": {
                "model": config.general.model,
                "llm_provider": config.general.llm_provider,
                "base_url": config.general.base_url,
                # api_key intentionally excluded — never expose secrets
                "timeout": config.general.timeout,
                "context_window_size": config.general.context_window_size,
                "max_context_window": config.general.max_context_window,
            },
            "generation": {
                "chunk_size": config.generation.chunk_size,
                "chunk_overlap": config.generation.chunk_overlap,
                "temperature": config.generation.temperature,
                "app_type": config.generation.app_type,
                "min_cases_per_chunk": config.generation.min_cases_per_chunk,
            },
            "quality": {
                "semantic_deduplication": config.quality.semantic_deduplication,
                "similarity_threshold": config.quality.similarity_threshold,
                "reviewer_pass": config.quality.reviewer_pass,
                "top_k": config.quality.top_k,
                "max_kb_batches": config.quality.max_kb_batches,
            },
            "knowledge": {
                "kb_chunk_size": config.knowledge.kb_chunk_size,
                "min_score_threshold": config.knowledge.min_score_threshold,
                "query_decomposition": config.knowledge.query_decomposition,
                "query_expansion": config.knowledge.query_expansion,
            },
        })
    except Exception as e:
        _send_error(request_id, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Loop
# ═══════════════════════════════════════════════════════════════════════════════

def _handle_cancel(request_id: str, params: dict) -> None:
    """Cancel an in-progress request."""
    target_id = params.get("target_id", "")
    if not target_id:
        _send_error(request_id, "target_id is required")
        return

    with _active_requests_lock:
        event = _active_requests.get(target_id)
        if event:
            event.set()
            _send_result(request_id, {"cancelled": True, "target_id": target_id})
        else:
            _send_result(request_id, {"cancelled": False, "target_id": target_id, "reason": "not found or already finished"})


HANDLERS = {
    "generate":   _handle_generate,
    "query":      _handle_query,
    "ingest":     _handle_ingest,
    "get_config": _handle_get_config,
    "cancel":     _handle_cancel,
}


def main() -> None:
    # ── Logging ───────────────────────────────────────────────────────────
    bridge_handler = BridgeLogHandler()
    bridge_handler.setLevel(logging.INFO)
    logging.getLogger("casecraft").addHandler(bridge_handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.DEBUG)
    stderr_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logging.getLogger("casecraft").addHandler(stderr_handler)
    logging.getLogger("casecraft").setLevel(logging.DEBUG)

    logger = logging.getLogger("casecraft.bridge")

    # ── Register VS Code LLM proxy callback ──────────────────────────────
    from core.config import config

    if config.general.llm_provider == "vscode":
        from core.llm_client import set_vscode_llm_callback
        set_vscode_llm_callback(vscode_llm_generate)
        logger.info("VS Code LLM proxy enabled – LLM calls will be routed to Copilot models")
    else:
        logger.info("Using native LLM provider: %s", config.general.llm_provider)

    # ── Signal ready ─────────────────────────────────────────────────────
    _write_json({"type": "ready"})

    # ── Read loop ────────────────────────────────────────────────────────
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        # Guard against oversized messages
        if len(line) > MAX_JSON_LINE_SIZE:
            logger.warning("Dropping oversized stdin message (%d bytes)", len(line))
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Reject non-dict messages
        if not isinstance(msg, dict):
            continue

        # LLM response from TypeScript (interleaved with normal requests)
        if msg.get("type") == "llm_response":
            _handle_llm_response(msg)
            continue

        raw_id = msg.get("id", "")
        request_id = _validate_request_id(raw_id) if raw_id else str(uuid.uuid4())

        method = msg.get("method")
        if not isinstance(method, str):
            _send_error(request_id, "method must be a string")
            continue

        params = msg.get("params", {})

        # Ensure params is a dict
        if not isinstance(params, dict):
            _send_error(request_id, "params must be a JSON object")
            continue

        handler = HANDLERS.get(method)
        if handler:
            # Run in a thread so the main loop stays available for
            # llm_response messages (needed for the VS Code LLM proxy).
            thread = threading.Thread(
                target=handler,
                args=(request_id, params),
                daemon=True,
            )
            thread.start()
        else:
            _send_error(request_id, f"Unknown method: {method}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(
            json.dumps({"type": "error", "data": {"message": str(e)}}),
            flush=True,
        )
        sys.exit(1)
