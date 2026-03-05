# CaseCraft — Copilot Instructions

## Project Overview

CaseCraft is a Python 3.10+ agentic QA engine that generates comprehensive test suites from
requirement documents (`.txt`, `.pdf`, `.xlsx`, `.docx`). It supports multiple LLM backends — 
Ollama (local), OpenAI-compatible APIs, Google Gemini, and GitHub Models (`copilot` provider) — 
with a ChromaDB-backed RAG knowledge base and a networkx knowledge graph
for cross-document reasoning.

## Architecture

- **CLI**: `cli/main.py` — primary entry point (`python -m cli.main`)
- **Ingestion CLI**: `cli/ingest.py` — ingests local docs, URLs, and sitemaps into the RAG knowledge base
- **MCP Server**: `casecraft_mcp.py` → `mcp_server/server.py` — exposes `generate_tests` and `query_knowledge` tools via stdio
- **Core Engine**: `core/generator.py` — chunking → condensation → LLM generation → deduplication → reviewer pass
- **RAG Pipeline**: `core/knowledge/retriever.py` — 7-stage hybrid retrieval (ChromaDB HNSW + BM25 + CrossEncoder re-rank + parent-child + knowledge graph)
- **Knowledge Graph**: `core/knowledge/graph.py` — networkx directed graph with 4 relation types (parent_of, same_source, cross_reference, shared_entity)
- **Vector Store**: `core/knowledge/vector_store.py` — ChromaDB persistent HNSW with cosine indexing
- **Cache**: `core/cache.py` — thread-safe LRU caches for condensation results, retrieval results, and rendered prompt templates; BM25 disk persistence
- **LLM Client**: `core/llm_client.py` — unified adapter for Ollama, OpenAI, Google, and Copilot (GitHub Models) backends
- **Prompt Templates**: `prompts/templates/` — Jinja2 templates for generation, condensation, reviewer, and cross-reference passes
- **Config**: `casecraft.yaml` — all settings (Pydantic v2 validated, env var overrides via `CASECRAFT_SECTION_KEY`)
- **Schema**: `core/schema.py` — Pydantic models for `TestCase` and `TestSuite`

## Key Commands

```bash
# Generate test suite from a feature file
python -m cli.main features/your_feature.txt --format excel

# Ingest documents into the knowledge base
python -m cli.ingest docs path/to/documents/

# Run tests
python -m pytest tests/ -v

# Start MCP server (for Copilot CLI / Claude Desktop / AnythingLLM)
python casecraft_mcp.py
```

## MCP Tools Available

1. **`generate_tests(file_path, app_type?)`** — Generates test suite from files in `features/`, `specs/`, or `docs/` directories. Returns test case count and output file paths.
2. **`query_knowledge(query, top_k?)`** — Searches the ChromaDB knowledge base using 7-stage hybrid retrieval. Returns relevant document snippets with source metadata.

## LLM Providers

| Provider | Config `llm_provider` | Notes |
|---|---|---|
| Ollama | `"ollama"` | Local models, no API key needed. Default. |
| OpenAI-compatible | `"openai"` | Works with OpenAI, LM Studio, vLLM, LocalAI, Azure. |
| Google Gemini | `"google"` | REST API, requires Google API key. |
| GitHub Models | `"copilot"` | Uses `GITHUB_TOKEN` env var (PAT with `models:read` scope). Access GPT-4o, o3-mini, DeepSeek-R1, Llama, Phi-4, Grok, etc. |

## Test Case Schema

Each generated test case follows `core/schema.py`:
- `use_case`, `test_case` (name), `test_type` (functionality/ui/performance/integration/usability/database/security/acceptance)
- `preconditions`, `test_data` (key-value pairs), `steps`, `priority` (high/medium/low)
- `dependencies`, `tags`, `expected_results`, `actual_results`

## Prompt Templates

Located in `prompts/templates/`:
- `generation.j2` — Main test generation prompt with app-type-specific sections (mobile, desktop, api)
- `condensation.j2` — Extracts testable facts from document chunks
- `reviewer.j2` — Reviews and polishes existing test suites (clarity, consistency)
- `checklist_cross_reference.j2` — Cross-references test suite against a checklist for missing coverage

All templates include anti-injection boundaries to prevent prompt injection from user-supplied documents.

## Conventions

- Feature/requirement files go in `features/`, `specs/`, or `docs/` directories
- Output files are saved to `outputs/` as both JSON and Excel
- Configuration is in `casecraft.yaml` (Pydantic v2 validated)
- All heavy ML models load lazily on first tool call (~30s startup)
- The knowledge base uses ChromaDB with HNSW indexing (cosine similarity)
- The knowledge graph is built at ingest time and persisted as JSON
- Tests use pytest with 139+ test cases across multiple test files
- Type hints throughout, Pyright-compliant

## Security

- Path sandboxing: files must be in `features/`, `specs/`, or `docs/`
- Prompt-injection fencing (regex heuristics) in all Jinja2 templates
- Index integrity hashing (SHA-256) for tamper detection
- Input size limits (10MB request, 5MB response, 500K text)
- API keys excluded from config dumps and diagnostics
- Rate limiting: 5-second cooldown between `generate_tests` MCP calls

## Important Notes

- For local Ollama: server must be running at `http://localhost:11434`
- For GitHub Models (`copilot` provider): set `GITHUB_TOKEN` env var with a PAT
- The default model is configurable in `casecraft.yaml` under `general.model`
- File paths are sandboxed to `features/`, `specs/`, and `docs/` for security
- App type (`web`, `mobile`, `desktop`, `api`) controls prompt-specific test scenarios
- Rate-limit mitigation for cloud providers: `max_workers: 1` (sequential), `llm_call_delay` (throttle between calls)
- Pipeline caching: condensation cache (skips redundant LLM calls), retrieval cache (skips redundant hybrid search), BM25 disk persistence (skips rebuild on startup). Configured in `casecraft.yaml` under `cache` section.
