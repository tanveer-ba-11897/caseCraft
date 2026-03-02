# CaseCraft as a VS Code Chat Participant Extension — Implementation Plan

## Overview

This document details the plan to turn CaseCraft into a native **`@casecraft` chat participant** inside VS Code Copilot Chat, giving it a dedicated identity, custom slash commands, and the ability to use **VS Code Copilot's own models** (GPT-4o, Claude 3.5 Sonnet, o1, Gemini) instead of self-hosted Ollama.

---

## Feasibility Summary

### Path 1: MCP Server (Already Done — Zero Effort)

CaseCraft's existing MCP server (`mcp_server/server.py`) can be used directly inside Copilot Chat with **no code changes**. Just add the MCP config to VS Code's `settings.json`:

```json
"mcp": {
  "servers": {
    "casecraft": {
      "command": "python",
      "args": ["c:\\Users\\tanveer-11897\\casecraft\\casecraft_mcp.py"],
      "type": "stdio"
    }
  }
}
```

### Path 2: Chat Participant Extension (This Plan)

A deeper integration where CaseCraft becomes a **`@casecraft` chat participant** in Copilot Chat — with custom slash commands, streaming progress, and the ability to use Copilot's built-in models.

### Path 3: GitHub Copilot Extension (Cloud/SaaS)

Host CaseCraft as a cloud service with OAuth. Higher effort, misaligned with local-first philosophy. Only relevant for SaaS distribution.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│  VS Code Extension (TypeScript)   — "vscode-casecraft"       │
│                                                              │
│  extension.ts                                                │
│  ├─ Registers @casecraft chat participant                    │
│  ├─ Defines slash commands: /generate, /query, /ingest       │
│  ├─ Request handler:                                         │
│  │   1. Reads user prompt & referenced files                 │
│  │   2. Optionally calls vscode.lm API (Copilot models)     │
│  │   3. Spawns Python subprocess for core pipeline           │
│  │   4. Streams results back into chat                       │
│  │   5. Opens generated files in editor                      │
│  └─ Settings UI (contributes configuration)                  │
│                                                              │
│  pythonBridge.ts                                             │
│  ├─ Manages Python child process lifecycle                   │
│  ├─ JSON-RPC or line-delimited JSON protocol                 │
│  └─ Handles progress events, errors, cancellation            │
│                                                              │
│  lmBridge.ts  (optional — Phase 2)                           │
│  ├─ Wraps vscode.lm.selectChatModels() API                  │
│  ├─ Implements streaming + JSON mode                         │
│  └─ Falls back to Ollama/OpenAI if no Copilot subscription   │
├──────────────────────────────────────────────────────────────┤
│  Python Backend  — existing casecraft core/ (unchanged)      │
│                                                              │
│  bridge_server.py  (NEW — thin JSON-RPC wrapper)             │
│  ├─ Reads JSON commands from stdin                           │
│  ├─ Dispatches to:                                           │
│  │   generate_test_suite()  → returns TestSuite JSON         │
│  │   query_knowledge()      → returns chunks                 │
│  │   ingest_documents()     → returns status                 │
│  │   condense_chunk()       → returns condensed text         │
│  ├─ Emits progress events to stdout                          │
│  └─ Uses existing llm_client.py OR accepts LLM responses     │
│       from TypeScript (when using Copilot models)            │
└──────────────────────────────────────────────────────────────┘
```

---

## How the Two Processes Communicate

The TypeScript extension spawns a single long-lived Python process (`bridge_server.py`). They communicate via **line-delimited JSON over stdin/stdout** (the same pattern used by LSP and MCP servers):

```
TypeScript → Python (stdin):
  {"id": 1, "method": "generate", "params": {"file_path": "features/login.pdf", "app_type": "web"}}

Python → TypeScript (stdout):
  {"id": 1, "type": "progress", "data": {"step": "parsing", "detail": "3 chunks extracted"}}
  {"id": 1, "type": "progress", "data": {"step": "retrieving", "detail": "12 KB chunks found"}}
  {"id": 1, "type": "progress", "data": {"step": "generating", "detail": "Chunk 1/3 complete"}}
  {"id": 1, "type": "result", "data": {"test_cases": [...], "json_path": "outputs/login.json", "excel_path": "outputs/login.xlsx"}}
```

This lets the TypeScript side stream real-time progress into the Copilot Chat response.

---

## Two LLM Modes

The extension supports **dual LLM backends**:

| Mode | How It Works | When Used |
|---|---|---|
| **Self-hosted (existing)** | Python calls Ollama/OpenAI/Google via `llm_client.py` as it does today | User has no Copilot subscription, or prefers local models |
| **Copilot models (new)** | TypeScript calls `vscode.lm.selectChatModels()`, sends the prompt, gets the response, and passes it back to Python for post-processing | User has Copilot subscription and wants GPT-4o / Claude Sonnet quality |

In Copilot-model mode, the flow changes slightly:

1. Python builds the prompt (Jinja2 template + RAG context) → sends it to TypeScript
2. TypeScript sends the prompt to the VS Code LM API → gets the LLM response
3. TypeScript sends the raw LLM response back to Python
4. Python runs the existing post-processing pipeline (JSON sanitization, normalization, dedup, etc.)

This keeps **all prompt engineering and post-processing in Python** (no duplication) while leveraging Copilot's models.

---

## Slash Commands

| Command | Action | Maps To |
|---|---|---|
| `@casecraft /generate` | Generate test suite from a file or selected text | `generate_test_suite()` in `core/generator.py` |
| `@casecraft /query` | Search the knowledge base | `KnowledgeRetriever.retrieve()` in `core/knowledge/retriever.py` |
| `@casecraft /ingest` | Ingest documents or URLs into the KB | `cli/ingest.py` logic |
| `@casecraft /config` | Show/change current configuration | Read/write `casecraft.yaml` |
| `@casecraft` (no command) | Freeform QA chat — answer questions about testing, the project, etc. | Prompt + RAG context sent to LLM |

---

## What Changes vs. What Stays

| Component | Change Required |
|---|---|
| `core/generator.py` | **No change** — called as-is from `bridge_server.py` |
| `core/parser.py` | **No change** |
| `core/prompts.py` + templates | **No change** |
| `core/schema.py` | **No change** |
| `core/exporter.py` | **No change** |
| `core/knowledge/` (all RAG modules) | **No change** |
| `core/config.py` | **Minor tweak** — accept config overrides from JSON-RPC params so the extension can pass settings from VS Code's settings UI |
| `core/llm_client.py` | **Add a new mode**: `"vscode"` provider that doesn't call any HTTP endpoint — instead returns the prompt to the bridge and accepts the LLM response back. This is ~40 lines of code. |
| `mcp_server/server.py` | **No change** — stays as-is for Claude Desktop / Other MCP clients |
| `cli/main.py` | **No change** — stays as-is for terminal usage |

### New Files to Create

| File | Purpose |
|---|---|
| `vscode-casecraft/package.json` | Extension manifest (participant, commands, config) |
| `vscode-casecraft/src/extension.ts` | Main entry: register participant, handle commands |
| `vscode-casecraft/src/participant.ts` | Chat participant handler logic |
| `vscode-casecraft/src/pythonBridge.ts` | Spawn & manage Python subprocess, JSON protocol |
| `vscode-casecraft/src/lmBridge.ts` | Wrapper around `vscode.lm` API for Copilot models |
| `vscode-casecraft/src/config.ts` | Read VS Code settings, map to CaseCraft config |
| `vscode-casecraft/src/outputPanel.ts` | Open generated files in editor, render summary |
| `bridge_server.py` (project root) | Python JSON-RPC bridge for the extension |

---

## Detailed Implementation Plan

### Phase 0: Prerequisite Setup (1–2 days)

| # | Task | Details |
|---|---|---|
| 0.1 | **Scaffold the VS Code extension** | Run `yo code` to generate a TypeScript extension. Set `engines.vscode` to `^1.93.0` (minimum for stable `lm` API). Add `"chatParticipants"` to `package.json`. |
| 0.2 | **Configure the dev environment** | Set up `tsconfig.json`, ESLint, webpack/esbuild bundler. Add `@types/vscode` as dev dependency. |
| 0.3 | **Verify Python availability** | Write a utility function that locates the Python interpreter (check `python3`, `python`, or the VS Code Python extension's selected interpreter via `ms-python.python` API). |
| 0.4 | **Add integration test harness** | Set up `@vscode/test-electron` for running extension integration tests. |

### Phase 1: Python Bridge + Basic `/generate` (3–5 days)

| # | Task | Details |
|---|---|---|
| 1.1 | **Create `bridge_server.py`** | A stdin/stdout JSON-RPC server that wraps `generate_test_suite()`, `query_knowledge()`, and `ingest` as methods. Emits progress events as JSON lines. Handles graceful shutdown on EOF. |
| 1.2 | **Create `pythonBridge.ts`** | Spawns `python bridge_server.py` as a child process. Implements `sendRequest(method, params)` → `Promise<result>`. Handles progress callbacks, error parsing, and process restart on crash. |
| 1.3 | **Register the `@casecraft` chat participant** | In `package.json`, declare `"chatParticipants": [{"id": "casecraft.agent", "name": "casecraft", "description": "...", "commands": [...]}]`. In `extension.ts`, call `vscode.chat.createChatParticipant()` and wire up the request handler. |
| 1.4 | **Implement `/generate` command handler** | Parse the user's prompt for a file path (or use `ChatRequest.references` for `#file` references). Call the Python bridge. Stream progress messages into the chat response via `ChatResponseStream.progress()`. Render the final test suite as a markdown table in the chat. |
| 1.5 | **Implement file opening** | After generation, offer buttons to open the JSON/Excel output files in VS Code using `ChatResponseStream.button()` → `vscode.commands.executeCommand('vscode.open', uri)`. |
| 1.6 | **Error handling & timeout** | Handle Python process crashes, timeout scenarios, and invalid file paths. Show user-friendly error messages in the chat. |
| 1.7 | **Unit tests for bridge protocol** | Test the JSON line parsing, request/response matching, and error cases in isolation (no VS Code needed). |

### Phase 2: Copilot LM Integration (3–4 days)

| # | Task | Details |
|---|---|---|
| 2.1 | **Create `lmBridge.ts`** | Wraps `vscode.lm.selectChatModels()`. Exposes an async `generate(prompt, options)` function. Handles streaming token collection and JSON mode. |
| 2.2 | **Add `"vscode"` provider to `llm_client.py`** | When `llm_provider: "vscode"`, the `generate()` method doesn't make an HTTP call. Instead, it sends `{"type": "llm_request", "prompt": "...", "json_mode": true}` to stdout and blocks reading stdin for the response. |
| 2.3 | **Wire the proxy in `pythonBridge.ts`** | When the Python process emits an `llm_request`, the TypeScript side intercepts it, calls `lmBridge.generate()`, and writes the response back to Python's stdin. |
| 2.4 | **Model selection UI** | Add a VS Code setting `casecraft.llmProvider` with enum `["ollama", "openai", "google", "copilot"]`. When `"copilot"`, add a model picker setting `casecraft.copilotModel` listing available models from `vscode.lm.selectChatModels()`. |
| 2.5 | **Fallback logic** | If the user selects `"copilot"` but has no Copilot subscription, detect the error from the LM API and fall back to the configured Ollama/OpenAI backend with a warning message. |
| 2.6 | **Test with each Copilot model** | Verify that GPT-4o, Claude Sonnet, and Gemini Flash all produce valid JSON that passes through CaseCraft's post-processing pipeline. |

### Phase 3: Additional Commands & UX (2–3 days)

| # | Task | Details |
|---|---|---|
| 3.1 | **Implement `/query` command** | Takes a natural language question, calls `bridge_server.py` → `KnowledgeRetriever.retrieve()`, renders results as formatted markdown with source citations. |
| 3.2 | **Implement `/ingest` command** | Accepts a folder path or URL. Calls the ingestion pipeline. Shows progress (pages crawled, chunks created). |
| 3.3 | **Implement `/config` command** | Shows current config in a nicely formatted markdown block. Supports inline overrides like `@casecraft /config app_type=mobile`. |
| 3.4 | **Freeform chat mode** | When no slash command is used, treat the message as a QA question. Retrieve RAG context, combine with the question, send to LLM, stream the answer. This makes `@casecraft` a knowledgeable QA assistant. |
| 3.5 | **`#file` reference support** | Support VS Code's `#file:path` syntax so users can drag files into the chat: `@casecraft /generate #file:features/login.pdf`. Extract the file URI from `ChatRequest.references`. |
| 3.6 | **Follow-up suggestions** | After generating tests, use `ChatResponseStream.markdown()` to suggest follow-ups: *"Would you like to /query the knowledge base for related features?"* or *"Run /generate with --reviewer for a quality pass?"* |

### Phase 4: Configuration & Settings UI (1–2 days)

| # | Task | Details |
|---|---|---|
| 4.1 | **Contribute VS Code settings** | In `package.json`'s `contributes.configuration`, define all CaseCraft settings matching `casecraft.yaml`: `casecraft.model`, `casecraft.llmProvider`, `casecraft.baseUrl`, `casecraft.chunkSize`, `casecraft.appType`, `casecraft.semanticDedup`, `casecraft.reviewerPass`, etc. |
| 4.2 | **Settings → config bridge** | In `config.ts`, read VS Code settings and convert them to a config JSON blob that gets passed to `bridge_server.py` on each request. This overrides `casecraft.yaml` when the extension is active. |
| 4.3 | **Welcome / onboarding** | On first activation, show a walkthrough page that helps the user: (a) select their Python interpreter, (b) install dependencies, (c) choose LLM provider. |

### Phase 5: Polish & Packaging (2–3 days)

| # | Task | Details |
|---|---|---|
| 5.1 | **Bundling** | Use esbuild to bundle the TypeScript into a single `extension.js`. Configure `.vscodeignore` to exclude source maps, test files, and the Python `__pycache__`. |
| 5.2 | **Python dependency check** | On activation, verify that `requirements-runtime.txt` packages are installed. If not, show a notification with an "Install" button that runs `pip install -r requirements-runtime.txt`. |
| 5.3 | **Extension icon & branding** | Create an icon, write the `README.md` for the marketplace, add screenshots. |
| 5.4 | **Integration tests** | End-to-end tests: activate extension → send `/generate` command → verify output files created → verify chat response contains test cases. |
| 5.5 | **Package as VSIX** | Run `vsce package` to create a `.vsix` file for local installation or marketplace publishing. |
| 5.6 | **Handle RAG embeddings dependency** | The heaviest dependency is `sentence-transformers` (requires PyTorch). For lightweight installs, add an option to use `vscode.lm.computeEmbedding()` (preview API) as an alternative embedder — this eliminates the torch dependency entirely. Make this configurable: `casecraft.embeddingProvider: "local" | "copilot"`. |

---

## Key `package.json` Structure (Reference)

```jsonc
{
  "name": "vscode-casecraft",
  "displayName": "CaseCraft – AI Test Case Generator",
  "version": "0.1.0",
  "engines": { "vscode": "^1.93.0" },
  "categories": ["Chat"],
  "extensionDependencies": ["github.copilot-chat"],
  "activationEvents": [],
  "main": "./dist/extension.js",
  "contributes": {
    "chatParticipants": [{
      "id": "casecraft.agent",
      "fullName": "CaseCraft",
      "name": "casecraft",
      "description": "Generate comprehensive test suites from feature documents using RAG",
      "isSticky": true,
      "commands": [
        { "name": "generate", "description": "Generate test cases from a feature document" },
        { "name": "query",    "description": "Search the product knowledge base" },
        { "name": "ingest",   "description": "Ingest documents or URLs into the knowledge base" },
        { "name": "config",   "description": "View or change CaseCraft configuration" }
      ]
    }],
    "configuration": {
      "title": "CaseCraft",
      "properties": {
        "casecraft.llmProvider": {
          "type": "string",
          "enum": ["copilot", "ollama", "openai", "google"],
          "default": "copilot",
          "description": "LLM backend to use for test generation"
        },
        "casecraft.model": {
          "type": "string",
          "default": "gpt-4o",
          "description": "Model identifier"
        },
        "casecraft.appType": {
          "type": "string",
          "enum": ["web", "mobile", "desktop", "api"],
          "default": "web"
        },
        "casecraft.pythonPath": {
          "type": "string",
          "default": "python",
          "description": "Path to Python interpreter"
        }
      }
    }
  }
}
```

---

## Models Available in VS Code Copilot

The `vscode.lm` API provides access to these models (no API keys required with a Copilot subscription):

- **GPT-4o** / **GPT-4o-mini** (OpenAI)
- **Claude 3.5 Sonnet** / **Claude 3.5 Haiku** (Anthropic)
- **o1** / **o3-mini** (OpenAI reasoning models)
- **Gemini 2.0 Flash** (Google)

All of these are **substantially better** at structured JSON generation than the local 3B–8B models CaseCraft currently targets. The existing Jinja2 prompts and Pydantic validation would work as-is — the output quality would simply improve.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Python process crashes mid-generation | Medium | Auto-restart logic in `pythonBridge.ts` + heartbeat mechanism |
| `vscode.lm` API doesn't support JSON mode natively | Medium | Use CaseCraft's existing `_clean_json_output()` sanitizer — it already handles non-JSON LLM output |
| `sentence-transformers` install fails (Windows + PyTorch) | Medium | Offer `copilot` embedding fallback via `vscode.lm.computeEmbedding()` |
| User has no Copilot subscription | Low | Graceful fallback to Ollama with a clear notification |
| Extension size too large if bundling Python deps | N/A | Python deps are **not** bundled with the extension — they live in the user's Python environment. The extension only bundles the TypeScript code (~100KB). |

---

## Estimated Timeline

| Phase | Duration | Cumulative |
|---|---|---|
| Phase 0: Setup | 1–2 days | 2 days |
| Phase 1: Bridge + `/generate` | 3–5 days | 7 days |
| Phase 2: Copilot LM Integration | 3–4 days | 11 days |
| Phase 3: Additional Commands | 2–3 days | 14 days |
| Phase 4: Settings UI | 1–2 days | 16 days |
| Phase 5: Polish & Packaging | 2–3 days | **~19 days** |

Roughly **3–4 weeks** for a single developer, or **2 weeks** with a pair. Phase 1 alone gives a working `@casecraft /generate` command, so you get value within the first week.

---

## Bottom Line

The entire Python core stays untouched. The only Python addition is a ~150-line `bridge_server.py` JSON-RPC wrapper, and the only modification to existing code is a small `"vscode"` provider option in `core/llm_client.py` (~40 lines) for the Copilot model proxy. Everything else is new TypeScript code in a separate `vscode-casecraft/` directory.
