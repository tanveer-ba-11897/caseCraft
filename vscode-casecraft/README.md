# CaseCraft — AI Test Case Generator

Generate comprehensive test suites from feature documents using RAG and Copilot models, directly inside VS Code.

## Features

- **@casecraft** chat participant in GitHub Copilot Chat
- `/generate` — Generate test cases from PDF, TXT, or MD feature documents
- `/query` — Search the product knowledge base
- `/ingest` — Ingest documents or URLs into the knowledge base
- `/config` — View current configuration
- Freeform QA chat with RAG context

## Requirements

- VS Code ≥ 1.93.0
- GitHub Copilot Chat extension
- Python 3.10+ with CaseCraft dependencies

## Getting Started

1. Install the extension
2. Open a workspace containing your CaseCraft project
3. Open Copilot Chat and type `@casecraft /generate features/your_file.pdf`

## Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `casecraft.llmProvider` | `copilot` | LLM backend (copilot, ollama, openai, google) |
| `casecraft.model` | `gpt-4o` | Model identifier |
| `casecraft.pythonPath` | `python` | Path to Python interpreter |
| `casecraft.appType` | `web` | Application type (web, mobile, desktop, api) |
| `casecraft.reviewerPass` | `false` | Enable AI reviewer for quality |

## Commands

- **CaseCraft: Show Output** — Show the diagnostic output channel
