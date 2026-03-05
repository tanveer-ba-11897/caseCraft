# Generate Test Suite

Generate a comprehensive test suite from a requirement document using CaseCraft.

## Instructions

Use the CaseCraft MCP server to generate test cases. Follow these steps:

1. Verify the target file exists and is within an allowed directory (`features/`, `specs/`, or `docs/`).
2. Determine the app type — ask the user or default to the value in `casecraft.yaml`.
3. Call the `generate_tests` MCP tool with the file path and app type.
4. Review the output — report the number of test cases and output file paths.
5. If the user wants, open or summarize the generated test cases.

## Supported File Types

- `.txt` — Plain text requirement documents
- `.pdf` — PDF requirement/specification documents
- `.xlsx` — Excel spreadsheets with requirements
- `.docx` — Word documents with feature descriptions

## App Types

| Type | Description |
|------|-------------|
| `web` | Web application testing (DOM, browser, HTTP) |
| `mobile` | Mobile app testing (gestures, connectivity, OS-specific) |
| `desktop` | Desktop application testing (OS integration, windowing) |
| `api` | API/backend testing (endpoints, payloads, status codes) |

## Example Usage

```
Generate tests for features/login_feature.txt as a web app
```

```
Run test generation on specs/payment_flow.pdf with app_type=api
```

## Notes

- First invocation takes ~30 seconds (ML model lazy loading)
- Rate limited to one call every 5 seconds
- Outputs saved to `outputs/` as JSON + Excel
- LLM backend is configurable: Ollama (local), OpenAI, Google Gemini, GitHub Models (`copilot`), or VS Code Copilot
- For local Ollama: server must be running at localhost:11434
- For GitHub Models: set `GITHUB_TOKEN` env var with a PAT that has `models:read` scope
