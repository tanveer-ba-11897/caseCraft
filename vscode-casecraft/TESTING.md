# CaseCraft VS Code Extension — Testing Guide

## Prerequisites

1. **VS Code** ≥ 1.93.0
2. **GitHub Copilot Chat** extension installed and active subscription
3. **Python 3.10+** with CaseCraft dependencies installed:
   ```bash
   pip install -r requirements-runtime.txt
   ```

## Quick Start: Extension Development Host

1. Open `vscode-casecraft/` as a workspace in VS Code
2. Press **F5** (or Run → Start Debugging)
3. Select **"Run CaseCraft Extension"**
4. A new VS Code window opens with the extension loaded

## Test Checklist

### 1. Activation Test
- [ ] Extension loads — look for the welcome notification:
  _"CaseCraft extension activated! Type @casecraft in Copilot Chat to get started."_
- [ ] Click "Open Chat" to verify it opens Copilot Chat
- [ ] Check **Output** → **CaseCraft** channel shows:
  `[CaseCraft] Python bridge is ready`

### 2. `/config` Command
- [ ] Type `@casecraft /config` in chat
- [ ] Should show JSON config with general, generation, quality, knowledge sections
- [ ] Verify `llm_provider` shows `"vscode"` (when using Copilot mode)

### 3. `/query` Command
- [ ] Type `@casecraft /query file encryption` in chat
- [ ] Should return knowledge base search results with sources
- [ ] Try with no query: `@casecraft /query` — should show usage hint

### 4. `/generate` Command  _(main test)_
- [ ] Type `@casecraft /generate features/File_Encryption.pdf` in chat
- [ ] Watch progress messages (loading, chunking, generating…)
- [ ] Result should show test case count + preview table
- [ ] "Open JSON" and "Open Excel" buttons should open the output files
- [ ] Try with `#file` reference: type `@casecraft /generate` then attach the file

### 5. `/ingest` Command
- [ ] `@casecraft /ingest docs knowledge_base/Web_help_Doc/`
- [ ] Should show documents processed, chunks created, total index size

### 6. Freeform Chat
- [ ] Type `@casecraft what are good test strategies for login forms?`
- [ ] Should use KB context + Copilot model to answer

### 7. Cancellation
- [ ] Start a `/generate` command, then click the stop button in chat
- [ ] Should cancel gracefully without hanging

### 8. Error Cases
- [ ] `@casecraft /generate nonexistent.pdf` — should show file not found error
- [ ] Set `casecraft.pythonPath` to invalid path, reload → should show error message
- [ ] Check "CaseCraft" output channel for diagnostic logs

## Settings to Verify

Open **Settings** (Ctrl+,) and search "casecraft":

| Setting | Default | Purpose |
|---------|---------|---------|
| `casecraft.llmProvider` | `copilot` | Should use Copilot models |
| `casecraft.model` | `gpt-4o` | Try `claude-3.5-sonnet` too |
| `casecraft.pythonPath` | `python` | Path to your Python |
| `casecraft.appType` | `web` | `web`, `mobile`, `desktop`, `api` |
| `casecraft.reviewerPass` | `false` | Set true for extra quality |

## Troubleshooting

- **"Python not found"**: Set `casecraft.pythonPath` in settings
- **"No Copilot models available"**: Ensure GitHub Copilot is authorized
- **Bridge keeps restarting**: Check **Output → CaseCraft** for Python errors
- **Model timeout**: Long generation may hit 30-min timeout; reduce doc size or config
