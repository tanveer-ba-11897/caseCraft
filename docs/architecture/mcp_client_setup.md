
# MCP Configuration Guide: AnythingLLM & Claude Desktop

## Overview

CaseCraft runs as an **MCP Server** on your local machine using the `casecraft_mcp.py` script. The "Client" (AnythingLLM or Claude Desktop) simply needs to know **where that script is**.

---

## 1. AnythingLLM Setup

> ⚠️ **You MUST use an Agent workspace, not a regular Chat workspace.**
> In Chat mode, the LLM answers directly and **never** calls MCP tools.
> In Agent mode, the LLM can see and invoke CaseCraft's tools.

### Step 1: Add the MCP Server

**Option A: GUI (Easiest)**

1. Open AnythingLLM > **Settings** (gear icon).
2. Go to **Agent Skills** > **MCP Servers**.
3. Click **Add MCP Server**.
4. Fill in:
    - **Name**: `CaseCraft`
    - **Command**: `python`
    - **Arguments**: `c:\Users\tanve\casecraft\casecraft_mcp.py`
5. Click **Save**. You should see `generate_tests` and `query_knowledge` listed as tools.

**Option B: JSON (Advanced)**

1. Navigate to `%APPDATA%\anythingllm-desktop\storage\plugins\`.
2. Create or edit `anythingllm_mcp_servers.json`:

```json
{
  "mcpServers": {
    "casecraft": {
      "command": "python",
      "args": ["c:\\Users\\tanve\\casecraft\\casecraft_mcp.py"],
      "env": {}
    }
  }
}
```

1. Restart AnythingLLM.

### Step 2: Create an Agent Workspace (Critical!)

1. Click **+ New Workspace**.
2. Name it (e.g., "QA Agent").
3. Open the workspace settings (gear icon on the workspace).
4. Change **Chat mode** from `Chat` to **`Agent`**.
5. Under **Agent Configuration**:
    - Ensure CaseCraft tools (`generate_tests`, `query_knowledge`) are **enabled/checked**.
    - Set the LLM model for the agent (e.g., `llama3.2:3b` via Ollama).
6. **Save** settings.

### Step 3: Use the Right Prompt

In the Agent workspace, ask:
> "Use the generate_tests tool to generate test cases for the file `examples/sample.pdf`"

**Tips for triggering tool calls:**

- Explicitly mention "use the generate_tests tool" in your prompt.
- If the LLM still answers directly, try: "Call the CaseCraft generate_tests function with file_path examples/sample.pdf".
- Check the response for a "Tool Call" indicator — this confirms MCP was used.

**Note**: AnythingLLM's MCP support is evolving. If it fails, check logs in `My Documents/AnythingLLM/logs`.

---

## 2. Claude Desktop

**How to Connect:**

1. Locate `claude_desktop_config.json`:
    - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
    - (Typically `C:\Users\tanve\AppData\Roaming\Claude\claude_desktop_config.json`)
2. Add this JSON snippet:

```json
{
  "mcpServers": {
    "casecraft": {
      "command": "python",
      "args": [
        "c:\\Users\\tanve\\casecraft\\casecraft_mcp.py"
      ]
    }
  }
}
```

1. Restart Claude Desktop. The 🔨 icon will appear in chat.

---

## 3. Switching Between Them

**Zero Code Changes Required.**
You can have **both** configured at the same time. The CaseCraft server doesn't care who calls it.

- Want to use AnythingLLM? Open AnythingLLM.
- Want to use Claude? Open Claude.
- Want to switch? Just close one app and open the other.

**Tip**: The only difference is the *Client Configuration* (JSON file vs GUI settings). The server logic `casecraft_mcp.py` remains identical.

---

## 4. Knowledge Base Maintenance

Currently, the MCP Server is **Read-Only** for generation and querying. To manage the Knowledge Base (add/remove docs), you must use the **CLI**.

### Ingesting New Docs

To add new documentation to the RAG index from your terminal:

```bash
# Add a Sitemap
python -m cli.ingest sitemap https://docs.example.com/sitemap.xml

# Add specific PDF/Text folder
python -m cli.ingest docs ./new-specs/
```

### Clearing the Knowledge Base

To wipe the index and start fresh (e.g., for a new project):

1. **Stop the MCP Server** (Close Claude/AnythingLLM or stop the python process).
2. **Delete the Index Files**:

    ```bash
    # Windows Command Prompt
    del knowledge_base\index.json
    del knowledge_base\index.json.sha256
    ```

3. **Re-ingest** your new documents.
4. **Restart** your AI Client.

---

## 5. Usage: Generating Test Cases

> ⚠️ **Do NOT paste full PRD/spec documents into the chat window.**

When you paste a large document directly into the chat, the MCP client (AnythingLLM/Claude) sends the entire text to its LLM in a single prompt. This causes Ollama to request massive memory:

```
model requires more system memory (15.9 GiB) than is available (9.1 GiB)
```

**✅ Correct Workflow:**

1. Save your PRD/spec file in the `examples/` folder (or `specs/` or `docs/`).
2. In the chat, ask:
    > "Generate test cases for the file `examples/my_feature.pdf`"
3. The agent will call CaseCraft's `generate_tests` tool, which reads and chunks the file internally — no memory issues.

---

## 6. Troubleshooting

| Error | Cause | Fix |
|:------|:------|:----|
| **Error -32000: Connection closed** | Server takes too long to start | CaseCraft uses lazy imports — ensure you're on the latest `server.py`. |
| **Model requires more system memory** | Context window too large for RAM | Set `context_window_ratio: 0.5` in `casecraft.yaml` to use only 50% of the native window. |
| **Memory error after config change** | Large doc pasted into chat | Don't paste docs — save as file in `examples/` and reference by path. |
| **Server not found** | Python path wrong | Ensure `python` points to the venv with deps installed. Use full path if needed. |
| **Error -32001: Request Timeout** | Generation took too long (>60s) | 1. Use a faster local model (`llama3.2:3b`).<br>2. Retry (first run is slower due to model loading).<br>3. Split large PDF into smaller files. |
