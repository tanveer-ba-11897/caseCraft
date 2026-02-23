# How CaseCraft Becomes an MCP Server

## What is MCP?
Think of **MCP (Model Context Protocol)** like a **USB standard for AI**.
*   **Without MCP**: If you want Claude to use Google Drive, someone must write a "Google Drive Plugin". If you want it to use Linear, someone writes a "Linear Plugin".
*   **With MCP**: You write a "Server" that says: *"I have these files"* and *"I can do these actions"*. Any AI app (Claude, AnythingLLM, Cursor) can then connect to it and use those files and actions instantly.

## How CaseCraft Works Today
Currently, CaseCraft is a **CLI Tool**:
1.  You run `python main.py generate my_doc.pdf`.
2.  It reads the PDF -> Chunks it -> Calls LLM -> Saves Excel file.
3.  The "Intelligence" is trapped inside your terminal.

## How CaseCraft Works as an MCP Server
We will wrap the core functions (`generator.py`, `retriever.py`) in a lightweight server layer using the `mcp-python` SDK.

### The Architecture
1.  **The Server**: A script `mcp_server/main.py`.
2.  **Resources (Read-Only Data)**:
    *   CaseCraft exposes your `knowledge_base` as **Resources**.
    *   *AI Query*: "Read the 'Login Specification' from CaseCraft knowledge."
    *   *Result*: The server returns the text content of that document.
3.  **Tools (Actions)**:
    *   CaseCraft exposes `generate_test_suite` as a **Tool**.
    *   *AI Query*: "Generate test cases for the Login feature."
    *   *Result*: The server runs the generation logic and returns the JSON test suite to the AI.

### Steps to Build It
1.  **Install SDK**: `pip install mcp`
2.  **Define Server**: Create an `McpServer` instance.
3.  **Register Tool**:
    ```python
    @server.tool()
    def generate_tests(feature_description: str, app_type: str = "web") -> str:
        # Call existing internal logic
        suite = core.generator.generate_test_suite_from_text(feature_description, ...)
        return suite.json()
    ```
4.  **Run It**: The server runs on `stdio` (standard input/output), meaning the AI app launches it as a subprocess and talks to it via JSON-RPC.

## User Workflow (The "Magic")
1.  You open **Claude Desktop** (or AnythingLLM).
2.  You go to settings and add an MCP Server: `python c:/casecraft/mcp_server/main.py`.
3.  You start a chat: *"Here is a screenshot of a login page. Use CaseCraft to generate test cases for it."*
4.  **Claude**:
    *   Analyzes the screenshot.
    *   Calls `casecraft.generate_tests(description="Login page with 2FA")`.
    *   Receives the JSON test cases.
    *   Shows you a beautiful table of test cases in the chat.

**Impact**: CaseCraft transforms from a "Script you run" into a "Skill your AI has".
