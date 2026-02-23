
# AnythingLLM Integration Feasibility Analysis

## What is AnythingLLM?
AnythingLLM is a **full-stack application** (UI + Backend + DB Manager) that turns documents into a chatbot workspace. It is typically designed to **be the platform**, not a library inside another tool.

## Integration Modes

### Mode 1: CaseCraft *inside* AnythingLLM (Most Likely)
*   **Concept**: You use AnythingLLM's UI to upload documents and chat. CaseCraft becomes a "Agent Skill" or "Tool" that AnythingLLM calls.
*   **Feasibility**: **High** (via MCP).
*   **How**: CaseCraft runs as an **MCP Server** (Model Context Protocol). You connect AnythingLLM's "Agent" to the CaseCraft MCP server. When you say "Generate test cases," AnythingLLM delegates the task to CaseCraft.
*   **Pros**: You get AnythingLLM's polished UI for free.
*   **Cons**: Requires AnythingLLM to support MCP (which is currently in beta/development in many tools).

### Mode 2: AnythingLLM *inside* CaseCraft (Not Recommended)
*   **Concept**: CaseCraft calls AnythingLLM's API to store/retrieve vectors instead of managing its own `knowledge_base` folder.
*   **Feasibility**: **Medium**. AnythingLLM has an API (`/api/v1/embeddings`), but it is designed for chat, not raw chunk retrieval for programmatic use.
*   **Pros**: Offloads vector storage.
*   **Cons**: Adds a huge dependency (running a full web app) just to store some vectors. Overkill.

### Mode 3: Parallel Usage (Current Best Path)
*   **Concept**: Use AnythingLLM for *exploring* your docs ("Chat with PDF"). Use CaseCraft CLI for *generating* the test suite.
*   **Integration**: Both tools point to the same **Vector Database** (e.g., Chroma or Pinecone).
*   **Feasibility**: **High**.
*   **Requirement**: We must implement the **Cloud/Pluggable Vector Store** (Phase 3) so CaseCraft can read the same DB that AnythingLLM writes to.

## Recommendation
**Do NOT "integrate" AnythingLLM code directly.**
Instead, build the **MCP Server** (Phase 4). This makes CaseCraft "plug-and-play" compatible with AnythingLLM, Claude Desktop, and other agentic platforms.

**Verdict**: Proceed with **MCP Server** implementation as the bridge.
