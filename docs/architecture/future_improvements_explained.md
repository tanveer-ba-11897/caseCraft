What and How they are useful for CaseCraft:

## 1. Cloud Vector DB (RAG in Cloud)
**What is it?**
Currently, CaseCraft stores its knowledge base (indexed documents) in a local file (`index.json`) or a local folder (`chroma/`). Moving to the cloud means using a managed service like **Pinecone**, **Weaviate**, or **Qdrant** to store these vector embeddings.

**How is it useful for CaseCraft?**
*   **Scalability**: Local files get slow and heavy if you index thousands of pages (e.g., massive legacy documentation). Cloud DBs handle millions of vectors instantly.
*   **Team Collaboration**: If you run CaseCraft on your machine, your `index.json` is verified only by *you*. Using a Cloud DB means the whole team shares the exact same "Brain". If one person ingests a new feature spec, everyone's CaseCraft instance immediately knows about it.
*   **Persistence**: You don't risk corrupting the index file or losing it if you clear your temp folders. Use "namespaces" to manage multiple projects without file clashes.
*   **Hybrid Search**: Cloud providers often offer better search algorithms (keyword + vector hybrid) than simple local cosine similarity, leading to more accurate test case generation.

## 2. MCP Server (Model Context Protocol)
**What is it?**
The **Model Context Protocol (MCP)** is a standard that allows an application (CaseCraft) to expose its data and tools to AI agents (like Claude Desktop, Cursor, or IDE assistants) in a universal way.

**How is it useful for CaseCraft?**
*   **"Chat with your QA Tool"**: instead of running `python main.py` in a terminal, you could open Claude Desktop and ask *"Generate test cases for the Login feature based on CaseCraft's knowledge."* The AI agent would call CaseCraft's tools behind the scenes.
*   **IDE Integration**: You could connect CaseCraft as an MCP server to your IDE. While coding a feature, you could ask your IDE assistant *"Check if my code covers all test cases from CaseCraft"*, and it would pull the relevant tests directly.
*   **Workflow Automation**: Other agents (e.g., a "Documentation Agent" or "Jira Agent") could query CaseCraft to see if test cases exist for a new requirement without you building a custom API integration for each one.
