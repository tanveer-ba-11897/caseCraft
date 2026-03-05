# RAG & LLM Technology Overview

This document analyzes specific technologies and parameters requested for review, detailing their purpose and potential impact on the CaseCraft project.

## 1. AnythingLLM

**What is it?**
A full-stack, "all-in-one" desktop/web application that turns documents into a chatbot. It bundles the Vector DB, the LLM connection, the Embedder, and the UI into a single installable package. It is designed for end-users, not just developers.

**Effect on CaseCraft:**

* **Platform Replacement**: AnythingLLM acts as a complete platform. If you integrated it, CaseCraft would likely stop managing its own `knowledge_base` and `llm_client`.
* **Architecture Change**: CaseCraft would become a "client" of AnythingLLM's API. You would upload docs to AnythingLLM, and CaseCraft would query AnythingLLM for context instead of using its own RAG pipeline.
* **Pros**: Zero maintenance on the database/ingestion side.
* **Cons**: Less control over the specific "chunking" or "prompting" logic unless you purely use it as a dumb backend.

## 2. R2R vs RagFlow vs Chroma

These tools operate at different layers of the stack.

### Chroma (ChromaDB)

**What is it?**
A **Vector Database**. It is purely a storage engine. It takes numbers (vectors) and stores them for fast retrieval.

* **Analogy**: It is like PostgreSQL or MySQL, but for AI embeddings.
* **Effect on CaseCraft**: This is a backend swap. Currently, CaseCraft uses a simple JSON file or local folder. Switching to Chroma would make CaseCraft faster and able to handle millions of documents, but the app logic remains exactly the same.

### R2R (RAG to Riches)

**What is it?**
A **Developer Framework (API)** for building RAG applications. It provides pre-built pipelines for "Ingest -> Chunk -> Embed -> Store".

* **Analogy**: It is like Django or Rails, but specifically for RAG backends.
* **Effect on CaseCraft**: You would replace your `core/knowledge/` folder with R2R calls. R2R would handle the complex logic of parsing PDFs and storing them. CaseCraft would just say "Here is a file" and "Give me context".

### RagFlow

**What is it?**
A **Deep Document Understanding Pipeline**. It focuses heavily on *parsing* complex documents (tables, OCR, scanned PDFs) before they even get to the database.

* **Analogy**: An advanced OCR/Scanner that structures data before filing it.
* **Effect on CaseCraft**: If CaseCraft fails to read complex formatted PDFs (e.g., invoices, technical specs with tables), integrating RagFlow would drastically improve the *quality* of the retrieved context. It is heavier than R2R but better for messy data.

**Summary Table**

| Tool | Layer | Role | CaseCraft Impact |
| :--- | :--- | :--- | :--- |
| **Chroma** | Storage | Holds the data | Better performance/scale. |
| **R2R** | Backend API | Manages the pipeline | Simplifies code maintenance. |
| **RagFlow** | Parsing Engine | Understands layout | Better accuracy on complex PDFs. |

## 3. Top K (Retrieval) vs Top P (Generation)

### Top K (In RAG Context)

**What is it?**
"How many relevant chunks should I retrieve?"

* If `Top K = 5`, the system looks for the 5 most relevant paragraphs in your documentation to answer the user.
* **Effect on CaseCraft**:
  * **High K (10+)**: Gives the LLM more context, but adds noise and might confuse the model (or hit token limits).
  * **Low K (3)**: Cleaner prompt, but might miss a critical edge case mentioned in a footnote.
  * *Implementation*: Pass this to `retriever.retrieve(k=...)`.

### Top P (Nucleus Sampling)

**What is it?**
"How creative/random should the next word be?"
It filters the possible next words to the top cumulative probability of $P$.

* **Top P = 0.1**: Only considers the absolute most likely words. (Very robotic, deterministic, consistent).
* **Top P = 0.9**: Considers a wider range of words. (More creative, varied, but risks hallucination).
* **Effect on CaseCraft**:
  * For **Test Cases**, you generally want a **Lower Top P** (e.g., 0.2 - 0.5) to ensure the test cases follow the exact schema and don't invent features.
  * *Implementation*: This is already in your `config.py` as `temperature` (similar concept), but `Top P` is a separate parameter often available in `options`.
