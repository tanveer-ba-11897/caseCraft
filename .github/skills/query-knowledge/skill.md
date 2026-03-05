# Query Knowledge Base

Search the CaseCraft RAG knowledge base for relevant requirement context.

## Instructions

Use the CaseCraft MCP server's `query_knowledge` tool to search the ChromaDB-backed knowledge base.

1. Take the user's question or topic as the search query.
2. Determine `top_k` — how many results to return (default: 3, max: 20).
3. Call `query_knowledge(query, top_k)` via the CaseCraft MCP server.
4. Present the results with source attribution.
5. Offer to generate test cases based on the retrieved context if appropriate.

## Retrieval Pipeline

The knowledge base uses a 7-stage hybrid retrieval pipeline:

1. **ChromaDB HNSW** — Dense vector search (cosine similarity, weight 0.7)
2. **BM25 Sparse** — Keyword matching (weight 0.3)
3. **Hybrid fusion** — Weighted combination of dense + sparse scores
4. **Parent-child expansion** — Expands child hits to full parent chunks
5. **Knowledge graph expansion** — BFS traversal for related chunks
6. **CrossEncoder re-ranking** — Neural re-ranking for precision
7. **Final deduplication** — Removes near-duplicate results

## Example Usage

```
What does the knowledge base say about user authentication?
```

```
Search KB for payment processing error handling with top_k=10
```

## Notes

- Query is truncated at 10,000 characters for safety
- top_k clamped to range [1, 20] (use -1 to retrieve all)
- Knowledge base must be ingested first via `python -m cli.ingest`
- The knowledge graph (if enabled) adds structural expansion beyond embedding similarity
- Results include source attribution from the original ingested documents
