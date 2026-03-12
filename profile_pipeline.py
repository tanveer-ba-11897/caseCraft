"""
Pipeline profiler — measures wall-clock time of every stage in CaseCraft.

Usage:
    python profile_pipeline.py <feature_file>

Reports:
    1. Document parsing + chunking
    2. ML model loading (embedder, reranker, BM25)
    3. KB retrieval (hybrid search)
    4. Prompt building (template rendering)
    5. LLM generation (single chunk, single attempt)
    6. JSON parsing + sanitization
    7. Semantic deduplication (embedding)
"""

import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("profiler")

if len(sys.argv) < 2:
    print("Usage: python profile_pipeline.py <feature_file>")
    sys.exit(1)

feature_file = sys.argv[1]
timings: dict[str, float] = {}

def timed(label: str):
    """Context manager that records wall-clock seconds for *label*."""
    class _Timer:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, *_):
            elapsed = time.perf_counter() - self.t0
            timings[label] = elapsed
            logger.info(">> %-40s  %8.2fs", label, elapsed)
    return _Timer()


# ── 1. Document parsing ────────────────────────────────────────────────────
with timed("1. Document parsing + chunking"):
    from core.parser import parse_document
    from core.config import config
    chunks = parse_document(
        feature_file,
        chunk_size=config.generation.chunk_size,
        overlap=config.generation.chunk_overlap,
    )
    from core.generator import _consolidate_small_chunks
    original_count = len(chunks)
    chunks = _consolidate_small_chunks(chunks, config.generation.chunk_size)
    logger.info("   %d raw chunks → %d consolidated, total %d chars",
                original_count, len(chunks), sum(len(c) for c in chunks))


# ── 2. ML model loading (embedder + reranker) ─────────────────────────────
with timed("2. Retriever init + ML model loading"):
    from core.generator import get_retriever
    retriever = get_retriever()
    # Force background preload to finish
    retriever._preload_thread.join()

logger.info("   Embedder loaded: %s", retriever.embedder is not None)
logger.info("   Reranker loaded: %s", retriever.reranker is not None)
logger.info("   BM25 built:      %s (%d docs)", retriever._bm25_built, len(retriever._bm25_ids))


# ── 3. KB retrieval ────────────────────────────────────────────────────────
with timed("3. KB retrieval (shared, full pipeline)"):
    from core.generator import _retrieve_product_context
    doc_text = "\n".join(chunks)[:6000]
    kb_batches = _retrieve_product_context(doc_text)

logger.info("   KB batches: %d, total chars: %d",
            len(kb_batches), sum(len(b) for b in kb_batches))


# ── 4. Prompt building ────────────────────────────────────────────────────
with timed("4. Prompt building (template render)"):
    from core.generator import _build_prompt
    test_chunk = chunks[0] if chunks else "No content"
    test_kb = kb_batches[0] if kb_batches else ""
    prompt = _build_prompt(
        feature_chunks=[test_chunk],
        product_context=test_kb,
        app_type=config.generation.app_type,
    )

logger.info("   Prompt length: %d chars (~%d tokens)", len(prompt), len(prompt) // 3)


# ── 5. LLM generation (single call) ───────────────────────────────────────
with timed("5. LLM generation (1 call)"):
    from core.llm_client import llm_client
    model = config.general.model
    llm_response = llm_client.generate(prompt, model, json_mode=True)

resp_len = len(llm_response) if llm_response else 0
logger.info("   Response: %d chars", resp_len)


# ── 6. JSON parsing + sanitization ────────────────────────────────────────
with timed("6. JSON parse + sanitize"):
    import json
    from core.generator import _clean_json_output, _normalize_test_cases, _sanitize_test_cases
    cleaned = _clean_json_output(llm_response)
    try:
        parsed = json.loads(cleaned)
        cases = _normalize_test_cases(parsed)
        cases = _sanitize_test_cases(cases)
        logger.info("   Parsed %d test cases", len(cases))
    except Exception as e:
        logger.warning("   JSON parse failed: %s", e)
        cases = []


# ── 7. Semantic deduplication ──────────────────────────────────────────────
if cases:
    with timed("7. Semantic deduplication"):
        from core.generator import _deduplicate_semantically
        deduped, _ = _deduplicate_semantically(cases, threshold=config.quality.similarity_threshold)
    logger.info("   %d → %d after dedup", len(cases), len(deduped))
else:
    timings["7. Semantic deduplication"] = 0.0
    logger.info("   Skipped (no cases)")


# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PIPELINE PROFILING SUMMARY")
print("=" * 70)
total = sum(timings.values())
for label, elapsed in timings.items():
    pct = elapsed / total * 100 if total > 0 else 0
    bar = "#" * int(pct / 2)
    print(f"  {label:<42s}  {elapsed:8.2f}s  ({pct:5.1f}%)  {bar}")
print(f"  {'TOTAL':<42s}  {total:8.2f}s")
print("=" * 70)

# Identify the bottleneck
bottleneck = max(timings, key=timings.get)  # type: ignore[arg-type]
print(f"\n  BOTTLENECK: {bottleneck} — {timings[bottleneck]:.1f}s ({timings[bottleneck]/total*100:.0f}% of total)")
print()

# Chunk analysis
print(f"  Chunks: {len(chunks)}")
print(f"  Estimated LLM calls per full run: {len(chunks)} generation", end="")
if not config.generation.skip_condensation:
    print(f" + {len(chunks)} condensation", end="")
if config.generation.min_cases_per_chunk > 0:
    print(f" + up to {len(chunks)} re-gen (if < {config.generation.min_cases_per_chunk} cases/chunk)", end="")
print()

if "5. LLM generation (1 call)" in timings:
    llm_time = timings["5. LLM generation (1 call)"]
    est_total = llm_time * len(chunks)
    print(f"  Estimated total LLM time (generation only): {est_total:.0f}s ({est_total/60:.1f} min)")
    if not config.generation.skip_condensation:
        print(f"  + condensation: ~{est_total:.0f}s more ({est_total*2/60:.1f} min total)")
