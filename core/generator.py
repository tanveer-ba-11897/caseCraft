import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from typing import Any, Callable, List, Optional

import numpy as np
from tqdm import tqdm

from core.schema import TestSuite
from core.parser import parse_document
from core.knowledge.retriever import KnowledgeRetriever
from core.knowledge.embedder import Embedder
from core.prompts import build_generation_prompt, build_condensation_prompt, build_reviewer_prompt
from core.llm_client import llm_client

logger = logging.getLogger("casecraft.generator")

# Precompiled regex patterns for _clean_json_output
_THINK_PATTERN = re.compile(r'<think>.*?</think>', re.DOTALL)
_CODE_BLOCK_PATTERN = re.compile(r'```(?:json)?(.*?)```', re.DOTALL)
# Precompiled regex for _clean_test_case_titles
_TITLE_PREFIX_PATTERN = re.compile(r'^(TC|Case|Test|Scenario|ID)\s*[-_#]?\s*\d+\s*[:\.\-]?\s*', re.IGNORECASE)


class GenerationError(Exception):
    """Raised when test case generation fails."""
    pass

# Thread-safe lazy loaded singletons
_retriever = None
_embedder = None
_singleton_lock = threading.Lock()

def get_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever
    with _singleton_lock:
        # Double-check inside lock
        if _retriever is None:
            from core.config import config
            _retriever = KnowledgeRetriever(
                persist_dir=config.knowledge.vector_db_path,
                min_score_threshold=config.knowledge.min_score_threshold,
            )
    return _retriever

def get_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder
    with _singleton_lock:
        # Double-check inside lock
        if _embedder is None:
            # Share the same SentenceTransformer model with the retriever if already loaded
            try:
                retriever = get_retriever()
                _embedder = Embedder(model=retriever.embedder)
            except Exception:
                # Retriever not available (no index), load embedder independently
                _embedder = Embedder()
    return _embedder



def _compute_kb_budget() -> int:
    """
    Compute the character budget available for knowledge base context
    in a single LLM prompt.

    Uses the effective context window (native × context_window_ratio)
    and a static output token estimate (25% of effective ctx) since
    the actual prompt is not yet constructed.  At request time, the
    real output budget is computed dynamically from the prompt size.

    Rough formula:
        effective_ctx          = native_window × context_window_ratio
        estimated_output       = effective_ctx × 0.25   (static fallback)
        available_input_tokens = effective_ctx - estimated_output
        available_input_chars  = available_input_tokens × 4  (~4 chars/token)
        kb_budget              = available_input_chars - PROMPT_OVERHEAD

    PROMPT_OVERHEAD accounts for the generation template (~5K chars),
    the condensed feature chunk (~3K chars), and a safety margin.
    """
    from core.config import config
    from core.llm_client import llm_client

    # Use dynamic context window resolution
    ctx = llm_client.get_effective_context_window(config.general.model)

    # Static output estimate (prompt_chars=0 triggers 25% fallback)
    max_out = llm_client.get_effective_max_output_tokens(config.general.model, prompt_chars=0)
    available_tokens = max(ctx - max_out, 1024)  # floor at 1024
    available_chars = available_tokens * 4

    PROMPT_OVERHEAD = 10_000  # template + condensed feature + safety margin
    budget = max(available_chars - PROMPT_OVERHEAD, 5_000)
    return budget


def _batch_context_strings(context_blocks: List[str], budget: int) -> List[str]:
    """
    Split a list of context block strings into batches where each batch's
    combined length stays within *budget* characters.

    Every block appears in exactly one batch — no knowledge is lost.
    If a single block exceeds the budget it gets its own batch.
    """
    if not context_blocks:
        return [""]

    batches: List[str] = []
    current_batch: List[str] = []
    current_len = 0
    SEPARATOR_LEN = 2  # "\n\n" between blocks

    for block in context_blocks:
        block_len = len(block) + (SEPARATOR_LEN if current_batch else 0)

        if current_batch and current_len + block_len > budget:
            # Flush current batch
            batches.append("\n\n".join(current_batch))
            current_batch = [block]
            current_len = len(block)
        else:
            current_batch.append(block)
            current_len += block_len

    if current_batch:
        batches.append("\n\n".join(current_batch))

    return batches


def _retrieve_product_context(
    feature_text: str,
    top_k: Optional[int] = None,
) -> List[str]:
    """
    Retrieve relevant product knowledge using query transforms, format it,
    and split into batches that fit within the model's context window budget.

    Query Transform Pipeline:
    1. Decomposition — splits feature text into focused sub-queries
    2. Expansion — enriches queries with domain synonyms
    3. Multi-query retrieval — merges results from all transformed queries

    Returns a list of context-batch strings. If the total context is small
    enough, the list will contain a single element.
    """
    from core.config import config
    if top_k is None:
        top_k = config.quality.top_k

    retriever = get_retriever()
    
    enable_decomposition = config.knowledge.query_decomposition
    enable_expansion = config.knowledge.query_expansion
    max_sub_queries = config.knowledge.max_sub_queries

    # Truncate the feature text for retrieval (embedding models use ~256 tokens)
    MAX_QUERY_CHARS = 1500
    if len(feature_text) > MAX_QUERY_CHARS and not enable_decomposition:
        query = feature_text[:MAX_QUERY_CHARS]
        logger.info("Truncated retrieval query from %d to %d chars", len(feature_text), MAX_QUERY_CHARS)
    else:
        query = feature_text

    use_transforms = enable_decomposition or enable_expansion
    
    if use_transforms:
        transforms_desc = []
        if enable_decomposition:
            transforms_desc.append("decomposition")
        if enable_expansion:
            transforms_desc.append("expansion") 
        logger.info("Running multi-query retrieval with %s (top_k=%s, max_sub_queries=%d)...",
              ', '.join(transforms_desc), top_k, max_sub_queries)
        
        chunks = retriever.retrieve_multi_query(
            feature_text=query,
            top_k=top_k,
            enable_decomposition=enable_decomposition,
            enable_expansion=enable_expansion,
            max_sub_queries=max_sub_queries,
        )
    else:
        # Standard single-query retrieval
        if len(query) > MAX_QUERY_CHARS:
            query = query[:MAX_QUERY_CHARS]
            logger.info("Truncated retrieval query to %d chars", MAX_QUERY_CHARS)
        logger.info("Running hybrid retrieval (top_k=%s)...", top_k)
        chunks = retriever.retrieve(
            query_text=query,
            top_k=top_k,
        )
    
    logger.info("Retrieved %d knowledge chunks", len(chunks))

    if not chunks:
        return [""]

    context_blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            f"[Context {idx}]\n{chunk.text}"
        )

    total_chars = sum(len(b) for b in context_blocks)
    budget = _compute_kb_budget()
    logger.info("KB context: %d chars total, budget per batch: %d chars", total_chars, budget)

    batches = _batch_context_strings(context_blocks, budget)
    logger.info("Split KB context into %d batch(es)", len(batches))
    return batches


def _build_prompt(
    feature_chunks: List[str],
    product_context: str,
    app_type: str = "web"
) -> str:
    return build_generation_prompt(feature_chunks, product_context, app_type)



def _condense_chunk(chunk: str, model: str) -> str:
    """
    Reduce a document chunk to concise, test relevant bullet points.
    Uses the condensation cache to skip redundant LLM calls for
    identical or overlapping chunk text.
    """
    # Check condensation cache first
    try:
        from core.config import config as _cfg
        if _cfg.cache.enable_condensation_cache:
            from core.cache import get_condensation_cache
            cache = get_condensation_cache()
            cached = cache.get(chunk)
            if cached is not None:
                logger.info("Condensation cache HIT (%d chars cached)", len(cached))
                return cached
    except Exception:
        pass

    logger.info("Condensing chunk (%d chars)... sending to LLM", len(chunk))
    prompt = build_condensation_prompt(chunk)

    try:
        condensed = llm_client.generate(prompt, model, json_mode=False)
        logger.info("Condensation complete (%d chars returned)", len(condensed))
    except Exception as e:
         logger.debug("LLM Error: %s", e)
         raise GenerationError(f"Chunk condensation failed: {str(e)}")

    if not condensed:
        raise GenerationError("Empty condensed chunk returned")

    result = condensed.strip()

    # Store in condensation cache
    try:
        if _cfg.cache.enable_condensation_cache:  # type: ignore[possibly-undefined]
            cache.put(chunk, result)  # type: ignore[possibly-undefined]
    except Exception:
        pass

    return result

def _normalize_test_cases(result) -> list[dict]:
    """
    Normalize model output into a list of test case dicts.
    Accepts multiple reasonable shapes.
    """
    # Case 1: already a list
    if isinstance(result, list):
        # Filter out non-dict items
        return [item for item in result if isinstance(item, dict)]

    # Case 2: object with test_cases key
    if isinstance(result, dict):
        # Case 2a: Empty dict -> No test cases found (valid)
        if not result:
            return []
            
        # Try common keys for test case arrays
        for key in ["test_cases", "testCases", "tests", "test_case", "data", "results"]:
            if key in result:
                val = result[key]
                if val is None:
                    return []
                if isinstance(val, list):
                    return [item for item in val if isinstance(item, dict)]
        
        # Case 3: single test case object (has typical test case fields)
        test_case_fields = ["use_case", "test_case", "steps", "expected_results", "preconditions"]
        if any(field in result for field in test_case_fields):
            return [result]
        
        # Case 4: nested structure - look for any list of dicts
        for key, value in result.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                return [item for item in value if isinstance(item, dict)]

    logger.debug("Cannot normalize result of type %s", type(result).__name__)
    if isinstance(result, dict):
        logger.debug("Dict keys: %s", list(result.keys())[:5])
    elif isinstance(result, list) and len(result) > 0:
        logger.debug("List item types: %s", [type(x).__name__ for x in result[:3]])
    
    # Fallback: if we can't find anything, return empty list instead of crashing
    # This assumes the chunk just had no testable content
    logger.debug("Normalization failed, assuming 0 test cases for this chunk.")
    return []




def _clean_json_output(text: str) -> str:
    """
    Sanitize LLM output to ensure valid JSON.
    1. Removes <think>...</think> blocks if present.
    2. Removes markdown code blocks.
    3. extracts the main JSON array [ ... ].
    """
    # 1. Remove <think> blocks (just in case model slips into thinking mode)
    text = _THINK_PATTERN.sub('', text)
    
    # 2. Remove markdown code blocks
    text = text.strip()
    if "```" in text:
        # Regex to capture content inside ```json ... ``` or just ``` ... ```
        match = _CODE_BLOCK_PATTERN.search(text)
        if match:
            text = match.group(1).strip()
            
    # 3. Extract pure JSON array or object
    # Determine which comes first: '[' or '{'
    start_arr = text.find('[')
    start_obj = text.find('{')
    
    # If both exist, pick the earlier one
    if start_arr != -1 and start_obj != -1:
        if start_arr < start_obj:
            mode = 'array'
        else:
            mode = 'object'
    elif start_arr != -1:
        mode = 'array'
    elif start_obj != -1:
        mode = 'object'
    else:
        # Neither found
        return text.strip()
        
    if mode == 'array':
        end = text.rfind(']')
        if end != -1 and end > start_arr:
            text = text[start_arr : end + 1]
    else:
        end = text.rfind('}')
        if end != -1 and end > start_obj:
            text = text[start_obj : end + 1]
            
    return text.strip()


def _generate_single_suite(
    prompt: str,
    model: str,
    max_retries: int,
) -> list[dict]:
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        logger.info("Generating tests (attempt %d/%d)... sending to LLM", attempt + 1, max_retries + 1)
        try:
            result = llm_client.generate(prompt, model, json_mode=True)
            logger.info("LLM response received (%s chars)", len(result) if isinstance(result, str) else 'parsed')
        except Exception as exc:
            logger.debug("Generate Error: %s", exc)
            last_error = exc
            continue

        # Compatibility check: if result is empty string, retry
        if not result:
            logger.debug("Empty result from LLM")
            last_error = GenerationError("Empty model output")
            continue

        if isinstance(result, str):
            result = result.strip()
            if not result:
                logger.debug("Empty result string")
                last_error = GenerationError("Empty model output")
                continue
            try:
                result = _clean_json_output(result)
                result = json.loads(result)
            except json.JSONDecodeError as exc:
                logger.debug("JSON Decode Error: %s | Text: %s...", exc, result[:50])
                last_error = exc
                continue

        try:
            test_cases = _normalize_test_cases(result)
        except GenerationError as exc:
            logger.debug("Normalization Error: %s", exc)
            last_error = exc
            continue

        return test_cases

    raise GenerationError(
        "Failed to generate valid test cases after retries."
    ) from last_error


def _condense_kb_batch(batch: str, model: str, batch_idx: int, total: int) -> str:
    """
    Condense a KB context batch into focused bullet points.
    Reuses the same condensation prompt as feature chunks since the goal
    is identical: extract test-relevant facts.

    Returns the condensed text, or the original if condensation fails
    (so we never silently lose context).
    """
    if not batch or not batch.strip():
        return ""
    try:
        logger.info("    Condensing KB batch %d/%d (%d chars)...", batch_idx + 1, total, len(batch))
        condensed = _condense_chunk(batch, model)
        ratio = len(condensed) / len(batch) * 100
        logger.info("    KB batch %d condensed: %d → %d chars (%.0f%%)",
              batch_idx + 1, len(batch), len(condensed), ratio)
        return condensed
    except GenerationError:
        # Condensation failed — fall back to raw context rather than losing it
        logger.warning("KB batch %d condensation failed, using raw context", batch_idx + 1)
        return batch


def _process_single_chunk(
    index: int,
    chunk: str,
    model: str,
    max_retries: int,
    product_context_batches: List[str],
    app_type: str,
) -> List[dict]:
    """
    Process a single feature-doc chunk with enhanced RAG:

    1. Condense the feature chunk once.
    2. Condense each KB context batch (compress ~3-5x).
    3. Re-batch the condensed KB context (fewer, denser batches).
    4. Generate test cases against each re-batched KB context.

    This ensures no knowledge is lost while dramatically reducing
    the number of LLM generation calls.
    """
    logger.info("=== Processing chunk %d ===", index + 1)
    condensed_feature = _condense_chunk(chunk, model)

    # --- KB Condensation ---
    num_raw_batches = len(product_context_batches)
    if num_raw_batches > 0 and product_context_batches != [""]:
        logger.info("  Condensing %d KB batch(es) for chunk %d...", num_raw_batches, index + 1)
        condensed_kb_blocks = []
        for i, kb_batch in enumerate(product_context_batches):
            condensed = _condense_kb_batch(kb_batch, model, i, num_raw_batches)
            if condensed.strip():
                condensed_kb_blocks.append(condensed)

        # Re-batch the condensed KB (it's now much smaller)
        if condensed_kb_blocks:
            budget = _compute_kb_budget()
            final_batches = _batch_context_strings(condensed_kb_blocks, budget)
            total_condensed = sum(len(b) for b in condensed_kb_blocks)
            logger.info("  KB condensation complete: %d raw batch(es) → %d generation batch(es) (%d chars)",
                  num_raw_batches, len(final_batches), total_condensed)
        else:
            final_batches = [""]
    else:
        final_batches = [""]

    # --- Test Case Generation ---
    all_cases: List[dict] = []
    num_gen_batches = len(final_batches)

    for batch_idx, ctx_batch in enumerate(final_batches):
        if num_gen_batches > 1:
            logger.info("  Chunk %d — generation batch %d/%d", index + 1, batch_idx + 1, num_gen_batches)

        chunk_prompt = _build_prompt(
            feature_chunks=[condensed_feature],
            product_context=ctx_batch,
            app_type=app_type,
        )

        try:
            cases = _generate_single_suite(
                prompt=chunk_prompt,
                model=model,
                max_retries=max_retries,
            )
            all_cases.extend(cases)
        except GenerationError as exc:
            logger.warning("Generation failed for chunk %d, batch %d: %s", index + 1, batch_idx + 1, exc)

    if not all_cases:
        logger.warning(
            "Chunk %d: all KB batch(es) produced 0 test cases. "
            "Skipping chunk — generation continues with remaining chunks.",
            index + 1,
        )
        return []

    # --- Min-cases-per-chunk re-generation ---
    from core.config import config
    min_required = config.generation.min_cases_per_chunk
    if min_required > 0 and len(all_cases) < min_required:
        logger.info("  Chunk %d produced %d cases (below min %d). Re-generating with emphasis on completeness...",
              index + 1, len(all_cases), min_required)
        # Build a supplementary prompt asking for MORE cases
        supplement_prompt = _build_prompt(
            feature_chunks=[condensed_feature],
            product_context=final_batches[0] if final_batches else "",
            app_type=app_type,
        )
        # Prepend instruction to generate additional/missing cases
        supplement_prompt = (
            "IMPORTANT: The previous generation produced too few test cases. "
            "You MUST generate MORE test cases covering additional scenarios: "
            "negative cases, boundary values, error handling, edge cases, "
            "input validation, state transitions, and integration points. "
            "Generate at least " + str(min_required - len(all_cases)) + " additional unique test cases.\n\n"
            + supplement_prompt
        )
        try:
            extra_cases = _generate_single_suite(
                prompt=supplement_prompt,
                model=model,
                max_retries=max_retries,
            )
            if extra_cases:
                logger.info("  Re-generation added %d more cases for chunk %d", len(extra_cases), index + 1)
                all_cases.extend(extra_cases)
        except GenerationError as exc:
            logger.warning("Re-generation failed for chunk %d: %s", index + 1, exc)

    return all_cases


def _generate_from_chunks(
    chunks: List[str],
    model: str,
    max_retries: int,
    per_chunk_kb_batches: Optional[List[List[str]]] = None,
    app_type: str = "web",
    max_workers: int = 4,
) -> List[dict]:
    """
    Generate test cases for each feature-doc chunk in parallel using threads.

    Each chunk receives its own KB context batches (per-chunk retrieval)
    so the knowledge context is relevant to that specific chunk.
    LLM calls are I/O-bound so threading provides near-linear speedup.
    """
    if per_chunk_kb_batches is None:
        per_chunk_kb_batches = [[""]] * len(chunks)

    all_test_cases: List[dict] = []

    if len(chunks) == 1:
        try:
            return _process_single_chunk(
                0, chunks[0], model, max_retries, per_chunk_kb_batches[0], app_type
            )
        except Exception as exc:
            logger.warning(
                "Chunk 1/1 failed (%s). Returning empty result.", exc,
            )
            return []

    # Use ThreadPoolExecutor for parallel I/O-bound LLM calls
    with ThreadPoolExecutor(max_workers=min(max_workers, len(chunks))) as executor:
        future_to_index = {
            executor.submit(
                _process_single_chunk,
                index, chunk, model, max_retries,
                per_chunk_kb_batches[index], app_type
            ): index
            for index, chunk in enumerate(chunks)
        }

        # Collect results in original chunk order with progress bar
        results_by_index = {}
        skipped = 0
        pbar = tqdm(
            as_completed(future_to_index),
            total=len(chunks),
            desc="Generating test cases",
            unit="chunk"
        )
        # Per-chunk timeout: use configured LLM timeout + 60s buffer per chunk
        from core.config import config as _timeout_cfg
        chunk_timeout = _timeout_cfg.general.timeout + 60

        for future in pbar:
            index = future_to_index[future]
            try:
                results_by_index[index] = future.result(timeout=chunk_timeout)
            except FutureTimeoutError:
                logger.warning(
                    "Chunk %d/%d timed out after %ds. Skipping — generation continues.",
                    index + 1, len(chunks), chunk_timeout,
                )
                results_by_index[index] = []
                skipped += 1
            except Exception as exc:
                logger.warning(
                    "Chunk %d/%d failed (%s). Skipping — generation continues.",
                    index + 1, len(chunks), exc,
                )
                results_by_index[index] = []
                skipped += 1
        pbar.close()

        if skipped:
            logger.info(
                "%d/%d chunk(s) skipped due to errors.", skipped, len(chunks),
            )

        # --- Retry skipped chunks sequentially (one at a time) ---
        failed_indices = [
            i for i in sorted(results_by_index.keys())
            if not results_by_index[i]
            and chunks[i].strip()  # skip genuinely empty chunks
        ]
        if failed_indices:
            logger.info(
                "Retrying %d skipped chunk(s) sequentially...",
                len(failed_indices),
            )
            recovered = 0
            for idx in failed_indices:
                logger.info(
                    "Retry: chunk %d/%d", idx + 1, len(chunks),
                )
                try:
                    result = _process_single_chunk(
                        idx, chunks[idx], model, max_retries,
                        per_chunk_kb_batches[idx], app_type,
                    )
                    if result:
                        results_by_index[idx] = result
                        recovered += 1
                        logger.info(
                            "Retry succeeded for chunk %d (%d case(s))",
                            idx + 1, len(result),
                        )
                    else:
                        logger.warning(
                            "Retry for chunk %d produced 0 cases. Giving up on this chunk.",
                            idx + 1,
                        )
                except Exception as exc:
                    logger.warning(
                        "Retry for chunk %d failed again (%s). Giving up on this chunk.",
                        idx + 1, exc,
                    )
            if recovered:
                logger.info(
                    "Retry pass recovered %d/%d chunk(s).",
                    recovered, len(failed_indices),
                )

        for i in sorted(results_by_index.keys()):
            all_test_cases.extend(results_by_index[i])

    return all_test_cases


def _coerce_str_list(items: Any) -> list[str]:
    """
    Coerce a value into a flat ``List[str]``.

    Handles common LLM output shapes that violate the schema:
      - plain string → single-element list
      - list of dicts (e.g. [{"step": "…"}]) → stringify each dict's values
      - list of mixed types → str() each
      - non-list → empty list
    """
    if isinstance(items, str):
        return [items] if items.strip() else []
    if not isinstance(items, list):
        return []
    result: list[str] = []
    for item in items:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            # Flatten dict values into a single string.
            # e.g. {"step": 1, "action": "Click login"} → "1 - Click login"
            parts = [str(v) for v in item.values() if v is not None]
            if parts:
                result.append(" - ".join(parts))
        elif isinstance(item, (int, float)):
            result.append(str(item))
        # Skip other types silently
    return result


def _sanitize_test_cases(test_cases: list[dict]) -> list[dict]:
    """
    Normalize LLM output to strictly match TestCase schema.
    """
    sanitized = []

    for case in test_cases:
        case = dict(case)  # shallow copy

        # expected_results must be List[str]
        case["expected_results"] = _coerce_str_list(case.get("expected_results", []))

        # Ensure required string fields have defaults
        if not case.get("use_case"):
            case["use_case"] = "General Use Case"
            
        if not case.get("test_case"):
            case["test_case"] = "Untitled Test Case"
            
        if not case.get("priority"):
            case["priority"] = "medium"

        # actual_results must be List[str]
        case["actual_results"] = _coerce_str_list(case.get("actual_results", []))

        # preconditions must be List[str]
        case["preconditions"] = _coerce_str_list(case.get("preconditions", []))

        # steps must be List[str]
        case["steps"] = _coerce_str_list(case.get("steps", []))

        # tags must be List[str]
        case["tags"] = _coerce_str_list(case.get("tags", []))

        # dependencies must be List[str]
        case["dependencies"] = _coerce_str_list(case.get("dependencies", []))

        # test_data must be Dict[str, str]
        td = case.get("test_data", {})
        if isinstance(td, dict):
            case["test_data"] = {k: str(v) for k, v in td.items()}
        elif isinstance(td, list):
            # Sometimes LLMs return test_data as a list of dicts
            merged = {}
            for item in td:
                if isinstance(item, dict):
                    for k, v in item.items():
                        merged[k] = str(v)
            case["test_data"] = merged
        else:
            case["test_data"] = {}

        sanitized.append(case)

    return sanitized

def _deduplicate_test_cases(
    test_cases: list[dict],
) -> list[dict]:
    """
    Remove duplicate or near duplicate test cases across chunks.
    """
    def normalize_for_dedup(text: str) -> str:
        """Normalize text for deduplication: lowercase, strip punctuation, collapse whitespace."""
        if text is None:
            return ""
        # Remove punctuation and extra whitespace
        text = re.sub(r'[^\w\s]', '', text.lower())
        return " ".join(text.split())
    
    seen_signatures: set[str] = set()
    deduplicated: list[dict] = []
    duplicates_found = 0

    for case in test_cases:
        use_case = normalize_for_dedup(case.get("use_case", ""))
        title = normalize_for_dedup(case.get("test_case", ""))

        steps = case.get("steps", [])
        steps_text = " ".join(
            normalize_for_dedup(step) for step in steps if isinstance(step, str)
        )

        # Create signature from normalized components
        signature = f"{use_case}|{title}|{steps_text}"

        if signature in seen_signatures:
            duplicates_found += 1
            continue

        seen_signatures.add(signature)
        deduplicated.append(case)

    if duplicates_found > 0:
        logger.info("Exact deduplication removed %d duplicates", duplicates_found)
    
    return deduplicated



def _deduplicate_semantically(
    test_cases: list[dict],
    threshold: float = 0.85,
    _cached_embeddings: Optional[list] = None,
) -> tuple[list[dict], list]:
    """
    Remove semantically similar test cases using embeddings.
    Returns (deduplicated_cases, embeddings) so embeddings can be reused.
    Uses batched matrix multiplication for efficient similarity computation.
    """
    if not test_cases:
        return [], []

    # 1. Prepare texts for embedding
    texts = []
    for case in test_cases:
        content = f"{case.get('use_case', '')} {case.get('test_case', '')}"
        steps = " ".join(case.get('steps', []))
        texts.append(f"{content} {steps}")

    # 2. Generate embeddings (reuse cached ones for existing cases, embed only new ones)
    if _cached_embeddings and len(_cached_embeddings) <= len(texts):
        # Embed only newly added cases
        new_texts = texts[len(_cached_embeddings):]
        try:
            new_embeddings = get_embedder().embed_strings(new_texts) if new_texts else []
        except Exception:
            return test_cases, _cached_embeddings or []
        embeddings = _cached_embeddings + new_embeddings
    else:
        try:
            embeddings = get_embedder().embed_strings(texts)
        except Exception:
            return test_cases, []

    if not embeddings:
        return test_cases, []

    # 3. Batched cosine similarity using matrix multiplication
    emb_matrix = np.array(embeddings, dtype=np.float32)
    similarity_matrix = emb_matrix @ emb_matrix.T

    indices_to_remove = set()
    n = len(embeddings)

    for i in range(n):
        if i in indices_to_remove:
            continue
        for j in range(i + 1, n):
            if j in indices_to_remove:
                continue
            if similarity_matrix[i, j] > threshold:
                indices_to_remove.add(j)

    kept_cases = []
    kept_embeddings = []
    for idx, case in enumerate(test_cases):
        if idx not in indices_to_remove:
            kept_cases.append(case)
            kept_embeddings.append(embeddings[idx])

    return kept_cases, kept_embeddings


def _prioritize_test_cases(test_cases: List[dict]) -> List[dict]:
    """
    Sort test cases by priority and dependencies.
    Order: High > Medium > Low.
    Secondary sort: Fewer dependencies first (independent tests run first).
    """
    priority_map = {"high": 0, "medium": 1, "low": 2}
    
    def sort_key(case):
        # 1. Priority Score (lower is better)
        p_score = priority_map.get(case.get("priority", "medium").lower(), 1)
        # 2. Dependency count (fewer is better)
        d_count = len(case.get("dependencies", []))
        return (p_score, d_count)
        
    return sorted(test_cases, key=sort_key)


def _review_test_suite(
    suite: TestSuite,
    model: str,
) -> TestSuite:
    """
    Perform a reviewer pass to improve test case quality.
    """
    current_json = suite.model_dump_json()
    
    prompt = build_reviewer_prompt(current_json)
    
    try:
        result = llm_client.generate(prompt, model, json_mode=True)
        
        if not result:
            return suite
            
        improved_suite_dict = json.loads(result)
        
        # Ensure we don't lose the original source doc info if LLM hallucinates it
        if "source_document" not in improved_suite_dict:
            improved_suite_dict["source_document"] = suite.source_document
            
        return TestSuite.model_validate(improved_suite_dict)
        
    except Exception:
        # If review fails, return original safe suite
        return suite



def _validate_dependencies(test_cases: List[dict]):
    """
    Check if dependencies exist in the suite. Log warnings for missing deps.
    """
    existing_tests = {case.get("test_case") for case in test_cases if case.get("test_case")}

    for case in test_cases:
        deps = case.get("dependencies", [])
        for dep in deps:
            if dep not in existing_tests:
                logger.warning(
                    "Test '%s' depends on '%s' which is not in the suite",
                    case.get("test_case", "<unnamed>"),
                    dep,
                )

def _clean_test_case_titles(test_cases: List[dict]) -> List[dict]:
    """
    Remove prefixes like 'TC-001: ', 'Case 1 - ', etc. from titles.
    """
    cleaned = []
    
    for case in test_cases:
        title = case.get("test_case", "")
        if title:
            new_title = _TITLE_PREFIX_PATTERN.sub("", title).strip()
            case["test_case"] = new_title
        cleaned.append(case)
    return cleaned


def generate_test_suite(
    file_path: str,
    model: Optional[str] = None,  # Uses config.general.model if None
    max_retries: int = 2,
    dedup_semantic: bool = True,
    reviewer_pass: bool = False,
    app_type: Optional[str] = None, # If None, uses config default
    progress: Optional[Callable[[str, str], None]] = None,
) -> TestSuite:
    """
    Generate a TestSuite using chunk wise generation.

    Args:
        progress: Optional callback ``(stage, message)`` invoked at each
                  major pipeline stage so callers (e.g. MCP server)
                  can surface real-time progress to the user.
    """
    from core.config import config
    from pathlib import Path

    def _progress(stage: str, msg: str) -> None:
        if progress:
            progress(stage, msg)
    
    # Use configured model if not explicitly provided
    if model is None:
        model = config.general.model
    
    # Resolve effective context window (auto-detect + cap)
    effective_ctx = llm_client.get_effective_context_window(model)
    # Static estimate (actual output tokens are computed dynamically per-call)
    effective_max_out = llm_client.get_effective_max_output_tokens(model, prompt_chars=0)
    
    logger.info("Config: chunk_size=%d, chunk_overlap=%d, model=%s, context_window=%d (%.0f%% of native), "
          "min_output_tokens=%d, dynamic_output_est=%d, top_k=%s, min_cases_per_chunk=%d",
          config.generation.chunk_size, config.generation.chunk_overlap,
          model, effective_ctx, config.general.context_window_ratio * 100,
          config.general.min_output_tokens, effective_max_out,
          config.quality.top_k, config.generation.min_cases_per_chunk)
    chunks = parse_document(
        file_path,
        chunk_size=config.generation.chunk_size,
        overlap=config.generation.chunk_overlap,
    )
    total_chars = sum(len(c) for c in chunks)
    logger.info("Document parsed: %d chunk(s), %d total chars", len(chunks), total_chars)
    _progress("parsing", f"Document parsed — {len(chunks)} chunk(s), {total_chars:,} characters")

    # --- Per-Chunk KB Retrieval ---
    # Retrieve KB context separately for each feature chunk so the retriever
    # ranks knowledge by relevance to THAT specific chunk (not the whole doc).
    # Combined with max_kb_batches, only the most relevant context is kept.
    max_kb_batches = config.quality.max_kb_batches
    logger.info("Per-chunk KB retrieval (top_k=%s, max_kb_batches=%d)...", config.quality.top_k, max_kb_batches)
    _progress("retrieval", f"Retrieving knowledge base context for {len(chunks)} chunk(s)…")

    per_chunk_kb: List[List[str]] = []
    total_kb_batches = 0
    for i, chunk in enumerate(tqdm(chunks, desc="Retrieving KB context", unit="chunk")):
        logger.debug("Retrieving KB context for chunk %d/%d...", i + 1, len(chunks))
        batches = _retrieve_product_context(chunk)

        # Apply max_kb_batches cap (batches are relevance-ordered)
        if max_kb_batches > 0 and len(batches) > max_kb_batches:
            logger.info("  Capping KB batches: %d → %d (most relevant)", len(batches), max_kb_batches)
            batches = batches[:max_kb_batches]

        per_chunk_kb.append(batches)
        total_kb_batches += len(batches)

    logger.info("Per-chunk retrieval complete: %d total KB batch(es) across %d chunk(s)", total_kb_batches, len(chunks))
    _progress("retrieval_done", f"Knowledge retrieval complete — {total_kb_batches} context batch(es)")

    # Determine final parameters (Arg > Config)
    final_app_type = app_type if app_type else config.generation.app_type
    max_workers = config.generation.max_workers

    # Generate test cases (parallel across feature chunks, sequential KB condensation + generation)
    logger.info("Starting enhanced RAG pipeline: %d chunk(s), %d KB batch(es) to condense + generate using model '%s' (max_workers=%d)...",
          len(chunks), total_kb_batches, model, max_workers)
    _progress("generating", f"Generating test cases from {len(chunks)} chunk(s) using {model}…")
    raw_test_cases = _generate_from_chunks(
        chunks=chunks,
        model=model,
        max_retries=max_retries,
        per_chunk_kb_batches=per_chunk_kb,
        app_type=final_app_type,
        max_workers=max_workers,
    )
    logger.info("Raw generation complete: %d test cases from all chunks", len(raw_test_cases))

    if not raw_test_cases:
        logger.warning("No test cases generated from any chunk. Returning empty suite.")
        _progress("complete", "No test cases could be generated from any chunk")
        return TestSuite(
            feature_name="Generated Feature",
            source_document=file_path,
            test_cases=[],
        )

    _progress("post_processing", f"Post-processing {len(raw_test_cases)} raw test cases…")


    merged_suite = {
        "feature_name": "Generated Feature",
        "source_document": file_path,
        "test_cases": raw_test_cases,
    }
    
    # 1. Sanitize LLM output to match schema
    merged_suite["test_cases"] = _sanitize_test_cases(merged_suite["test_cases"])
    
    # NEW: Clean prefixes from titles (e.g. TC-001)
    merged_suite["test_cases"] = _clean_test_case_titles(merged_suite["test_cases"])
    
    # 2. Exact Deduplication
    merged_suite["test_cases"] = _deduplicate_test_cases(
        merged_suite["test_cases"]
    )
    
    _cached_emb = None
    if dedup_semantic:
        _progress("deduplication", "Running semantic deduplication…")
        merged_suite["test_cases"], _cached_emb = _deduplicate_semantically(
            merged_suite["test_cases"],
            threshold=config.quality.similarity_threshold
        )

    # 3. Checklist Cross-Reference Pass
    input_path = Path(file_path)
    checklist_path = input_path.parent / f"{input_path.stem}_checklist.txt"
    if checklist_path.exists():
        logger.info("Found supporting checklist: %s. Cross-referencing...", checklist_path.name)
        _progress("cross_reference", f"Cross-referencing with checklist {checklist_path.name}…")
        try:
            with open(checklist_path, "r", encoding="utf-8") as f:
                checklist_text = f.read()
                
            from core.prompts import build_cross_reference_prompt
            # Dump current suite to JSON for the prompt
            current_suite_json = json.dumps(merged_suite["test_cases"], indent=2)
            cross_ref_prompt = build_cross_reference_prompt(current_suite_json, checklist_text)
            
            # Generate missing cases
            missing_cases = _generate_single_suite(
                prompt=cross_ref_prompt,
                model=model,
                max_retries=max_retries
            )
            
            if missing_cases:
                logger.info("Cross-reference generated %d additional test cases.", len(missing_cases))
                missing_cases = _sanitize_test_cases(missing_cases)
                missing_cases = _clean_test_case_titles(missing_cases)
                merged_suite["test_cases"].extend(missing_cases)
                
                # Re-deduplicate (exact + semantic with cached embeddings)
                merged_suite["test_cases"] = _deduplicate_test_cases(merged_suite["test_cases"])
                if dedup_semantic:
                    merged_suite["test_cases"], _cached_emb = _deduplicate_semantically(
                        merged_suite["test_cases"],
                        threshold=config.quality.similarity_threshold,
                        _cached_embeddings=_cached_emb,
                    )
            else:
                logger.info("Cross-reference complete. No missing test cases identified.")
        except Exception as e:
            logger.warning("Checklist cross-reference failed: %s. Proceeding with original suite.", e)
    
    # 4. Prioritization
    merged_suite["test_cases"] = _prioritize_test_cases(
        merged_suite["test_cases"]
    )
    
    # 5. Dependency Validation (Side effect: warnings)
    _validate_dependencies(merged_suite["test_cases"])

    try:
        final_suite = TestSuite.model_validate(merged_suite)
        
        # 3. Optional Reviewer Pass
        if reviewer_pass:
            _progress("reviewing", "Running AI reviewer pass…")
            final_suite = _review_test_suite(final_suite, model)

        _progress("complete", f"Done — {len(final_suite.test_cases)} test cases ready")

        # Log cache performance statistics
        try:
            from core.cache import log_cache_stats
            log_cache_stats()
        except Exception:
            pass

        return final_suite
    except Exception as exc:
        raise GenerationError(
            "Merged test cases failed schema validation"
        ) from exc
