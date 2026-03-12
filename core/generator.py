import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional

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
# Trailing commas before ] or } (common LLM error)
_TRAILING_COMMA = re.compile(r',\s*([\]}])')
# Precompiled regex for _clean_test_case_titles
_TITLE_PREFIX_PATTERN = re.compile(r'^(TC|Case|Test|Scenario|ID)\s*[-_#]?\s*\d+\s*[:\.\-]?\s*', re.IGNORECASE)


class GenerationError(Exception):
    """Raised when test case generation fails."""
    pass


def _consolidate_small_chunks(chunks: List[str], target_size: int) -> List[str]:
    """
    Merge adjacent small chunks so that each consolidated chunk is close to
    *target_size* characters.  This dramatically reduces the number of LLM
    calls when the section-aware parser creates many tiny sections (common
    with PDFs that have lots of headings).

    Chunks that already exceed *target_size* are kept as-is.
    """
    if not chunks or target_size <= 0:
        return chunks

    consolidated: List[str] = []
    buffer = chunks[0]

    for chunk in chunks[1:]:
        candidate = buffer + "\n\n" + chunk
        if len(candidate) <= target_size:
            buffer = candidate
        else:
            consolidated.append(buffer)
            buffer = chunk

    if buffer:
        consolidated.append(buffer)

    return consolidated

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


def _warmup_retriever() -> None:
    """Pre-load retriever + ML models so they're ready before the first KB query.

    Called from a background thread at the start of generate_test_suite().
    Swallows all exceptions — if the KB is empty or models can't load,
    the actual retrieval call will handle it later.
    """
    try:
        get_retriever()  # triggers __init__ → background _preload_all thread
    except Exception:
        pass  # retrieval errors handled at call time



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
        available_input_chars  = available_input_tokens × CHARS_PER_TOKEN
        kb_budget              = available_input_chars - PROMPT_OVERHEAD

    PROMPT_OVERHEAD accounts for the generation template (~5K chars),
    the condensed feature chunk (~3K chars), and a safety margin.
    """
    from core.config import config
    from core.llm_client import llm_client, CHARS_PER_TOKEN

    # Use dynamic context window resolution
    ctx = llm_client.get_effective_context_window(config.general.model)

    # Static output estimate (prompt_chars=0 triggers 25% fallback)
    max_out = llm_client.get_effective_max_output_tokens(config.general.model, prompt_chars=0)
    available_tokens = max(ctx - max_out, 1024)  # floor at 1024
    available_chars = available_tokens * CHARS_PER_TOKEN

    PROMPT_OVERHEAD = 10_000  # template + condensed feature + safety margin
    budget = max(available_chars - PROMPT_OVERHEAD, 5_000)
    logger.debug("KB budget: %d chars (ctx=%d, max_out=%d, available_tok=%d, overhead=%d)",
                 budget, ctx, max_out, available_tokens, PROMPT_OVERHEAD)
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
    app_type: str = "web",
    max_cases: int = 0,
) -> str:
    return build_generation_prompt(feature_chunks, product_context, app_type, max_cases=max_cases)


def _truncate_prompt_to_budget(prompt: str, model: str) -> str:
    """
    If *prompt* exceeds the model's context window (minus a minimum output
    reserve), truncate the tail so the LLM receives a valid-length input.

    The truncation is a last-resort safety net — upstream budgeting should
    prevent this in normal operation.
    """
    from core.llm_client import llm_client, CHARS_PER_TOKEN
    from core.config import config

    ctx = llm_client.get_effective_context_window(model)
    min_out = max(config.general.min_output_tokens, 256)
    max_input_tokens = ctx - min_out
    max_input_chars = max_input_tokens * CHARS_PER_TOKEN

    if len(prompt) <= max_input_chars:
        return prompt

    overshoot = len(prompt) - max_input_chars
    logger.warning(
        "Prompt exceeds context budget by %d chars (%d tokens). "
        "Truncating KB context to fit.",
        overshoot, overshoot // CHARS_PER_TOKEN,
    )
    return prompt[:max_input_chars]



def _condense_chunk(chunk: str, model: str) -> str:
    """
    Reduce a document chunk to concise, test relevant bullet points.
    """
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

    return condensed.strip()

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
    3. Extracts the main JSON array [ ... ].
    4. Repairs common LLM JSON errors (trailing commas, truncated output).
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

    # 4. Repair common JSON errors from small LLMs
    text = _repair_json(text)

    return text.strip()


def _escape_inner_quote(match: re.Match, text: str) -> str:
    """
    Determine whether a matched double-quote is a *structural* JSON quote
    (key/value boundary) or an unescaped quote inside a string value.

    Heuristic: count un-escaped quotes from the start of text up to this
    position.  At a structural boundary the count is even (we're between
    strings); inside a string the count is odd.  Only escape when odd.
    """
    pos = match.start()
    # Count non-escaped quotes before this position
    n_quotes = 0
    i = 0
    while i < pos:
        if text[i] == '"' and (i == 0 or text[i - 1] != '\\'):
            n_quotes += 1
        i += 1
    # Odd count → we're inside a string value → escape it
    if n_quotes % 2 == 1:
        return '\\"'
    return '"'


def _fix_unescaped_inner_quotes(text: str) -> str:
    """
    Single-pass scan that escapes double-quotes found *inside* JSON string
    values while leaving structural quotes untouched.

    Walks the string tracking whether we are inside or outside a JSON string
    (by counting un-escaped quotes).  A quote that is inside a string *and*
    bordered by word/space characters on both sides is treated as an
    unescaped inner quote and replaced with \\".
    """
    result: list[str] = []
    in_string = False
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        # Handle escape sequences inside strings
        if in_string and ch == '\\' and i + 1 < n:
            result.append(ch)
            result.append(text[i + 1])
            i += 2
            continue

        if ch == '"':
            if not in_string:
                # Opening a string
                in_string = True
                result.append(ch)
            else:
                # Is this the real closing quote, or an unescaped inner quote?
                # Look ahead: structural closing quotes are followed by
                # , : ] } or whitespace (then one of those).
                rest = text[i + 1:].lstrip()
                if not rest or rest[0] in ',:]}\n':
                    # Structural close
                    in_string = False
                    result.append(ch)
                else:
                    # Unescaped inner quote — escape it
                    result.append('\\"')
            i += 1
            continue

        result.append(ch)
        i += 1

    return "".join(result)


def _repair_json(text: str) -> str:
    """
    Fix common JSON syntax errors that small LLMs produce:
    - Trailing commas before ] or }
    - Truncated output (unbalanced brackets)
    - Single quotes instead of double quotes (in simple cases)
    - Unescaped double quotes inside string values
    """
    # Remove trailing commas: ,] or ,}
    text = _TRAILING_COMMA.sub(r'\1', text)

    # Fix unescaped double quotes inside JSON string values.
    # E.g.  "Click the "Save" button"  →  "Click the \"Save\" button"
    text = _fix_unescaped_inner_quotes(text)

    # Fix truncated JSON: close unbalanced brackets/braces
    # Count open vs close for [ ] and { }
    open_sq = text.count('[')
    close_sq = text.count(']')
    open_br = text.count('{')
    close_br = text.count('}')

    if open_sq > close_sq or open_br > close_br:
        # Truncated output — try to close it cleanly.
        # Strip trailing partial content back to the last complete value
        text = text.rstrip()

        # Remove trailing incomplete key-value pairs or partial strings
        # Find the last clean "anchor" (a complete value ending)
        # This handles mid-string truncation like: "steps": ["Click lo
        last_good = max(
            text.rfind('"]'),        # end of string array
            text.rfind('"}'),        # end of object with string value
            text.rfind('"]),'),      # array value followed by comma (rare)
            text.rfind('"},' ),      # object value followed by comma
            text.rfind('}],'),       # end of array of objects
            text.rfind('true'),      # boolean values
            text.rfind('false'),
            text.rfind('null'),
        )

        if last_good > 0:
            # Advance past the anchor text to keep it
            # Find the end of the anchor sequence
            for end_char in (']', '}', ',', 'e', 'l'):
                idx = text.find(end_char, last_good)
                if idx != -1:
                    last_good = idx + 1
                    break
            text = text[:last_good]

        # Remove any trailing comma after truncation
        text = text.rstrip().rstrip(',')

        # Re-count and close
        open_sq = text.count('[')
        close_sq = text.count(']')
        open_br = text.count('{')
        close_br = text.count('}')

        # Close braces first, then brackets (inner → outer)
        text += '}' * (open_br - close_br)
        text += ']' * (open_sq - close_sq)

    return text


def _parse_llm_json(raw: str, attempt_label: str) -> Optional[list]:
    """Parse LLM output as JSON, with repair fallback. Returns parsed object or None."""
    if not raw or not raw.strip():
        logger.warning("Empty result string (%s)", attempt_label)
        return None
    raw = raw.strip()
    cleaned = _clean_json_output(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json
            repaired = repair_json(cleaned, return_objects=False)
            if isinstance(repaired, str):
                parsed = json.loads(repaired)
            else:
                parsed = repaired  # type: ignore[assignment]
            logger.info("json_repair recovered malformed JSON (%s)", attempt_label)
            return parsed  # type: ignore[return-value]
        except Exception:
            logger.warning("JSON parse failed (%s) | First 200 chars: %.200s", attempt_label, raw)
            return None


def _generate_single_suite(
    prompt: str,
    model: str,
    max_retries: int,
    max_output_hint: int = 0,
) -> list[dict]:
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        logger.info("Generating tests (attempt %d/%d)... sending to LLM", attempt + 1, max_retries + 1)
        try:
            result = llm_client.generate(prompt, model, json_mode=True,
                                         max_output_hint=max_output_hint,
                                         json_array_min_items=3)
            logger.info("LLM response received (%s chars)", len(result) if isinstance(result, str) else 'parsed')
        except Exception as exc:
            logger.warning("LLM call failed (attempt %d): %s", attempt + 1, exc)
            last_error = exc
            continue

        if not result:
            logger.warning("Empty result from LLM (attempt %d)", attempt + 1)
            last_error = GenerationError("Empty model output")
            continue

        if isinstance(result, str):
            parsed = _parse_llm_json(result, f"attempt {attempt + 1}")
            if parsed is None:
                last_error = GenerationError("JSON parse failed")
                continue
            result = parsed

        try:
            test_cases = _normalize_test_cases(result)
        except GenerationError as exc:
            logger.warning("Normalization failed (attempt %d): %s", attempt + 1, exc)
            last_error = exc
            continue

        if not test_cases:
            logger.info("Normalization returned 0 test cases (attempt %d)", attempt + 1)
            return test_cases

        # ── Accumulation loop for small-model early-EOS ──────────────
        # Small models (e.g. 8B) often emit 1 case and stop.  Instead of
        # re-sending the identical prompt (which gets the same result),
        # send a focused continuation prompt that includes the cases
        # already generated so the model produces *different* ones.
        _MIN_CASES_FOR_CONTINUATION = 3
        _MAX_CONTINUATIONS = 3
        accumulated = list(test_cases)

        if len(accumulated) < _MIN_CASES_FOR_CONTINUATION:
            logger.info("  Only %d case(s) generated — entering continuation loop (up to %d rounds)",
                        len(accumulated), _MAX_CONTINUATIONS)

        for cont_round in range(_MAX_CONTINUATIONS):
            if len(accumulated) >= _MIN_CASES_FOR_CONTINUATION:
                break

            existing_summary = "\n".join(
                f"  {i+1}. {c.get('test_case', 'unnamed')}" for i, c in enumerate(accumulated)
            )
            continuation_prompt = (
                "You are a senior QA engineer. You already generated these test cases:\n"
                + existing_summary
                + "\n\nThe feature documentation needs MORE coverage. "
                "Generate " + str(_MIN_CASES_FOR_CONTINUATION - len(accumulated))
                + " ADDITIONAL test cases covering DIFFERENT scenarios "
                "not listed above. Focus on: negative/error paths, boundary values, "
                "edge cases, and alternate flows.\n\n"
                "Return ONLY a valid JSON array of new test case objects using the "
                "same schema as above. No markdown, no code fences.\n\n"
                + prompt  # include original prompt for feature context
            )

            # Truncate to budget
            continuation_prompt = _truncate_prompt_to_budget(continuation_prompt, model)

            logger.info("  Continuation round %d/%d — requesting %d more case(s)...",
                        cont_round + 1, _MAX_CONTINUATIONS,
                        _MIN_CASES_FOR_CONTINUATION - len(accumulated))
            try:
                cont_result = llm_client.generate(
                    continuation_prompt, model, json_mode=True,
                    max_output_hint=max_output_hint,
                )
            except Exception as exc:
                logger.warning("  Continuation round %d failed: %s", cont_round + 1, exc)
                break

            if not cont_result:
                logger.warning("  Continuation round %d returned empty", cont_round + 1)
                break

            if isinstance(cont_result, str):
                parsed = _parse_llm_json(cont_result, f"continuation {cont_round + 1}")
                if parsed is None:
                    break
                cont_result = parsed

            try:
                extra_cases = _normalize_test_cases(cont_result)
            except GenerationError:
                logger.warning("  Continuation round %d normalization failed", cont_round + 1)
                break

            if not extra_cases:
                logger.info("  Continuation round %d produced 0 cases — stopping", cont_round + 1)
                break

            # Deduplicate by test_case name
            existing_names = {c.get("test_case", "").lower().strip() for c in accumulated}
            new_cases = [c for c in extra_cases if c.get("test_case", "").lower().strip() not in existing_names]

            if new_cases:
                logger.info("  Continuation round %d added %d new case(s)", cont_round + 1, len(new_cases))
                accumulated.extend(new_cases)
            else:
                logger.info("  Continuation round %d produced only duplicates — stopping", cont_round + 1)
                break

        if len(accumulated) > len(test_cases):
            logger.info("  Accumulation complete: %d → %d cases", len(test_cases), len(accumulated))

        return accumulated

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
    from core.config import config as _gen_cfg
    logger.info("=== Processing chunk %d ===", index + 1)
    if _gen_cfg.generation.skip_condensation:
        logger.info("  Skipping condensation (skip_condensation=true), using raw chunk (%d chars)", len(chunk))
        condensed_feature = chunk
    else:
        condensed_feature = _condense_chunk(chunk, model)

    # --- KB Context Preparation (no per-batch condensation) ---
    # Skip per-batch KB condensation — it adds N extra LLM calls per chunk
    # and the generation LLM handles raw context well enough.
    # Just merge and batch the raw KB context within budget.
    #
    # Additional guard: cap total KB context at 3× the feature chunk size.
    # A 2.7K feature chunk doesn't need 28K of KB context — that bloats
    # the prompt, slows prompt eval on CPU, and dilutes relevance.
    max_kb_ratio = _gen_cfg.generation.max_kb_ratio
    kb_char_cap = len(condensed_feature) * max_kb_ratio

    num_raw_batches = len(product_context_batches)
    if num_raw_batches > 0 and product_context_batches != [""]:
        budget = min(_compute_kb_budget(), kb_char_cap)
        # Split merged KB batches back into individual context blocks
        # so we can keep top-ranked chunks that fit within budget
        # instead of blindly truncating mid-sentence.
        import re as _re
        individual_blocks: List[str] = []
        for kb_batch in product_context_batches:
            if not kb_batch.strip():
                continue
            # Each block starts with "[Context N]" — split on that boundary
            parts = _re.split(r'(?=\[Context \d+\])', kb_batch)
            for part in parts:
                part = part.strip()
                if part:
                    individual_blocks.append(part)

        # Keep only the top-ranked blocks that fit within budget.
        # Retrieval returns blocks ranked by relevance, so we take
        # from the front — most relevant first, least relevant dropped.
        kept_blocks: List[str] = []
        running_len = 0
        dropped = 0
        for block in individual_blocks:
            block_cost = len(block) + (2 if kept_blocks else 0)  # "\n\n" separator
            if running_len + block_cost <= budget:
                kept_blocks.append(block)
                running_len += block_cost
            else:
                dropped += 1
        if dropped > 0:
            logger.info("  KB context: kept %d/%d blocks within %d-char budget (dropped %d least relevant)",
                        len(kept_blocks), len(individual_blocks), budget, dropped)

        if kept_blocks:
            final_batches = _batch_context_strings(kept_blocks, budget)
            total_kb_chars = sum(len(b) for b in final_batches)
            logger.info("  KB context: %d raw batch(es) → %d generation batch(es), %d chars (cap %d)",
                  num_raw_batches, len(final_batches), total_kb_chars, kb_char_cap)
        else:
            final_batches = [""]
    else:
        final_batches = [""]

    # --- Test Case Generation ---
    all_cases: List[dict] = []
    num_gen_batches = len(final_batches)
    max_cases = _gen_cfg.generation.max_cases_per_chunk

    # Estimate output tokens to tighten num_predict.
    # Each JSON test case is roughly 300-400 tokens; add a 30% buffer
    # for the surrounding array brackets and model overhead.
    TOKENS_PER_CASE = 400
    BUFFER = 1.3
    if max_cases > 0:
        max_output_hint = int(max_cases * TOKENS_PER_CASE * BUFFER)
    else:
        # No explicit cap — estimate from chunk size.
        # Empirically ~1 test case per 500-800 chars of feature text.
        # Use the conservative end (500) so we don't under-allocate,
        # then add a generous 50% buffer on top.
        CHARS_PER_CASE_ESTIMATE = 500
        estimated_cases = max(3, len(condensed_feature) // CHARS_PER_CASE_ESTIMATE)
        max_output_hint = int(estimated_cases * TOKENS_PER_CASE * 1.5)
        logger.info("  No max_cases cap — estimated %d cases from %d-char chunk → output hint %d tok",
                     estimated_cases, len(condensed_feature), max_output_hint)

    for batch_idx, ctx_batch in enumerate(final_batches):
        if num_gen_batches > 1:
            logger.info("  Chunk %d — generation batch %d/%d", index + 1, batch_idx + 1, num_gen_batches)

        chunk_prompt = _build_prompt(
            feature_chunks=[condensed_feature],
            product_context=ctx_batch,
            app_type=app_type,
            max_cases=max_cases,
        )
        chunk_prompt = _truncate_prompt_to_budget(chunk_prompt, model)

        try:
            cases = _generate_single_suite(
                prompt=chunk_prompt,
                model=model,
                max_retries=max_retries,
                max_output_hint=max_output_hint,
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
            "Generate at least " + str(min_required - len(all_cases)) + " more test cases. "
            "Focus on: negative cases, boundary values, error handling, edge cases.\n\n"
            + supplement_prompt
        )
        supplement_prompt = _truncate_prompt_to_budget(supplement_prompt, model)
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
        for future in pbar:
            index = future_to_index[future]
            try:
                results_by_index[index] = future.result()
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

        for i in sorted(results_by_index.keys()):
            all_test_cases.extend(results_by_index[i])

    return all_test_cases


def _coerce_str_list(items) -> list[str]:
    """
    Coerce *items* into a flat ``list[str]``.

    Handles every shape LLMs actually return:
    - ``str``              → ``[s]``
    - ``list[str]``        → pass-through
    - ``list[dict]``       → flatten each dict's values into one string
    - ``list[mixed]``      → ``str()`` every element
    - anything else        → ``[]``
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
            # {"step": 1, "action": "Click"} → "step: 1, action: Click"
            parts = [f"{k}: {v}" for k, v in item.items() if v is not None]
            if parts:
                result.append(", ".join(parts))
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

        # Ensure required string fields have defaults
        if not case.get("use_case"):
            case["use_case"] = "General Use Case"
        if not case.get("test_case"):
            case["test_case"] = "Untitled Test Case"
        if not case.get("priority"):
            case["priority"] = "medium"

        # All list[str] fields — coerce dicts/mixed types to strings
        case["expected_results"] = _coerce_str_list(case.get("expected_results", []))
        case["actual_results"] = _coerce_str_list(case.get("actual_results", []))
        case["preconditions"] = _coerce_str_list(case.get("preconditions", []))
        case["steps"] = _coerce_str_list(case.get("steps", []))
        case["tags"] = _coerce_str_list(case.get("tags", []))
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
    prompt = _truncate_prompt_to_budget(prompt, model)
    
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
    
    # --- Eager retriever warm-up ---
    # Start loading ML models (SentenceTransformer, CrossEncoder, BM25)
    # in the background BEFORE document parsing.  Document parsing for PDFs
    # takes ~12 s; model loading takes ~7 s.  By overlapping them the user
    # never waits for model init — it finishes while the PDF is still being
    # parsed.
    _progress("init", "Warming up retriever models…")
    _warmup_thread = threading.Thread(target=_warmup_retriever, daemon=True)
    _warmup_thread.start()

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

    # --- Consolidate small chunks ---
    # Section-aware parsing often creates many tiny chunks (e.g. 86 chunks
    # averaging ~650 chars for a 56K doc).  Each chunk triggers a separate
    # LLM call, so consolidating them to the target chunk_size cuts the
    # number of (slow) LLM invocations dramatically.
    original_count = len(chunks)
    chunks = _consolidate_small_chunks(chunks, config.generation.chunk_size)
    if len(chunks) < original_count:
        logger.info("Consolidated %d small chunks → %d (target %d chars)",
              original_count, len(chunks), config.generation.chunk_size)
    _progress("parsing", f"Document parsed — {len(chunks)} chunk(s), {total_chars:,} characters")

    # --- KB Retrieval ---
    max_kb_batches = config.quality.max_kb_batches
    shared_kb = config.quality.shared_kb_retrieval

    if shared_kb:
        # Single retrieval using the full document text → shared across all chunks
        logger.info("Shared KB retrieval (top_k=%s, max_kb_batches=%d)...", config.quality.top_k, max_kb_batches)
        _progress("retrieval", "Retrieving shared knowledge base context…")
        doc_text = "\n".join(chunks)[:6000]  # Representative excerpt
        shared_batches = _retrieve_product_context(doc_text)
        if max_kb_batches > 0 and len(shared_batches) > max_kb_batches:
            shared_batches = shared_batches[:max_kb_batches]
        per_chunk_kb = [shared_batches] * len(chunks)
        total_kb_batches = len(shared_batches)
        logger.info("Shared retrieval complete: %d KB batch(es) shared across %d chunk(s)", total_kb_batches, len(chunks))
    else:
        # Per-chunk retrieval — more targeted but N× slower
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
            cross_ref_prompt = _truncate_prompt_to_budget(cross_ref_prompt, model)
            
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

        return final_suite
    except Exception as exc:
        raise GenerationError(
            "Merged test cases failed schema validation"
        ) from exc
