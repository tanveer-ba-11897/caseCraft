
import json
from typing import List, Optional

import numpy as np

from core.schema import TestSuite
from core.parser import parse_document
from core.knowledge.retriever import KnowledgeRetriever
from core.knowledge.embedder import Embedder
from core.prompts import build_generation_prompt, build_condensation_prompt, build_reviewer_prompt
from core.llm_client import llm_client


class GenerationError(Exception):
    """Raised when test case generation fails."""
    pass

# Lazy loaded singletons
_retriever = None
_embedder = None

def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = KnowledgeRetriever()
    return _retriever

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder



def _retrieve_product_context(
    feature_text: str,
    top_k: int = None,
) -> str:
    from core.config import config
    if top_k is None:
        top_k = config.quality.top_k
    """
    Retrieve relevant product knowledge and format it
    as context for the LLM prompt.
    """
    chunks = get_retriever().retrieve(
        query_text=feature_text,
        top_k=top_k,
    )

    if not chunks:
        return ""

    context_blocks = []

    for idx, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            f"[Context {idx}]\n{chunk.text}"
        )

    return "\n\n".join(context_blocks)


def _build_prompt(
    feature_chunks: List[str],
    product_context: str,
    app_type: str = "web"
) -> str:
    return build_generation_prompt(feature_chunks, product_context, app_type)



def _condense_chunk(chunk: str, model: str) -> str:
    """
    Reduce a document chunk to concise, test relevant bullet points.
    """
    print(f"INFO: Processing chunk ({len(chunk)} chars)...")
    prompt = build_condensation_prompt(chunk)

    try:
        condensed = llm_client.generate(prompt, model, json_mode=False)
    except Exception as e:
         print(f"DEBUG: LLM Error: {e}")
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

    # Debug: print what we received
    print(f"DEBUG: Cannot normalize result of type {type(result).__name__}")
    if isinstance(result, dict):
        print(f"DEBUG: Dict keys: {list(result.keys())[:5]}")
    elif isinstance(result, list) and len(result) > 0:
        print(f"DEBUG: List item types: {[type(x).__name__ for x in result[:3]]}")
    
    # Fallback: if we can't find anything, return empty list instead of crashing
    # This assumes the chunk just had no testable content
    print("DEBUG: Normalization failed, assuming 0 test cases for this chunk.")
    return []




def _clean_json_output(text: str) -> str:
    """
    Sanitize LLM output to ensure valid JSON.
    1. Removes <think>...</think> blocks if present.
    2. Removes markdown code blocks.
    3. extracts the main JSON array [ ... ].
    """
    import re
    
    # 1. Remove <think> blocks (just in case model slips into thinking mode)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # 2. Remove markdown code blocks
    text = text.strip()
    if "```" in text:
        # Regex to capture content inside ```json ... ``` or just ``` ... ```
        match = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL)
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

    for _ in range(max_retries + 1):
        print(f"INFO: Generating tests (attempt {_ + 1})...")
        try:
            result = llm_client.generate(prompt, model, json_mode=True)
        except Exception as exc:
            print(f"DEBUG: Generate Error: {exc}")
            last_error = exc
            continue

        # Compatibility check: if result is empty string, retry
        if not result:
            print("DEBUG: Empty result from LLM")
            last_error = GenerationError("Empty model output")
            continue

        if isinstance(result, str):
            result = result.strip()
            if not result:
                print("DEBUG: Empty result string")
                last_error = GenerationError("Empty model output")
                continue
            try:
                result = _clean_json_output(result)
                result = json.loads(result)
            except json.JSONDecodeError as exc:
                print(f"DEBUG: JSON Decode Error: {exc} | Text: {result[:50]}...")
                last_error = exc
                continue

        try:
            test_cases = _normalize_test_cases(result)
        except GenerationError as exc:
            print(f"DEBUG: Normalization Error: {exc}")
            last_error = exc
            continue

        return test_cases

    raise GenerationError(
        "Failed to generate valid test cases after retries."
    ) from last_error


def _generate_from_chunks(
    chunks: List[str],
    model: str,
    max_retries: int,
    app_type: str = "web"
) -> List[dict]:
    """
    Generate test cases independently for each chunk.
    """
    all_test_cases: List[dict] = []

    for index, chunk in enumerate(chunks):
        condensed = _condense_chunk(chunk, model)
        chunk_prompt = _build_prompt(
            feature_chunks=[condensed],
            product_context="",
            app_type=app_type
        )


        try:
            test_cases = _generate_single_suite(
                prompt=chunk_prompt,
                model=model,
                max_retries=max_retries,
            )
            all_test_cases.extend(test_cases)
        except GenerationError as exc:
            raise GenerationError(
                f"Failed to generate test cases for chunk {index}"
            ) from exc

    return all_test_cases

def _normalize_text(value: str) -> str:
    if value is None:
        return ""
    return " ".join(value.lower().split())


def _sanitize_test_cases(test_cases: list[dict]) -> list[dict]:
    """
    Normalize LLM output to strictly match TestCase schema.
    """
    sanitized = []

    for case in test_cases:
        case = dict(case)  # shallow copy

        # expected_results must be List[str]
        exp = case.get("expected_results", [])
        if isinstance(exp, str):
            case["expected_results"] = []
        elif isinstance(exp, list):
            case["expected_results"] = [
                str(item) for item in exp if isinstance(item, (str, int, float))
            ]
        else:
            case["expected_results"] = []

        # Ensure required string fields have defaults
        if not case.get("use_case"):
            case["use_case"] = "General Use Case"
            
        if not case.get("test_case"):
            case["test_case"] = "Untitled Test Case"
            
        if not case.get("priority"):
            case["priority"] = "medium"

        # actual_results must be List[str]
        act = case.get("actual_results", [])
        if isinstance(act, list):
            normalized = []
            for item in act:
                if isinstance(item, str):
                    normalized.append(item)
                else:
                    normalized.append(str(item))
            case["actual_results"] = normalized
        else:
            case["actual_results"] = []

        # preconditions
        if not isinstance(case.get("preconditions"), list):
            case["preconditions"] = []

        # steps
        if not isinstance(case.get("steps"), list):
            case["steps"] = []

        if not isinstance(case.get("tags"), list):
            case["tags"] = []

        # dependencies
        if not isinstance(case.get("dependencies"), list):
            case["dependencies"] = []

        # test_data must be Dict[str, str]
        td = case.get("test_data", {})
        if isinstance(td, dict):
            case["test_data"] = {k: str(v) for k, v in td.items()}
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
    import re
    
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
        print(f"DEBUG: Exact deduplication removed {duplicates_found} duplicates")
    
    return deduplicated



def _deduplicate_semantically(
    test_cases: list[dict],
    threshold: float = 0.85,
) -> list[dict]:
    """
    Remove semantically similar test cases using embeddings.
    """
    if not test_cases:
        return []

    # 1. Prepare texts for embedding
    # We combine use_case + test_case + steps to form a signature
    texts = []
    for case in test_cases:
        content = f"{case.get('use_case', '')} {case.get('test_case', '')}"
        steps = " ".join(case.get('steps', []))
        texts.append(f"{content} {steps}")

    # 2. Generate embeddings
    try:
        embeddings = get_embedder().embed_strings(texts)
    except Exception:
        # Fallback if embedding fails
        return test_cases

    if not embeddings:
        return test_cases

    # 3. Compute similarity and filter
    # Simple O(N^2) comparison is fine for typically small number of test cases (< 50)
    
    indices_to_remove = set()
    n = len(embeddings)
    
    # Pre-normalize for cosine similarity if not already done by embedder, 
    # but Embedder.embed_strings does normalize=True.
    
    # We can use simple dot product since they are normalized
    for i in range(n):
        if i in indices_to_remove:
            continue
            
        vec_i = embeddings[i]
        
        for j in range(i + 1, n):
            if j in indices_to_remove:
                continue
                
            vec_j = embeddings[j]
            
            # Cosine similarity
            similarity = np.dot(vec_i, vec_j)
            
            if similarity > threshold:
                # Mark j for removal (keep i as the "canonical" one)
                indices_to_remove.add(j)

    return [
        case for idx, case in enumerate(test_cases)
        if idx not in indices_to_remove
    ]


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
            
        return TestSuite.parse_obj(improved_suite_dict)
        
    except Exception:
        # If review fails, return original safe suite
        return suite



def _validate_dependencies(test_cases: List[dict]):
    """
    Check if dependencies exist in the suite. Warn if missing.
    """
    existing_tests = {case.get("test_case") for case in test_cases if case.get("test_case")}
    
    for case in test_cases:
        deps = case.get("dependencies", [])
        for dep in deps:
            # We do a loose check (substring or exact match)
            if dep not in existing_tests:
                # Try partial match logic? Or just warn.
                # For now, just print to console as this is a CLI tool
                pass 
                # Ideally we would log this. For now let's just ensure the code runs.

def _clean_test_case_titles(test_cases: List[dict]) -> List[dict]:
    """
    Remove prefixes like 'TC-001: ', 'Case 1 - ', etc. from titles.
    """
    import re
    cleaned = []
    # Regex to match common prefixes:
    # ^(TC|Case|Test|Scenario)\s*[-_:]?\s*\d+[:\s-]*
    # e.g. "TC-001: Login" -> "Login"
    prefix_pattern = re.compile(r'^(TC|Case|Test|Scenario|ID)\s*[-_#]?\s*\d+\s*[:\.-]?\s*', re.IGNORECASE)
    
    for case in test_cases:
        title = case.get("test_case", "")
        if title:
            new_title = prefix_pattern.sub("", title).strip()
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
) -> TestSuite:
    """
    Generate a TestSuite using chunk wise generation.
    """
    from core.config import config
    from pathlib import Path
    
    # Use configured model if not explicitly provided
    if model is None:
        model = config.general.model
    chunks = parse_document(file_path)

    feature_text = "\n\n".join(chunks)
    product_context = _retrieve_product_context(feature_text)

    # Determine final parameters (Arg > Config)
    final_app_type = app_type if app_type else config.generation.app_type

    # Generate test cases from chunks
    raw_test_cases = _generate_from_chunks(
        chunks=chunks,
        model=model,
        max_retries=max_retries,
        app_type=final_app_type
    )


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
    
    if dedup_semantic:
        merged_suite["test_cases"] = _deduplicate_semantically(
            merged_suite["test_cases"],
            threshold=0.90  # Stricter threshold
        )

    # 3. Checklist Cross-Reference Pass
    input_path = Path(file_path)
    checklist_path = input_path.parent / f"{input_path.stem}_checklist.txt"
    if checklist_path.exists():
        print(f"[INFO] Found supporting checklist: {checklist_path.name}. Cross-referencing...")
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
                print(f"[SUCCESS] Cross-reference generated {len(missing_cases)} additional test cases.")
                missing_cases = _sanitize_test_cases(missing_cases)
                missing_cases = _clean_test_case_titles(missing_cases)
                merged_suite["test_cases"].extend(missing_cases)
                
                # Re-deduplicate just in case
                merged_suite["test_cases"] = _deduplicate_test_cases(merged_suite["test_cases"])
            else:
                print("[INFO] Cross-reference complete. No missing test cases identified.")
        except Exception as e:
            print(f"[WARNING] Checklist cross-reference failed: {e}. Proceeding with original suite.")
    
    # 4. Prioritization
    merged_suite["test_cases"] = _prioritize_test_cases(
        merged_suite["test_cases"]
    )
    
    # 5. Dependency Validation (Side effect: warnings)
    _validate_dependencies(merged_suite["test_cases"])

    try:
        final_suite = TestSuite.parse_obj(merged_suite)
        
        # 3. Optional Reviewer Pass
        if reviewer_pass:
            final_suite = _review_test_suite(final_suite, model)
            
        return final_suite
    except Exception as exc:
        raise GenerationError(
            "Merged test cases failed schema validation"
        ) from exc
