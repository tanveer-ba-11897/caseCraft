"""
Live benchmarking: run the full CaseCraft pipeline against each locally
available Ollama model using a real feature document.

Measures per-model:
  - Total pipeline time (parsing → generation → dedup)
  - Number of test cases generated
  - JSON parse success rate (clean vs repaired vs failed)
  - Token throughput (from Ollama stats)
  - Mean test-case quality score (field completeness)

Usage:
    python benchmark_live.py [feature_file]

    Defaults to features/File_Encryption.pdf if no file given.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import load_config
from core.llm_client import llm_client

# ---------------------------------------------------------------------------
# Logging — capture JSON-repair events from the generator
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("casecraft.benchmark")


# ---------------------------------------------------------------------------
# Helper: count JSON-repair events by intercepting log records
# ---------------------------------------------------------------------------
class RepairCounter(logging.Handler):
    """Count JSON-repair related log events during a benchmark run."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.json_parse_failures = 0
        self.json_repair_recoveries = 0
        self.empty_results = 0

    def reset(self):
        self.json_parse_failures = 0
        self.json_repair_recoveries = 0
        self.empty_results = 0

    def emit(self, record):
        msg = record.getMessage()
        if "JSON parse failed" in msg:
            self.json_parse_failures += 1
        elif "json_repair recovered" in msg:
            self.json_repair_recoveries += 1
        elif "Empty result" in msg:
            self.empty_results += 1


# ---------------------------------------------------------------------------
# Helper: quality score for a single test case dict
# ---------------------------------------------------------------------------
def quality_score(tc: dict) -> float:
    """
    Score 0.0–1.0 based on field completeness and richness.
    Weights: steps (30%), expected_results (25%), preconditions (15%),
             test_data (15%), use_case+test_case titles (15%).
    """
    score = 0.0

    # Steps — at least 2 meaningful steps
    steps = tc.get("steps", [])
    if isinstance(steps, list) and len(steps) >= 2:
        score += 0.30
    elif isinstance(steps, list) and len(steps) == 1:
        score += 0.15

    # Expected results — at least 1
    er = tc.get("expected_results", [])
    if isinstance(er, list) and len(er) >= 1:
        score += 0.25

    # Preconditions — at least 1
    pre = tc.get("preconditions", [])
    if isinstance(pre, list) and len(pre) >= 1:
        score += 0.15

    # Test data — non-empty dict
    td = tc.get("test_data", {})
    if isinstance(td, dict) and td:
        score += 0.15

    # Titles — both non-empty and non-generic
    uc = tc.get("use_case", "")
    tcn = tc.get("test_case", "")
    if uc and tcn and len(tcn) > 10:
        score += 0.15
    elif uc or tcn:
        score += 0.07

    return round(score, 2)


# ---------------------------------------------------------------------------
# Helper: get model info from Ollama
# ---------------------------------------------------------------------------
def get_model_info(model: str) -> dict:
    try:
        resp = requests.post(
            f"{llm_client.base_url}/api/show",
            json={"name": model},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        info = {}
        model_info = data.get("model_info", {})
        for key, value in model_info.items():
            if "context_length" in key.lower():
                info["context_length"] = value
                break
        details = data.get("details", {})
        info["parameter_size"] = details.get("parameter_size", "?")
        info["quantization"] = details.get("quantization_level", "?")
        info["family"] = details.get("family", "?")
        return info
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Run pipeline for one model
# ---------------------------------------------------------------------------
def benchmark_model(
    model: str,
    feature_file: str,
    config,
    repair_counter: RepairCounter,
) -> dict:
    """Run the full CaseCraft pipeline for *model* and collect metrics."""
    from core.generator import generate_test_suite

    result = {
        "model": model,
        "status": "ok",
        "total_time_s": 0,
        "test_cases": 0,
        "mean_quality": 0.0,
        "json_parse_failures": 0,
        "json_repair_recoveries": 0,
        "empty_results": 0,
    }

    # Attach the counter to the generator logger
    gen_logger = logging.getLogger("casecraft.generator")
    gen_logger.addHandler(repair_counter)
    repair_counter.reset()

    try:
        start = time.monotonic()
        suite = generate_test_suite(
            file_path=feature_file,
            model=model,
            max_retries=1,
            dedup_semantic=config.quality.semantic_deduplication,
            reviewer_pass=False,    # skip reviewer for speed
            app_type=config.generation.app_type,
        )
        elapsed = time.monotonic() - start

        cases = [tc.model_dump() for tc in suite.test_cases]
        scores = [quality_score(tc) for tc in cases]

        result["total_time_s"] = round(elapsed, 1)
        result["test_cases"] = len(cases)
        result["mean_quality"] = round(sum(scores) / len(scores), 2) if scores else 0.0
        result["json_parse_failures"] = repair_counter.json_parse_failures
        result["json_repair_recoveries"] = repair_counter.json_repair_recoveries
        result["empty_results"] = repair_counter.empty_results

    except Exception as exc:
        result["status"] = f"ERROR: {exc}"
        result["total_time_s"] = round(time.monotonic() - start, 1)
    finally:
        gen_logger.removeHandler(repair_counter)
        # Unload model to free RAM for the next one
        llm_client.unload_model(model)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    feature_file = sys.argv[1] if len(sys.argv) > 1 else "features/File_Encryption.pdf"
    if not Path(feature_file).exists():
        print(f"Feature file not found: {feature_file}")
        sys.exit(1)

    config = load_config()

    # Discover models
    try:
        resp = requests.get(f"{llm_client.base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
    except Exception as exc:
        print(f"Cannot reach Ollama: {exc}")
        sys.exit(1)

    print("=" * 100)
    print("  CaseCraft Live Pipeline Benchmark")
    print("=" * 100)
    print(f"  Feature file : {feature_file}")
    print(f"  Models       : {len(models)} — {', '.join(models)}")
    print(f"  Config       : chunk_size={config.generation.chunk_size}, "
          f"skip_condensation={config.generation.skip_condensation}, "
          f"app_type={config.generation.app_type}")
    print("=" * 100)
    print()

    repair_counter = RepairCounter()
    results = []

    for idx, model in enumerate(models, 1):
        info = get_model_info(model)
        params = info.get("parameter_size", "?")
        family = info.get("family", "?")
        ctx = info.get("context_length", "?")

        print(f"[{idx}/{len(models)}] {model}  ({family}, {params}, ctx={ctx})")
        print(f"  Running full pipeline...")

        r = benchmark_model(model, feature_file, config, repair_counter)
        r["info"] = info
        results.append(r)

        if r["status"] == "ok":
            print(f"  Time:           {r['total_time_s']:>8.1f} s")
            print(f"  Test cases:     {r['test_cases']:>8d}")
            print(f"  Mean quality:   {r['mean_quality']:>8.2f}")
            print(f"  JSON failures:  {r['json_parse_failures']:>8d}")
            print(f"  JSON recovered: {r['json_repair_recoveries']:>8d}")
            print(f"  Empty results:  {r['empty_results']:>8d}")
        else:
            print(f"  {r['status']}")
        print()

    # ── Summary Table ──
    print()
    print("=" * 130)
    header = (
        f"{'Model':<25} {'Params':<8} {'Time(s)':<9} {'Cases':<7} "
        f"{'Quality':<9} {'JSON Fail':<10} {'Recovered':<10} {'Empty':<7} {'Status':<10}"
    )
    print(header)
    print("-" * 130)

    # Sort by test cases descending, then quality
    results.sort(key=lambda x: (x.get("test_cases", 0), x.get("mean_quality", 0)), reverse=True)

    for r in results:
        params = r.get("info", {}).get("parameter_size", "?")
        if r["status"] == "ok":
            print(
                f"{r['model']:<25} {params:<8} {r['total_time_s']:<9.1f} "
                f"{r['test_cases']:<7d} {r['mean_quality']:<9.2f} "
                f"{r['json_parse_failures']:<10d} {r['json_repair_recoveries']:<10d} "
                f"{r['empty_results']:<7d} {'OK':<10}"
            )
        else:
            status_short = r["status"][:40]
            print(f"{r['model']:<25} {params:<8} {r['total_time_s']:<9.1f} "
                  f"{'—':<7} {'—':<9} {'—':<10} {'—':<10} {'—':<7} {status_short:<10}")

    print("=" * 130)
    print()
    print("Quality: 0.0–1.0 field-completeness score (steps, expected_results, preconditions, test_data, titles)")
    print("JSON Fail: parse errors before repair | Recovered: saved by json_repair | Empty: empty LLM responses")
    print()

    # ── JSON dump for further analysis ──
    out_path = Path("outputs") / "benchmark_results.json"
    out_path.parent.mkdir(exist_ok=True)
    # Strip non-serializable info
    for r in results:
        r.pop("info", None)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Raw results saved to {out_path}")


if __name__ == "__main__":
    main()
