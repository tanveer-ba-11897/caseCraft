"""
Quick throughput benchmark for all locally available Ollama models.
Measures tokens/second, time-to-first-token, and total response time.
"""

import json
import time
import requests
import sys

OLLAMA_URL = "http://localhost:11434"
PROMPT = (
    "Generate 3 test cases for a login feature in JSON format. "
    "Each test case should have: test_case, steps, expected_results."
)

def get_models():
    """Fetch list of local models from Ollama."""
    resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]

def get_model_info(model: str) -> dict:
    """Get model metadata (parameter count, context length)."""
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/show", json={"name": model}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        info = {}
        # Extract context length
        model_info = data.get("model_info", {})
        for key, value in model_info.items():
            if "context_length" in key.lower():
                info["context_length"] = value
                break
        # Extract parameter count
        details = data.get("details", {})
        info["parameter_size"] = details.get("parameter_size", "?")
        info["quantization"] = details.get("quantization_level", "?")
        info["family"] = details.get("family", "?")
        return info
    except Exception:
        return {}

def benchmark_model(model: str) -> dict:
    """Run a benchmark against a single model using streaming to measure TTFT."""
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": PROMPT,
        "stream": True,
        "options": {
            "temperature": 0.2,
            "num_predict": 512,   # cap output for consistent comparison
        },
    }

    result = {
        "model": model,
        "status": "ok",
        "ttft_ms": 0,
        "total_time_s": 0,
        "eval_count": 0,
        "prompt_eval_count": 0,
        "tokens_per_sec": 0,
        "prompt_eval_tps": 0,
    }

    try:
        start = time.monotonic()
        first_token_time = None
        full_response = ""
        
        resp = requests.post(url, json=payload, timeout=300, stream=True)
        resp.raise_for_status()

        final_stats = {}
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            
            if chunk.get("response") and first_token_time is None:
                first_token_time = time.monotonic()
                
            full_response += chunk.get("response", "")
            
            if chunk.get("done"):
                final_stats = chunk
                break

        end = time.monotonic()
        total = end - start

        # Use Ollama's own stats if available
        eval_count = final_stats.get("eval_count", 0)
        eval_duration_ns = final_stats.get("eval_duration", 0)
        prompt_eval_count = final_stats.get("prompt_eval_count", 0)
        prompt_eval_duration_ns = final_stats.get("prompt_eval_duration", 0)

        result["total_time_s"] = round(total, 2)
        result["eval_count"] = eval_count
        result["prompt_eval_count"] = prompt_eval_count
        result["response_length"] = len(full_response)

        if first_token_time:
            result["ttft_ms"] = round((first_token_time - start) * 1000, 0)

        if eval_duration_ns > 0 and eval_count > 0:
            result["tokens_per_sec"] = round(eval_count / (eval_duration_ns / 1e9), 1)
        
        if prompt_eval_duration_ns > 0 and prompt_eval_count > 0:
            result["prompt_eval_tps"] = round(prompt_eval_count / (prompt_eval_duration_ns / 1e9), 1)

    except Exception as e:
        result["status"] = f"ERROR: {e}"

    return result

def main():
    print("=" * 90)
    print("  CaseCraft Model Throughput Benchmark")
    print("=" * 90)
    
    models = get_models()
    print(f"\nFound {len(models)} model(s): {', '.join(models)}\n")

    results = []
    for model in models:
        info = get_model_info(model)
        ctx = info.get("context_length", "?")
        params = info.get("parameter_size", "?")
        quant = info.get("quantization", "?")
        family = info.get("family", "?")
        
        print(f"Benchmarking: {model} ({family}, {params}, {quant}, ctx={ctx})")
        print(f"  Prompt: {len(PROMPT)} chars, max_output: 512 tokens")
        
        r = benchmark_model(model)
        r["info"] = info
        results.append(r)

        if r["status"] == "ok":
            print(f"  TTFT:           {r['ttft_ms']:>8.0f} ms")
            print(f"  Total time:     {r['total_time_s']:>8.2f} s")
            print(f"  Output tokens:  {r['eval_count']:>8d}")
            print(f"  Generation TPS: {r['tokens_per_sec']:>8.1f} tok/s")
            print(f"  Prompt TPS:     {r['prompt_eval_tps']:>8.1f} tok/s")
        else:
            print(f"  {r['status']}")
        print()

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Model':<30} {'Params':<8} {'TTFT(ms)':<10} {'Gen TPS':<10} {'Prompt TPS':<12} {'Total(s)':<10} {'Tokens':<8}")
    print("-" * 90)
    
    # Sort by generation TPS (descending)
    results.sort(key=lambda x: x.get("tokens_per_sec", 0), reverse=True)
    
    for r in results:
        params = r.get("info", {}).get("parameter_size", "?")
        if r["status"] == "ok":
            print(f"{r['model']:<30} {params:<8} {r['ttft_ms']:<10.0f} {r['tokens_per_sec']:<10.1f} {r['prompt_eval_tps']:<12.1f} {r['total_time_s']:<10.2f} {r['eval_count']:<8d}")
        else:
            print(f"{r['model']:<30} {params:<8} {'ERROR':<10}")

    print("=" * 90)
    print("\nTPS = tokens/second | TTFT = time to first token | Higher TPS = faster")
    print("Prompt TPS = prompt processing speed (prefill) | Gen TPS = generation speed (decoding)")

if __name__ == "__main__":
    main()
