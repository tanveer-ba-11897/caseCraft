
import logging
import os
import random
import requests
import threading
import time
from typing import Any, Dict, Optional, Tuple
from core.config import config

logger = logging.getLogger("casecraft.llm")

# HTTP status codes that should trigger a retry
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

# ── Per-model specifications ────────────────────────────────────────────────
# Hard ceilings imposed by the provider.  Each entry contains:
#   - context_window : model's maximum context window (tokens)
#   - max_output     : model's maximum output / completion tokens
# Unknown models fall back to detect_native_context_window() for context
# and have no output cap.
MODEL_SPECS: Dict[str, Dict[str, int]] = {
    # OpenAI
    "gpt-4o":       {"context_window": 128_000, "max_output": 16_384},
    "gpt-4o-mini":  {"context_window": 128_000, "max_output": 16_384},
    "o1":           {"context_window": 200_000, "max_output": 100_000},
    "o1-mini":      {"context_window": 128_000, "max_output": 65_536},
    "o3-mini":      {"context_window": 200_000, "max_output": 100_000},
    "o4-mini":      {"context_window": 200_000, "max_output": 100_000},
    "gpt-5":        {"context_window": 200_000, "max_output": 100_000},
    "gpt-5-mini":   {"context_window": 200_000, "max_output": 100_000},
    "gpt-5-nano":   {"context_window": 200_000, "max_output": 32_768},
    # Meta
    "Meta-Llama-3.1-405B-Instruct": {"context_window": 128_000, "max_output": 4_096},
    "Meta-Llama-3.1-70B-Instruct":  {"context_window": 128_000, "max_output": 4_096},
    "Meta-Llama-3.1-8B-Instruct":   {"context_window": 128_000, "max_output": 4_096},
    # DeepSeek
    "DeepSeek-R1":      {"context_window": 128_000, "max_output": 16_384},
    "DeepSeek-R1-0528": {"context_window": 128_000, "max_output": 16_384},
    # Microsoft
    "Phi-4":    {"context_window": 16_384, "max_output": 4_096},
    "MAI-DS-R1": {"context_window": 128_000, "max_output": 16_384},
    # Mistral
    "Mistral-large":      {"context_window": 128_000, "max_output": 4_096},
    "Mistral-large-2411": {"context_window": 128_000, "max_output": 4_096},
    "Mistral-small":      {"context_window": 32_000,  "max_output": 4_096},
    # Cohere
    "Cohere-command-r-plus": {"context_window": 128_000, "max_output": 4_096},
    "Cohere-command-r":      {"context_window": 128_000, "max_output": 4_096},
    # xAI
    "xai-grok-3":      {"context_window": 131_072, "max_output": 16_384},
    "xai-grok-3-mini": {"context_window": 131_072, "max_output": 16_384},
    # Ollama local models
    "llama3.2:1b":  {"context_window": 131_072, "max_output": 2_048},
    "llama3.2:3b":  {"context_window": 131_072, "max_output": 4_096},
    "llama3.1:8b":  {"context_window": 131_072, "max_output": 4_096},
    "qwen2.5:7b":   {"context_window": 131_072, "max_output": 4_096},
    "gemma3:4b-it-q4_K_M": {"context_window": 131_072, "max_output": 4_096},
}

# Default max_output cap for models NOT listed in MODEL_SPECS.
# Prevents runaway generation when the model lacks a hard ceiling.
DEFAULT_MAX_OUTPUT = 4_096

# Approximate characters per token (used to estimate input tokens from prompt length).
# Conservative (3) to avoid underestimating token counts for structured prompts
# with JSON, whitespace, and special characters — prevents num_ctx truncation.
CHARS_PER_TOKEN = 3

# ── GitHub Models (Copilot) ────────────────────────────────────────────────
# When llm_provider is "copilot", requests go through GitHub's Models API.
# Supports both the legacy and current endpoints:
#   - Legacy:  https://models.inference.ai.azure.com
#   - Current: https://models.github.ai/inference
# Requires a GitHub PAT with 'models:read' scope.
# Env var: GITHUB_TOKEN or CASECRAFT_GENERAL_API_KEY
GITHUB_MODELS_BASE_URL = "https://models.github.ai/inference"
GITHUB_MODELS_SUPPORTED = {
    # OpenAI
    "gpt-4o", "gpt-4o-mini", "o1", "o1-mini", "o3-mini", "o4-mini",
    "gpt-5", "gpt-5-mini", "gpt-5-nano",
    # Meta
    "Meta-Llama-3.1-405B-Instruct", "Meta-Llama-3.1-70B-Instruct",
    "Meta-Llama-3.1-8B-Instruct",
    # Mistral
    "Mistral-large", "Mistral-large-2411", "Mistral-small",
    # Cohere
    "Cohere-command-r-plus", "Cohere-command-r",
    # DeepSeek
    "DeepSeek-R1", "DeepSeek-R1-0528",
    # Microsoft
    "Phi-4", "MAI-DS-R1",
    # xAI
    "xai-grok-3", "xai-grok-3-mini",
}


class LLMClient:
    """
    Abstract client for interacting with LLM backends.
    Supports 'ollama' (native) and 'openai' (compatible) formats.
    Features exponential backoff with jitter for transient failures
    and configurable inter-call throttling for rate-limited providers.
    """
    
    def __init__(self):
        self.provider = config.general.llm_provider
        self.base_url = config.general.base_url
        self.api_key = config.general.api_key.get_secret_value()
        self.timeout = config.general.timeout
        self._resolved_context_window: Optional[int] = None
        self._effective_context_window: Optional[int] = None
        
        # Retry configuration (could be moved to config if needed)
        self.max_retries = 3
        self.base_delay = 1.0  # seconds
        self.max_delay = 60.0  # seconds

        # ── Inter-call throttle ────────────────────────────────────────
        # Enforces a minimum gap between consecutive LLM API calls.
        # Prevents rate-limit (429) errors on providers with strict RPM
        # caps such as GitHub Models free tier (10-15 RPM).
        self._call_delay = config.general.llm_call_delay
        self._last_call_time: float = 0.0
        self._throttle_lock = threading.Lock()

        # Warn if using unencrypted HTTP for non-localhost addresses
        if self.base_url.startswith("http://") and not any(
            h in self.base_url for h in ("localhost", "127.0.0.1", "0.0.0.0")
        ):
            logger.warning("LLM base_url uses unencrypted HTTP for a non-localhost address. "
                  "API keys may be transmitted in plaintext.")

    def _is_retryable_error(self, error: Exception) -> Tuple[bool, Optional[float]]:
        """
        Determine if an error should trigger a retry.
        Returns (should_retry, retry_after_hint).
        """
        if isinstance(error, requests.exceptions.Timeout):
            return True, None
        if isinstance(error, requests.exceptions.ConnectionError):
            return True, None
        if isinstance(error, requests.HTTPError):
            response = error.response
            if response is not None and response.status_code in RETRYABLE_STATUS_CODES:
                # Check for Retry-After header (common with 429)
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        return True, float(retry_after)
                    except ValueError:
                        pass
                return True, None
        return False, None

    def _request_with_retry(
        self,
        method: str,
        url: str,
        provider_name: str,
        **kwargs
    ) -> requests.Response:
        """
        Execute an HTTP request with exponential backoff retry.
        
        Args:
            method: HTTP method (get, post, etc.)
            url: Request URL.
            provider_name: Name for logging (e.g., 'Ollama', 'OpenAI').
            **kwargs: Passed to requests (json, headers, timeout, etc.)
            
        Returns:
            Response object on success.
            
        Raises:
            The last exception if all retries fail.
        """
        last_error: Optional[Exception] = None
        request_func = getattr(requests, method.lower())
        
        for attempt in range(self.max_retries + 1):
            try:
                response = request_func(url, **kwargs)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                last_error = e
                should_retry, retry_after = self._is_retryable_error(e)

                # Log response body for non-retryable errors (e.g. 400)
                # to help diagnose payload issues.
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        body = e.response.text[:500]
                        logger.error("Response body from %s (HTTP %d): %s",
                                     provider_name, e.response.status_code, body)
                    except Exception:
                        pass

                if not should_retry or attempt >= self.max_retries:
                    logger.error("LLM Connection Error (%s): %s", provider_name, e)
                    raise
                
                # Calculate delay with exponential backoff + jitter
                if retry_after is not None:
                    delay = retry_after
                else:
                    delay = self.base_delay * (2 ** attempt)
                    # Add jitter (±25%)
                    delay = delay * (0.75 + random.random() * 0.5)
                    delay = min(delay, self.max_delay)
                
                status_info = ""
                if hasattr(e, 'response') and e.response is not None:
                    status_info = f" (HTTP {e.response.status_code})"
                
                logger.warning(
                    "Request to %s failed%s, retrying in %.1fs (attempt %d/%d): %s",
                    provider_name, status_info, delay, attempt + 1, self.max_retries, e
                )
                time.sleep(delay)
        
        # Should not reach here, but just in case
        raise last_error if last_error else RuntimeError("Unexpected retry loop exit")

    def detect_native_context_window(self, model: str) -> int:
        """
        Detect the model's native (maximum) context window.

        - **Ollama**: queries ``/api/show`` for the model's ``context_length``.
        - **Other providers**: defaults to 131072 (128K), which is safe for
          most cloud models (GPT-4o, Gemini, DeepSeek, etc.).

        The result is cached for the lifetime of this client instance.
        """
        if self._resolved_context_window is not None:
            return self._resolved_context_window

        native: int = 0

        if self.provider == "ollama":
            try:
                url = f"{self.base_url}/api/show"
                response = requests.post(url, json={"name": model}, timeout=15)
                response.raise_for_status()
                data = response.json()

                # Ollama returns model info with parameters or model_info
                model_info = data.get("model_info", {})
                for key, value in model_info.items():
                    if "context_length" in key.lower():
                        native = int(value)
                        logger.info("Auto-detected native context window for '%s': %d tokens", model, native)
                        break

                # Also check parameters string (older Ollama versions)
                if native <= 0:
                    params_str = data.get("parameters", "")
                    if params_str:
                        for line in params_str.split("\n"):
                            if "num_ctx" in line:
                                try:
                                    native = int(line.split()[-1])
                                    logger.info("Auto-detected context window from parameters: %d tokens", native)
                                except (ValueError, IndexError):
                                    pass
                                break

            except Exception as e:
                logger.warning("Context window auto-detection failed: %s. Using default.", e)

        # Fallback: use MODEL_SPECS if available, else 128K / 8192
        if native <= 0:
            spec = MODEL_SPECS.get(model)
            if spec:
                native = spec["context_window"]
                logger.info("Using MODEL_SPECS context window for '%s': %d tokens", model, native)
            else:
                native = 131_072 if self.provider != "ollama" else 8192
                logger.info("Using default native context window: %d tokens", native)

        self._resolved_context_window = native
        return native

    def get_effective_context_window(self, model: str) -> int:
        """
        Return the effective context window after applying ``context_window_ratio``.

        effective = native_window × context_window_ratio

        The result is floored at 2048 tokens to ensure a usable minimum.
        Cached after first computation to avoid redundant log spam.
        """
        if self._effective_context_window is not None:
            return self._effective_context_window
        native = self.detect_native_context_window(model)
        ratio = max(0.01, min(config.general.context_window_ratio, 1.0))
        effective = max(int(native * ratio), 2048)
        logger.info("Effective context window: %d tokens (%.0f%% of %d native)",
                     effective, ratio * 100, native)
        self._effective_context_window = effective
        return effective

    def get_effective_max_output_tokens(self, model: str, prompt_chars: int = 0) -> int:
        """
        Return the max output tokens for a given request, **dynamically** sized
        based on how large the prompt actually is.

        When *prompt_chars* > 0 (dynamic mode):
            estimated_input = prompt_chars / CHARS_PER_TOKEN
            output_budget   = effective_ctx - estimated_input
        When *prompt_chars* == 0 (static fallback — used by ``_compute_kb_budget``):
            output_budget   = effective_ctx × 0.25  (conservative estimate)

        The result is:
            - floored at ``min_output_tokens`` (config, default 1024)
            - capped at the model's provider-imposed maximum from ``MODEL_SPECS``
        """
        ctx = self.get_effective_context_window(model)
        min_out = max(config.general.min_output_tokens, 256)

        # Use a quieter log level for static/repeated calls to reduce log noise
        _log = logger.info if prompt_chars > 0 else logger.debug

        if prompt_chars > 0:
            # Dynamic: allocate remaining budget to output after input
            estimated_input_tokens = prompt_chars // CHARS_PER_TOKEN
            budget = ctx - estimated_input_tokens
            _log("Dynamic token allocation: %d ctx - %d est. input = %d raw output budget",
                 ctx, estimated_input_tokens, budget)
        else:
            # Static fallback (25% of effective ctx)
            budget = int(ctx * 0.25)

        # Floor at min_output_tokens
        budget = max(budget, min_out)

        # Cap at model-specific ceiling for ALL providers.
        # Without a cap, Ollama's num_predict gets the entire remaining budget
        # (e.g. 77K tokens for a small prompt against a 78K window), which
        # bloats the KV cache and cripples CPU throughput even though the
        # model stops at EOS naturally.
        spec = MODEL_SPECS.get(model)
        if spec:
            cap = spec["max_output"]
        else:
            cap = DEFAULT_MAX_OUTPUT
            logger.warning("Model '%s' not in MODEL_SPECS — applying default max_output cap of %d. "
                           "Use 'python -m cli.main add-model %s' to register it.",
                           model, cap, model)
        if budget > cap:
            _log("Clamping output tokens from %d to %d (cap for '%s')",
                 budget, cap, model)
            budget = cap

        _log("Effective max_output_tokens: %d (ctx=%d, prompt_chars=%d, min=%d)",
             budget, ctx, prompt_chars, min_out)
        return budget

    def _throttle(self) -> None:
        """
        Enforce the minimum inter-call delay (``llm_call_delay``).

        Thread-safe: concurrent worker threads will queue up behind the
        lock so that no two calls are closer together than the configured
        delay.  The wait happens *before* the HTTP request, not after,
        so the very first call of a session is never delayed.
        """
        if self._call_delay <= 0:
            return

        with self._throttle_lock:
            now = time.monotonic()
            elapsed = now - self._last_call_time
            remaining = self._call_delay - elapsed
            if remaining > 0:
                logger.debug("Throttling: waiting %.1fs before next LLM call", remaining)
                time.sleep(remaining)
            self._last_call_time = time.monotonic()

    def generate(self, prompt: str, model: str, json_mode: bool = False,
                 max_output_hint: int = 0,
                 json_array_min_items: int = 0) -> str:
        """
        Generate text completion or chat response.

        Args:
            max_output_hint: Optional soft cap on output tokens. When > 0,
                the actual ``num_predict`` / ``max_tokens`` is clamped to
                this value *if* it is lower than the model's hard cap.
                Callers can estimate this from ``max_cases_per_chunk``
                (e.g. 5 cases × 400 tok/case = 2000).
            json_array_min_items: When > 0 and using Ollama with json_mode,
                use a JSON schema with ``minItems`` instead of plain
                ``"format": "json"``.  This prevents the grammar from
                allowing ``]`` (array close) until at least N objects
                have been generated — fixing early-EOS with small models.
        """
        # Auto-register unknown Ollama models on first call so they get
        # proper MODEL_SPECS caps instead of the 4096 fallback.
        if self.provider == "ollama" and model not in MODEL_SPECS:
            try:
                self.auto_register_ollama_model(model)
            except Exception as e:
                logger.warning("Auto-registration of '%s' failed: %s", model, e)

        prompt_len = len(prompt)
        logger.info("Sending request to %s (%d chars, json_mode=%s, timeout=%ds)...",
              self.provider, prompt_len, json_mode, self.timeout)

        # Enforce inter-call delay (thread-safe) before hitting the API
        self._throttle()

        start_time = time.time()
        try:
            if self.provider == "ollama":
                result = self._generate_ollama(prompt, model, json_mode, max_output_hint,
                                               json_array_min_items=json_array_min_items)
            elif self.provider == "openai":
                result = self._generate_openai_compatible(prompt, model, json_mode)
            elif self.provider == "google":
                result = self._generate_google(prompt, model, json_mode)
            elif self.provider == "copilot":
                result = self._generate_copilot(prompt, model, json_mode)
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
            elapsed = time.time() - start_time
            logger.info("Response received in %.1fs (%d chars)", elapsed, len(result) if result else 0)
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("Request failed after %.1fs: %s", elapsed, e)
            raise

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        """Check if a model is a reasoning model that generates <think> blocks."""
        name = model.lower()
        return any(tag in name for tag in ("deepseek-r1", "r1:", "r1-", "/r1"))

    def _generate_ollama(self, prompt: str, model: str, json_mode: bool,
                         max_output_hint: int = 0,
                         json_array_min_items: int = 0) -> str:
        """
        Native Ollama API (/api/generate) with streaming progress.

        Uses ``stream: true`` so the user sees periodic progress updates
        instead of a frozen terminal for 5+ minutes.  The full response
        text is accumulated and returned as before.
        """
        url = f"{self.base_url}/api/generate"
        effective_max_out = self.get_effective_max_output_tokens(model, prompt_chars=len(prompt))

        # Apply caller's hint (e.g. from max_cases_per_chunk) if tighter
        if max_output_hint > 0 and max_output_hint < effective_max_out:
            logger.info("Applying max_output_hint: %d → %d tokens", effective_max_out, max_output_hint)
            effective_max_out = max_output_hint

        # Reasoning models (DeepSeek-R1, etc.) generate <think> blocks that
        # consume output tokens before producing useful content.  Ollama strips
        # the think blocks from the response, so if num_predict is too small the
        # model spends all tokens thinking and returns empty output.
        # Multiply the budget so enough tokens remain after the thinking phase.
        is_reasoning = self._is_reasoning_model(model)
        if is_reasoning:
            REASONING_MULTIPLIER = 4
            boosted = effective_max_out * REASONING_MULTIPLIER
            # Respect the effective context window as an upper bound
            effective_ctx = self.get_effective_context_window(model)
            estimated_input = len(prompt) // CHARS_PER_TOKEN
            ceiling = max(effective_ctx - estimated_input, effective_max_out)
            effective_max_out = min(boosted, ceiling)
            logger.info("Reasoning model detected — boosted num_predict to %d", effective_max_out)

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": config.generation.temperature,
                "top_p": config.generation.top_p,
                "num_predict": effective_max_out,
            }
        }

        # num_ctx: auto-size to actual request needs for best throughput.
        # Ollama pre-allocates the KV cache to num_ctx, so oversizing it
        # wastes memory and *cripples* CPU throughput — attention cost
        # scales linearly with context length at every decoding step.
        # Example: 78K num_ctx → ~80 s/token; 4K num_ctx → ~0.2 s/token.
        explicit_num_ctx = config.general.num_ctx
        effective_ctx = self.get_effective_context_window(model)
        if explicit_num_ctx > 0:
            optimal_ctx = explicit_num_ctx
            logger.info("num_ctx from config: %d", optimal_ctx)
        else:
            estimated_input = len(prompt) // CHARS_PER_TOKEN
            # Use the *actual* num_predict (effective_max_out) rather than
            # the model's spec max_output.  effective_max_out already
            # incorporates max_output_hint, so num_ctx stays tight to
            # the real KV-cache need — smaller context = faster CPU decode.
            output_alloc = effective_max_out
            raw_ctx = int((estimated_input + output_alloc) * 1.1)
            raw_ctx = max(2048, min(raw_ctx, effective_ctx))
            # Snap to bucket so consecutive calls reuse Ollama's KV cache.
            # Changing num_ctx between calls forces a full model reload
            # (~4-10s penalty per call).  Fixed buckets avoid this.
            _NUM_CTX_BUCKETS = [2048, 3072, 4096, 6144, 8192, 12288,
                                16384, 24576, 32768, 49152, 65536, 131072]
            optimal_ctx = raw_ctx
            for bucket in _NUM_CTX_BUCKETS:
                if bucket >= raw_ctx:
                    optimal_ctx = bucket
                    break
            optimal_ctx = min(optimal_ctx, effective_ctx)
            logger.info("num_ctx auto-sized to %d (raw %d, input~%d tok + output=%d tok)",
                        optimal_ctx, raw_ctx, estimated_input, output_alloc)
        payload["options"]["num_ctx"] = optimal_ctx

        if json_mode:
            if json_array_min_items > 0:
                # Use a JSON schema so the grammar requires at least N items
                # before ] becomes a valid token — prevents small models from
                # closing the array after just 1 object.
                payload["format"] = {
                    "type": "array",
                    "minItems": json_array_min_items,
                    "items": {"type": "object"},
                }
                logger.info("Using JSON schema with minItems=%d", json_array_min_items)
            else:
                payload["format"] = "json"

        # --- Streaming request ---
        import json as _json

        response = requests.post(
            url, json=payload, timeout=self.timeout, stream=True, verify=True,
        )
        response.raise_for_status()

        chunks: list[str] = []
        token_count = 0
        last_log_time = time.time()
        _PROGRESS_INTERVAL = 10  # log progress every N seconds
        prompt_eval_count = 0
        prompt_eval_duration = 0
        eval_count_total = 0
        eval_duration_total = 0

        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            try:
                chunk_data = _json.loads(raw_line)
            except _json.JSONDecodeError:
                continue

            token_text = chunk_data.get("response", "")
            if token_text:
                chunks.append(token_text)
                token_count += 1

            # Periodic progress logging so the user knows it's alive
            now = time.time()
            if now - last_log_time >= _PROGRESS_INTERVAL:
                logger.info("  ...generating: %d tokens so far (%.0fs elapsed)",
                            token_count, now - (last_log_time - _PROGRESS_INTERVAL + _PROGRESS_INTERVAL))
                last_log_time = now

            # Final chunk contains the telemetry
            if chunk_data.get("done", False):
                prompt_eval_count = chunk_data.get("prompt_eval_count", 0)
                prompt_eval_duration = chunk_data.get("prompt_eval_duration", 0)
                eval_count_total = chunk_data.get("eval_count", 0)
                eval_duration_total = chunk_data.get("eval_duration", 0)
                break

        result_text = "".join(chunks)

        # Log Ollama's native performance telemetry
        if eval_duration_total > 0:
            gen_tps = eval_count_total / (eval_duration_total / 1e9)
            logger.info("Ollama stats: %d prompt tok (%.1f tok/s), %d gen tok (%.1f tok/s), num_ctx=%d",
                        prompt_eval_count,
                        prompt_eval_count / (prompt_eval_duration / 1e9) if prompt_eval_duration > 0 else 0,
                        eval_count_total, gen_tps, optimal_ctx)

        return result_text

    def _generate_openai_compatible(self, prompt: str, model: str, json_mode: bool) -> str:
        """
        OpenAI-compatible API (/v1/chat/completions).
        Works with LM Studio, LocalAI, vLLM, Ollama v1, and GitHub Models.
        """
        # Resolve API key: if placeholder, fall back to GITHUB_TOKEN env var
        api_key = self.api_key
        if not api_key or api_key in ("ollama", "", "GITHUB_TOKEN"):
            api_key = os.environ.get("GITHUB_TOKEN", "") or os.environ.get("CASECRAFT_GENERAL_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OpenAI-compatible provider requires an API key. "
                "Set CASECRAFT_GENERAL_API_KEY or GITHUB_TOKEN env var, "
                "or api_key in casecraft.yaml."
            )

        # Ensure URL ends with /v1 or /v1/chat/completions correctly
        base = self.base_url.rstrip('/')
        if not base.endswith("/v1"):
            base += "/v1"
        url = f"{base}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        messages = [{"role": "user", "content": prompt}]
        
        effective_max_out = self.get_effective_max_output_tokens(model, prompt_chars=len(prompt))
        payload = {
            "model": model,
            "messages": messages,
            "temperature": config.generation.temperature,
            "top_p": config.generation.top_p,
            "max_tokens": effective_max_out,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = self._request_with_retry(
            "post", url, "OpenAI/Generic",
            headers=headers, json=payload, timeout=self.timeout, verify=True
        )
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        return ""

    def _generate_google(self, prompt: str, model: str, json_mode: bool) -> str:
        """
        Google Gemini API (REST).
        Docs: https://ai.google.dev/api/rest/v1beta/models/generateContent
        """
        # Base URL should be: https://generativelanguage.googleapis.com/v1beta
        base = self.base_url.rstrip('/')
        url = f"{base}/models/{model}:generateContent"

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }

        effective_max_out = self.get_effective_max_output_tokens(model, prompt_chars=len(prompt))
        generation_config = {
            "temperature": config.generation.temperature,
            "topP": config.generation.top_p,
            "maxOutputTokens": effective_max_out,
        }

        if json_mode:
            generation_config["response_mime_type"] = "application/json"

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": generation_config
        }

        response = self._request_with_retry(
            "post", url, "Google Gemini",
            headers=headers, json=payload, timeout=self.timeout, verify=True
        )
        data = response.json()
        
        # Parse response
        # { "candidates": [ { "content": { "parts": [ { "text": "..." } ] } } ] }
        if "candidates" in data and len(data["candidates"]) > 0:
            content = data["candidates"][0].get("content", {})
            parts = content.get("parts", [])
            if parts:
                return parts[0].get("text", "")
        return ""

    def _generate_copilot(self, prompt: str, model: str, json_mode: bool) -> str:
        """
        GitHub Models API (OpenAI-compatible).
        Uses the GitHub Models inference endpoint with a GitHub PAT.
        Docs: https://docs.github.com/en/github-models
        """
        # Resolve API key: prefer explicit config, then GITHUB_TOKEN env var
        api_key = self.api_key
        if not api_key or api_key in ("ollama", "", "GITHUB_TOKEN"):
            api_key = os.environ.get("GITHUB_TOKEN", "")
        if not api_key:
            raise ValueError(
                "GitHub Models requires a GitHub PAT. Set GITHUB_TOKEN env var "
                "or api_key in casecraft.yaml (needs 'models:read' scope)."
            )

        # Use user-configured base_url if it looks like a GitHub Models endpoint,
        # otherwise fall back to the default GITHUB_MODELS_BASE_URL.
        base = self.base_url.rstrip('/')
        if "models.github.ai" in base or "models.inference.ai.azure.com" in base:
            # Strip trailing /v1 if present (we build the path ourselves)
            if base.endswith("/v1"):
                base = base[:-3]
            endpoint_base = base
        else:
            endpoint_base = GITHUB_MODELS_BASE_URL

        url = f"{endpoint_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        effective_max_out = self.get_effective_max_output_tokens(model, prompt_chars=len(prompt))
        messages = [{"role": "user", "content": prompt}]

        # Reasoning models (o1, o3, o4, etc.) don't accept temperature/top_p
        is_reasoning = model.startswith(("o1", "o3", "o4"))

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            # GitHub Models API follows latest OpenAI spec:
            # use 'max_completion_tokens' (not deprecated 'max_tokens').
            "max_completion_tokens": effective_max_out,
        }

        if not is_reasoning:
            payload["temperature"] = config.generation.temperature
            payload["top_p"] = config.generation.top_p

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = self._request_with_retry(
            "post", url, "GitHub Models (Copilot)",
            headers=headers, json=payload, timeout=self.timeout, verify=True
        )
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        return ""

    def auto_register_ollama_model(self, model: str) -> Dict[str, int]:
        """
        Query Ollama for a model's specs and register it in MODEL_SPECS at runtime.

        Auto-detects context_window from Ollama's /api/show endpoint.
        Sets max_output based on model size heuristics:
          - <2B params  → 2,048
          - <10B params → 4,096
          - ≥10B params → 8,192

        Returns the spec dict that was registered.
        Raises RuntimeError if the model cannot be queried.
        """
        if model in MODEL_SPECS:
            logger.info("Model '%s' already in MODEL_SPECS.", model)
            return MODEL_SPECS[model]

        url = f"{self.base_url}/api/show"
        try:
            response = requests.post(url, json={"name": model}, timeout=15)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Cannot query Ollama for model '{model}': {e}")

        data = response.json()
        model_info = data.get("model_info", {})

        # Detect context_window
        context_window = 0
        for key, value in model_info.items():
            if "context_length" in key.lower():
                context_window = int(value)
                break
        if context_window <= 0:
            context_window = 8_192  # safe fallback

        # Detect parameter count for max_output heuristic
        param_count = model_info.get("general.parameter_count", 0)
        if param_count and param_count < 2_000_000_000:
            max_output = 2_048
        elif param_count and param_count < 10_000_000_000:
            max_output = 4_096
        else:
            max_output = 8_192

        spec = {"context_window": context_window, "max_output": max_output}
        MODEL_SPECS[model] = spec
        logger.info("Registered '%s' in MODEL_SPECS: %s", model, spec)
        return spec

    def unload_model(self, model: str) -> bool:
        """
        Unload a model from Ollama memory by setting keep_alive to 0.
        Only works with the 'ollama' provider.
        Returns True if successful, False otherwise.
        """
        if self.provider != "ollama":
            return False
        
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": "",
            "keep_alive": 0
        }
        try:
            response = requests.post(url, json=payload, timeout=30, verify=True)
            response.raise_for_status()
            logger.info("Model '%s' unloaded from Ollama memory.", model)
            return True
        except requests.RequestException as e:
            logger.warning("Failed to unload model from Ollama: %s", e)
            return False

# Global/Singleton instance
llm_client = LLMClient()
