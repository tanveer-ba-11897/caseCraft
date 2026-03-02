
import logging
import random
import requests
import time
from typing import Callable, Dict, Any, Optional, Tuple
from core.config import config

logger = logging.getLogger("casecraft.llm")

# HTTP status codes that should trigger a retry
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

# ── VS Code LLM Proxy ────────────────────────────────────────────────────────
# When llm_provider is "vscode", LLM calls are proxied through the bridge
# server to the VS Code Language Model API (Copilot models).  The callback
# is set by bridge_server.py at startup.
_vscode_llm_callback: Optional[Callable[..., str]] = None


def set_vscode_llm_callback(callback: Callable[..., str]) -> None:
    """Register the VS Code LLM proxy function (called by bridge_server.py)."""
    global _vscode_llm_callback
    _vscode_llm_callback = callback
    logger.info("VS Code LLM proxy callback registered")

class LLMClient:
    """
    Abstract client for interacting with LLM backends.
    Supports 'ollama' (native) and 'openai' (compatible) formats.
    Features exponential backoff with jitter for transient failures.
    """
    
    def __init__(self):
        self.provider = config.general.llm_provider
        self.base_url = config.general.base_url
        self.api_key = config.general.api_key.get_secret_value()
        self.timeout = config.general.timeout
        self._resolved_context_window: Optional[int] = None
        
        # Retry configuration (could be moved to config if needed)
        self.max_retries = 3
        self.base_delay = 1.0  # seconds
        self.max_delay = 60.0  # seconds

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

    def detect_context_window(self, model: str) -> int:
        """
        Auto-detect the model's native context window from the Ollama API.
        Falls back to config.general.context_window_size if detection fails.
        Result is capped at config.general.max_context_window.
        """
        if self._resolved_context_window is not None:
            return self._resolved_context_window

        detected = config.general.context_window_size
        max_cap = config.general.max_context_window

        if config.general.auto_detect_context_window and self.provider == "ollama":
            try:
                url = f"{self.base_url}/api/show"
                response = requests.post(url, json={"name": model}, timeout=15)
                response.raise_for_status()
                data = response.json()

                # Ollama returns model info with parameters or model_info
                model_info = data.get("model_info", {})
                # Look for context_length in model_info keys
                for key, value in model_info.items():
                    if "context_length" in key.lower():
                        detected = int(value)
                        logger.info("Auto-detected context window for '%s': %d tokens", model, detected)
                        break

                # Also check parameters string (older Ollama)
                if detected == config.general.context_window_size:
                    params_str = data.get("parameters", "")
                    if params_str:
                        for line in params_str.split("\n"):
                            if "num_ctx" in line:
                                try:
                                    detected = int(line.split()[-1])
                                    logger.info("Auto-detected context window from parameters: %d tokens", detected)
                                except (ValueError, IndexError):
                                    pass
                                break

            except Exception as e:
                logger.warning("Context window auto-detection failed: %s. Using configured value.", e)

        # If still -1 (no explicit config, no detection), use conservative default
        if detected <= 0:
            detected = 8192
            logger.info("No context window detected, using default: %d tokens", detected)

        # Apply the max cap from YAML
        if max_cap > 0 and detected > max_cap:
            logger.info("Capping context window: %d → %d tokens (max_context_window)", detected, max_cap)
            detected = max_cap

        self._resolved_context_window = detected
        return detected

    def get_effective_context_window(self, model: str) -> int:
        """
        Return the effective context window to use for this model.
        Uses explicit config if set, otherwise auto-detects and caps.
        """
        explicit = config.general.context_window_size
        if explicit > 0 and not config.general.auto_detect_context_window:
            # User explicitly set a value and disabled auto-detect
            return explicit
        return self.detect_context_window(model)

    def get_effective_max_output_tokens(self, model: str) -> int:
        """
        Return the effective max output tokens for this model.
        When max_output_tokens is -1, auto-scales to output_token_ratio of
        the effective context window, capped at max_output_tokens_cap.
        """
        explicit = config.general.max_output_tokens
        if explicit > 0:
            return explicit

        # Auto-scale: use a fraction of the effective context window
        ctx = self.get_effective_context_window(model)
        ratio = max(0.05, min(config.general.output_token_ratio, 0.5))
        scaled = int(ctx * ratio)

        # Apply floor and cap
        scaled = max(scaled, 1024)  # never below 1024
        cap = config.general.max_output_tokens_cap
        if cap > 0 and scaled > cap:
            scaled = cap

        logger.info("Auto-scaled max_output_tokens: %d (%.0f%% of %d ctx, cap=%d)",
              scaled, ratio * 100, ctx, cap)
        return scaled

    def generate(self, prompt: str, model: str, json_mode: bool = False) -> str:
        """
        Generate text completion or chat response.
        """
        prompt_len = len(prompt)
        logger.info("Sending request to %s (%d chars, json_mode=%s, timeout=%ds)...",
              self.provider, prompt_len, json_mode, self.timeout)
        start_time = time.time()
        try:
            if self.provider == "ollama":
                result = self._generate_ollama(prompt, model, json_mode)
            elif self.provider == "openai":
                result = self._generate_openai_compatible(prompt, model, json_mode)
            elif self.provider == "google":
                result = self._generate_google(prompt, model, json_mode)
            elif self.provider == "vscode":
                result = self._generate_vscode(prompt, model, json_mode)
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
            elapsed = time.time() - start_time
            logger.info("Response received in %.1fs (%d chars)", elapsed, len(result) if result else 0)
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("Request failed after %.1fs: %s", elapsed, e)
            raise

    def _generate_ollama(self, prompt: str, model: str, json_mode: bool) -> str:
        """
        Native Ollama API (/api/generate).
        """
        url = f"{self.base_url}/api/generate"
        effective_max_out = self.get_effective_max_output_tokens(model)
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.generation.temperature,
                "top_p": config.generation.top_p,
                "num_predict": effective_max_out,
            }
        }

        # If user explicitly sets context window (e.g. 8192, 16384), pass it. 
        # If -1, auto-detect from model and cap at max_context_window.
        effective_ctx = self.get_effective_context_window(model)
        payload["options"]["num_ctx"] = effective_ctx
        
        if json_mode:
            payload["format"] = "json"

        response = self._request_with_retry(
            "post", url, "Ollama",
            json=payload, timeout=self.timeout, verify=True
        )
        return response.json().get("response", "")

    def _generate_openai_compatible(self, prompt: str, model: str, json_mode: bool) -> str:
        """
        OpenAI-compatible API (/v1/chat/completions).
        Works with LM Studio, LocalAI, vLLM, and Ollama v1.
        """
        # Ensure URL ends with /v1 or /v1/chat/completions correctly
        base = self.base_url.rstrip('/')
        if not base.endswith("/v1"):
            base += "/v1"
        url = f"{base}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        messages = [{"role": "user", "content": prompt}]
        
        effective_max_out = self.get_effective_max_output_tokens(model)
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

        effective_max_out = self.get_effective_max_output_tokens(model)
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

    def _generate_vscode(self, prompt: str, model: str, json_mode: bool) -> str:
        """
        Proxy LLM generation through the VS Code Language Model API.

        The bridge_server.py registers a callback via set_vscode_llm_callback()
        that sends the prompt to TypeScript over stdout and blocks until the
        response arrives on stdin.
        """
        if _vscode_llm_callback is None:
            raise RuntimeError(
                "VS Code LLM proxy not available. "
                "The 'vscode' provider can only be used when running inside "
                "the VS Code CaseCraft extension (bridge_server.py)."
            )
        return _vscode_llm_callback(prompt, model, json_mode)

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
