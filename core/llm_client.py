
import requests
from typing import Dict, Any, Optional
from core.config import config

class LLMClient:
    """
    Abstract client for interacting with LLM backends.
    Supports 'ollama' (native) and 'openai' (compatible) formats.
    """
    
    def __init__(self):
        self.provider = config.general.llm_provider
        self.base_url = config.general.base_url
        self.api_key = config.general.api_key.get_secret_value()
        self.timeout = config.general.timeout

        # Warn if using unencrypted HTTP for non-localhost addresses
        if self.base_url.startswith("http://") and not any(
            h in self.base_url for h in ("localhost", "127.0.0.1", "0.0.0.0")
        ):
            print("WARNING: LLM base_url uses unencrypted HTTP for a non-localhost address. "
                  "API keys may be transmitted in plaintext.")

    def generate(self, prompt: str, model: str, json_mode: bool = False) -> str:
        """
        Generate text completion or chat response.
        """
        if self.provider == "ollama":
            return self._generate_ollama(prompt, model, json_mode)
        elif self.provider == "openai":
            return self._generate_openai_compatible(prompt, model, json_mode)
        elif self.provider == "google":
            return self._generate_google(prompt, model, json_mode)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _generate_ollama(self, prompt: str, model: str, json_mode: bool) -> str:
        """
        Native Ollama API (/api/generate).
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.generation.temperature,
                "top_p": config.generation.top_p,
                "num_predict": config.general.max_output_tokens,
            }
        }

        # If user explicitly sets context window (e.g. 8192, 16384), pass it. 
        # If -1, do NOT pass it, letting Ollama use the Modelfile default (often 2048 or model native max).
        if config.general.context_window_size > 0:
            payload["options"]["num_ctx"] = config.general.context_window_size
        
        if json_mode:
            payload["format"] = "json"

        try:
            response = requests.post(url, json=payload, timeout=self.timeout, verify=True)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.RequestException as e:
            print(f"LLM Connection Error (Ollama): {e}")
            raise

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
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": config.generation.temperature,
            "top_p": config.generation.top_p,
            "max_tokens": config.general.max_output_tokens,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout, verify=True)
            response.raise_for_status()
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            return ""
        except requests.RequestException as e:
            print(f"LLM Connection Error (OpenAI/Generic): {e}")
            raise

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

        generation_config = {
            "temperature": config.generation.temperature,
            "topP": config.generation.top_p,
            "maxOutputTokens": config.general.max_output_tokens,
        }

        if json_mode:
            generation_config["response_mime_type"] = "application/json"

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": generation_config
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout, verify=True)
            response.raise_for_status()
            data = response.json()
            
            # Parse response
            # { "candidates": [ { "content": { "parts": [ { "text": "..." } ] } } ] }
            if "candidates" in data and len(data["candidates"]) > 0:
                content = data["candidates"][0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            return ""
        except requests.RequestException as e:
            print(f"LLM Connection Error (Google Gemini): {e}")
            # Try to print more detail if available
            if hasattr(e, 'response') and e.response is not None:
                print(f"Google API Response: {e.response.text}")
            raise

# Global/Singleton instance
llm_client = LLMClient()
