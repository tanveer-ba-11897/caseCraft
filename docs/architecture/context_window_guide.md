
## Context Window Behavior Explained

CaseCraft now defaults `context_window_size` to `-1`.

### What does `-1` mean?
When `context_window_size` is set to `-1`:
1.  **CaseCraft does NOT send a `num_ctx` Limit**: It tells Ollama "Use your default settings for this model".
2.  **Model Defaults**: Most Ollama models default to 2048 or 4096.
3.  **To Use Maximum Capacity**: You must currently set this manually (e.g., 32768, 128000) because there is no API way to ask a model "What is your max?" dynamically before sending the request.

### Recommendation
If you switch models frequently:
1.  Keep the default `-1` for general speed and low memory usage.
2.  When using a specific high-context model (e.g., `llama3.1` which supports 128k), set the ENV var just for that run:
    ```powershell
    $env:CASECRAFT_GENERAL_CONTEXT_WINDOW_SIZE=32768
    python -m cli.start
    ```
