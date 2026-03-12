
## Context Window & Dynamic Token Allocation

CaseCraft dynamically allocates input and output tokens based on the actual prompt size, with per-model hard caps.

### How It Works

1. **Auto-detection**: CaseCraft detects the model's native context window from Ollama (`/api/show`). For cloud models, per-model specs from `MODEL_SPECS` provide accurate context windows (e.g., gpt-4o: 128K, Phi-4: 16K).

2. **`context_window_ratio`** (0.0–1.0): Controls what fraction of the native window to use.
   - `0.75` (default) = use 75% of the model's native window
   - `0.5` = use half — reduces RAM usage on constrained machines
   - `1.0` = use the full native window

3. **Dynamic output allocation**: Output tokens are allocated **dynamically** based on actual prompt size:
   - `effective_ctx = native_window × context_window_ratio`
   - `estimated_input = len(prompt) / 4`  (approx 4 chars per token)
   - `output_budget = effective_ctx - estimated_input`
   - Result is floored at `min_output_tokens` (default 1024)
   - Result is capped at the model's hard limit from `MODEL_SPECS`

4. **`min_output_tokens`** (default 1024): Minimum guaranteed output tokens, even when the prompt is very large.

### Dynamic Allocation Examples

| Scenario | Effective CTX | Prompt Chars | Est. Input Tokens | Raw Output Budget | Model Cap | Final Output |
|---|---|---|---|---|---|---|
| gpt-4o, small prompt | 96,000 | 8,000 | 2,000 | 94,000 | 16,384 | **16,384** (capped) |
| gpt-4o, large prompt | 96,000 | 320,000 | 80,000 | 16,000 | 16,384 | **16,000** |
| gpt-4o, huge prompt  | 96,000 | 400,000 | 100,000 | -4,000 | 16,384 | **1,024** (floored) |
| Phi-4, small prompt  | 12,288 | 4,000 | 1,000 | 11,288 | 4,096 | **4,096** (capped) |
| llama3.2:1b (Ollama) | 65,536 | 12,000 | 3,000 | 62,536 | — | **62,536** (uncapped) |

### Per-Model Specifications (`MODEL_SPECS`)

Hard limits from each provider. CaseCraft enforces these automatically:

| Model | Context Window | Max Output |
|---|---|---|
| gpt-4o / gpt-4o-mini | 128,000 | 16,384 |
| o1 / o3-mini / o4-mini | 200,000 | 100,000 |
| DeepSeek-R1 | 128,000 | 16,384 |
| Phi-4 | 16,384 | 4,096 |
| Meta-Llama-3.1-*B | 128,000 | 4,096 |
| Mistral-large | 128,000 | 4,096 |
| xai-grok-3 | 131,072 | 16,384 |

### Configuration

In `casecraft.yaml`:

```yaml
general:
  context_window_ratio: 0.75  # 75% of native window
  min_output_tokens: 1024     # Floor — output never drops below this
```

Or via environment variables:

```powershell
$env:CASECRAFT_GENERAL_CONTEXT_WINDOW_RATIO=0.5
$env:CASECRAFT_GENERAL_MIN_OUTPUT_TOKENS=2048
python -m cli.main features/your_feature.txt
```
