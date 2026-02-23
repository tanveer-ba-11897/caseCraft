# Local LLM & Performance Options

This guide outlines alternatives to Llama 3.1 8B for faster or better execution in CaseCraft.

## Quick Recommendation Map

| If you want... | Recommended Model | Engine/Provider | Pros | Cons |
| :--- | :--- | :--- | :--- | :--- |
| **Maximum Speed (Local)** | **Llama 3.2 3B** | Ollama | 🚀 2-3x faster | Weaker at complex logic |
| **Best Coding Capability** | **Qwen 2.5 7B** | Ollama | 🧠 Smarter (80% vs 72% HumanEval) | Same speed as Llama 3.1 |
| **Blazing Speed (Cloud)** | **Llama 3.3 70B** | **Groq** | ⚡ Instant (500+ tok/s) | Free tier has rate limits |

---

## 1. Faster Local Models (Ollama)

If Llama 3.1 8B is too slow, your best bet is a smaller, optimized model.

### Option A: Llama 3.2 3B (Fastest)
Optimized for speed and edge devices.
- **Speed**: ~50-70 tokens/sec (CPU)
- **Quality**: Good for simple tasks, but may struggle with complex reasoning.
- **Setup**:
  ```bash
  ollama pull llama3.2:3b
  ```
- **Config (`casecraft.yaml`)**:
  ```yaml
  general:
    model: "llama3.2:3b"
  ```

### Option B: Qwen 2.5 7B (Smartest)
Often beats Llama 3.1 8B in coding benchmarks (HumanEval, LiveCodeBench).
- **Speed**: Similar to Llama 3.1 8B (it's the same size).
- **Quality**: Excellent. If you found Llama 3.1 "dumb", try this.
- **Setup**:
  ```bash
  ollama pull qwen2.5:7b
  ```
- **Config (`casecraft.yaml`)**:
  ```yaml
  general:
    model: "qwen2.5:7b"
  ```

---

## 2. Using Groq (Cloud / Free Tier)

Groq uses specialized LPUs to deliver 300-1000 tokens/second. It feels instant.

**Free Tier Limits (2025):**
- **Requests**: ~14,400/day (Generous)
- **Rate Limit**: ~30 requests/minute (This is the catch. CaseCraft might hit this in a loop).
- **Cost**: Free.

**Setup**:
1. Get an API Key from [console.groq.com](https://console.groq.com).
2. Update `casecraft.yaml`:
   ```yaml
   general:
     llm_provider: "openai"  # Groq is OpenAI-compatible
     base_url: "https://api.groq.com/openai/v1"
     api_key: "gsk_..."      # Your Key
     model: "llama-3.3-70b-versatile" # Or "llama3-8b-8192"
   ```

---

## 3. Llama.cpp (Backend Alternative)

**"How is it?"**:
Llama.cpp is the engine that actually runs these models. **Ollama** is just a user-friendly wrapper around Llama.cpp.
- **Direct Llama.cpp**: Slightly lower memory overhead (no background service), potentially 5-10% faster.
- **Ollama**: Much easier to manage.
- **Verdict**: Unless you are extremely RAM constrained, switching from Ollama to raw Llama.cpp Server won't give you a massive speed boost. The model size is the bottleneck.

**To use Llama.cpp Server:**
1. Download `llama-server.exe` from [GitHub](https://github.com/ggerganov/llama.cpp/releases).
2. Download a `.gguf` model file (e.g. from HuggingFace).
3. Run: `./llama-server.exe -m model.gguf -c 8192 --port 8080 --host 0.0.0.0`
4. Config `casecraft.yaml`:
   ```yaml
   general:
     llm_provider: "openai"
     base_url: "http://localhost:8080/v1"
     api_key: "none"
   ```
