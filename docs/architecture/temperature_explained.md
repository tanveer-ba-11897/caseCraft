
# Understanding "Temperature" in LLMs

## What is it?
**Temperature** is a parameter (usually between `0.0` and `1.0` or higher) that controls the **randomness** or **creativity** of the model's output.

## How it works (Simplified)
When an LLM predicts the next word, it assigns a probability to every possible word in its vocabulary.

**Example**: "The sky is..."
*   "blue" (80%)
*   "cloudy" (15%)
*   "green" (0.1%)

### Low Temperature (e.g., 0.1)
The model becomes **Deterministic** and **Conservative**.
*   It almost always picks the #1 most likely word ("blue").
*   **Result**: Consistent, logical, factual, less prone to mistakes.
*   **Best for**: Coding, Math, QA Extraction (CaseCraft default: `0.2`).

### High Temperature (e.g., 0.8 - 1.2)
The model becomes **Creative** and **Diverse**.
*   It flattens the probabilities, making "cloudy" or even "green" viable options to be picked.
*   **Result**: Varied phrasing, interesting stories, brainstorming.
*   **Risk**: Higher chance of hallucination or nonsense ("The sky is lasagna").
*   **Best for**: Creative writing, poetry, generating varied marketing copy.

## In CaseCraft
CaseCraft defaults to `0.2`.
*   **Why?**: We want **Test Cases** to follow a strict JSON schema and grounded facts from the documentation. We do *not* want the model to get creative and invent features that don't exist.
