/**
 * VS Code Language Model Bridge
 *
 * Wraps the `vscode.lm` API to provide a simple `generate()` function
 * that can be called when the Python bridge asks for an LLM completion
 * via the "vscode" provider.
 */

import * as vscode from 'vscode';

export class LmBridge {
    private cachedModel: vscode.LanguageModelChat | null = null;
    private cachedModelId = '';

    /**
     * Select a Copilot-provided language model.
     *
     * @param preferredModel  Substring to match against model id/name
     *                        (e.g. "gpt-4o", "claude-3.5-sonnet").
     */
    async selectModel(
        preferredModel?: string
    ): Promise<vscode.LanguageModelChat> {
        if (
            this.cachedModel &&
            (!preferredModel || this.cachedModelId === preferredModel)
        ) {
            return this.cachedModel;
        }

        const models = await vscode.lm.selectChatModels({ vendor: 'copilot' });

        if (!models || models.length === 0) {
            throw new Error(
                'No Copilot language models available. ' +
                    'Ensure you have an active GitHub Copilot subscription.'
            );
        }

        let selected: vscode.LanguageModelChat | undefined;

        if (preferredModel) {
            const lower = preferredModel.toLowerCase();
            selected = models.find(
                (m) =>
                    m.id.toLowerCase().includes(lower) ||
                    m.name.toLowerCase().includes(lower)
            );
        }

        // Fall back to first available model
        if (!selected) {
            selected = models[0];
        }

        this.cachedModel = selected;
        this.cachedModelId = selected.id;
        return selected;
    }

    /**
     * Generate a completion using a Copilot model.
     * Retries up to 2 times on transient failures.
     *
     * @returns The full response text (all streamed tokens concatenated).
     */
    async generate(
        prompt: string,
        preferredModel?: string,
        _jsonMode?: boolean,
        token?: vscode.CancellationToken
    ): Promise<string> {
        const maxRetries = 2;
        let lastError: Error | undefined;

        for (let attempt = 0; attempt <= maxRetries; attempt++) {
            try {
                const model = await this.selectModel(preferredModel);
                const messages = [vscode.LanguageModelChatMessage.User(prompt)];
                const response = await model.sendRequest(messages, {}, token);

                let result = '';
                for await (const chunk of response.text) {
                    if (token?.isCancellationRequested) {
                        throw new Error('LLM request was cancelled');
                    }
                    result += chunk;
                }
                return result;
            } catch (err: unknown) {
                lastError = err instanceof Error ? err : new Error(String(err));

                // Don't retry on cancellation
                if (token?.isCancellationRequested) {
                    throw lastError;
                }

                // Don't retry on "no models available"
                if (lastError.message.includes('No Copilot')) {
                    throw lastError;
                }

                // Invalidate cached model on failure — it may have become stale
                this.cachedModel = null;
                this.cachedModelId = '';

                if (attempt < maxRetries) {
                    // Exponential backoff: 1s, 2s
                    await new Promise((r) =>
                        setTimeout(r, 1000 * (attempt + 1))
                    );
                }
            }
        }

        throw lastError ?? new Error('LLM generation failed after retries');
    }

    /**
     * Return a summary of all available Copilot models.
     */
    async listModels(): Promise<
        Array<{ id: string; name: string; vendor: string }>
    > {
        const models = await vscode.lm.selectChatModels({ vendor: 'copilot' });
        return (models || []).map((m) => ({
            id: m.id,
            name: m.name,
            vendor: m.vendor,
        }));
    }
}
