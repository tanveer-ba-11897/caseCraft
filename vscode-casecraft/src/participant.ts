/**
 * CaseCraft Chat Participant
 *
 * Registers as `@casecraft` inside VS Code Copilot Chat and provides
 * slash commands:
 *
 *   /generate — Generate test cases from a feature document
 *   /query    — Search the product knowledge base
 *   /ingest   — Ingest documents or URLs into the KB
 *   /config   — View current CaseCraft configuration
 *   (none)    — Freeform QA chat with RAG context
 */

import * as vscode from 'vscode';
import * as path from 'path';
import { PythonBridge } from './pythonBridge';
import { CaseCraftConfig } from './config';

const PARTICIPANT_ID = 'casecraft.agent';

/** Icons shown alongside streamed progress updates in the chat panel. */
const progressIcons: Record<string, string> = {
    starting: '📤',
    parsing: '📄',
    retrieval: '🔍',
    retrieval_done: '✅',
    generating: '⚙️',
    post_processing: '🧹',
    deduplication: '🧬',
    cross_reference: '📋',
    reviewing: '🔎',
    complete: '🎉',
};

export class CaseCraftParticipant implements vscode.Disposable {
    private participant: vscode.ChatParticipant;
    private bridge: PythonBridge;
    private config: CaseCraftConfig;

    constructor(
        context: vscode.ExtensionContext,
        bridge: PythonBridge,
        config: CaseCraftConfig
    ) {
        this.bridge = bridge;
        this.config = config;

        this.participant = vscode.chat.createChatParticipant(
            PARTICIPANT_ID,
            this.handleRequest.bind(this)
        );

        this.participant.iconPath = vscode.Uri.joinPath(
            context.extensionUri,
            'icon.png'
        );

        context.subscriptions.push(this.participant);
    }

    /* ── Top-level dispatcher ───────────────────────────────────────────── */

    private async handleRequest(
        request: vscode.ChatRequest,
        context: vscode.ChatContext,
        stream: vscode.ChatResponseStream,
        token: vscode.CancellationToken
    ): Promise<vscode.ChatResult> {
        try {
            switch (request.command) {
                case 'generate':
                    return await this.handleGenerate(request, stream, token);
                case 'query':
                    return await this.handleQuery(request, stream, token);
                case 'ingest':
                    return await this.handleIngest(request, stream, token);
                case 'config':
                    return await this.handleConfig(stream);
                default:
                    return await this.handleFreeform(
                        request,
                        context,
                        stream,
                        token
                    );
            }
        } catch (err: unknown) {
            const message =
                err instanceof Error ? err.message : String(err);
            stream.markdown(`\n\n**Error:** ${message}`);
            return {};
        }
    }

    /* ── Helpers ─────────────────────────────────────────────────────────── */

    /**
     * Extract a file path from the chat request.
     * Checks `#file` references first, then falls back to parsing the prompt.
     */
    private extractFilePath(
        request: vscode.ChatRequest
    ): string | undefined {
        // VS Code #file references
        for (const ref of request.references) {
            if (ref.value instanceof vscode.Uri) {
                return ref.value.fsPath;
            }
        }

        // Fallback: regex on the prompt text
        const text = request.prompt.trim();
        if (text) {
            const match = text.match(
                /(?:^|\s)((?:[\w./\\-]+\.(?:pdf|txt|md)))/i
            );
            if (match) {
                let filePath = match[1];
                // Resolve relative paths against project root
                if (!path.isAbsolute(filePath)) {
                    filePath = path.join(
                        this.config.getProjectRoot(),
                        filePath
                    );
                }
                return filePath;
            }
        }

        return undefined;
    }

    /* ── /generate ──────────────────────────────────────────────────────── */

    private async handleGenerate(
        request: vscode.ChatRequest,
        stream: vscode.ChatResponseStream,
        token: vscode.CancellationToken
    ): Promise<vscode.ChatResult> {
        const filePath = this.extractFilePath(request);
        if (!filePath) {
            stream.markdown(
                'Please provide a feature document to generate tests from.\n\n' +
                    '**Usage:**\n' +
                    '- `@casecraft /generate features/your_file.pdf`\n' +
                    '- Or drag a file into the chat using `#file`.\n'
            );
            return {};
        }

        stream.progress('Starting test generation…');

        // Track stages we've already reported to avoid duplicate messages
        const reportedStages = new Set<string>();

        const result = await this.bridge.sendRequest(
            'generate',
            {
                file_path: filePath,
                app_type: this.config.getAppType(),
                dedup_semantic: this.config.getSemanticDedup(),
                reviewer_pass: this.config.getReviewerPass(),
            },
            (step, detail) => {
                // Show the progress spinner text
                stream.progress(detail);

                // Stream a markdown activity log for key stages
                if (!reportedStages.has(step)) {
                    reportedStages.add(step);
                    const icon = progressIcons[step] ?? '🔄';
                    stream.markdown(`${icon} ${detail}\n\n`);
                }
            },
            token
        );

        // ── Render results ────────────────────────────────────────────────
        const count = result.test_case_count as number;
        const featureName = result.feature_name as string;
        const testCases = (result.test_cases as Array<Record<string, unknown>>) ?? [];

        stream.markdown(`## Test Generation Complete\n\n`);
        stream.markdown(
            `**${count}** test cases generated for **${featureName}**\n\n`
        );

        // Preview table
        if (testCases.length > 0) {
            const previewCount = Math.min(testCases.length, 5);
            stream.markdown(
                `### Preview (first ${previewCount} of ${count} cases)\n\n`
            );
            stream.markdown(`| # | Test Case | Type | Priority |\n`);
            stream.markdown(`|---|-----------|------|----------|\n`);
            for (let i = 0; i < previewCount; i++) {
                const tc = testCases[i];
                stream.markdown(
                    `| ${i + 1} | ${tc.test_case} | ${tc.test_type} | ${tc.priority} |\n`
                );
            }
            stream.markdown('\n');
        }

        // Buttons to open output files
        stream.markdown(`**Output files:**\n\n`);

        if (result.json_path) {
            const jsonUri = vscode.Uri.file(result.json_path as string);
            stream.button({
                command: 'vscode.open',
                arguments: [jsonUri],
                title: '📄 Open JSON',
            });
        }
        if (result.excel_path) {
            const excelUri = vscode.Uri.file(result.excel_path as string);
            stream.button({
                command: 'vscode.open',
                arguments: [excelUri],
                title: '📊 Open Excel',
            });
        }

        return {};
    }

    /* ── /query ─────────────────────────────────────────────────────────── */

    private async handleQuery(
        request: vscode.ChatRequest,
        stream: vscode.ChatResponseStream,
        token: vscode.CancellationToken
    ): Promise<vscode.ChatResult> {
        const query = request.prompt.trim();
        if (!query) {
            stream.markdown(
                'Please provide a search query.\n\n' +
                    '**Usage:** `@casecraft /query login authentication flow`\n'
            );
            return {};
        }

        stream.progress('Searching knowledge base…');

        const result = await this.bridge.sendRequest('query', {
            query,
            top_k: 5,
        }, undefined, token);

        const chunks =
            (result.chunks as Array<Record<string, unknown>>) ?? [];
        const count = result.count as number;

        if (chunks.length === 0) {
            stream.markdown(
                'No relevant results found in the knowledge base.\n'
            );
            return {};
        }

        stream.markdown(`## Knowledge Base Results\n\n`);
        stream.markdown(`Found **${count}** relevant chunks:\n\n`);

        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i];
            stream.markdown(`### Result ${i + 1}\n`);
            stream.markdown(`*Source: ${chunk.source}*\n\n`);
            stream.markdown(`${chunk.text}\n\n---\n\n`);
        }

        return {};
    }

    /* ── /ingest ────────────────────────────────────────────────────────── */

    private async handleIngest(
        request: vscode.ChatRequest,
        stream: vscode.ChatResponseStream,
        token: vscode.CancellationToken
    ): Promise<vscode.ChatResult> {
        const prompt = request.prompt.trim();
        if (!prompt) {
            stream.markdown(
                'Please provide a source to ingest.\n\n' +
                    '**Usage:**\n' +
                    '- `@casecraft /ingest docs ./knowledge_base/raw/`\n' +
                    '- `@casecraft /ingest url https://docs.example.com/page`\n' +
                    '- `@casecraft /ingest sitemap https://docs.example.com/sitemap.xml`\n'
            );
            return {};
        }

        // Parse "docs ./path" or "url https://..." or "sitemap https://..."
        const parts = prompt.split(/\s+/, 2);
        const sourceType = parts[0]?.toLowerCase() ?? 'docs';
        const sourcePath = parts[1] ?? prompt;

        stream.progress(`Ingesting from ${sourcePath}…`);

        const result = await this.bridge.sendRequest(
            'ingest',
            {
                source_type: sourceType,
                path: sourcePath,
            },
            (_step, detail) => {
                stream.progress(detail);
            },
            token
        );

        stream.markdown(`## Ingestion Complete\n\n`);
        stream.markdown(
            `- **Documents processed:** ${result.documents}\n` +
                `- **Chunks created:** ${result.chunks}\n` +
                `- **Total index size:** ${result.total_index_size}\n`
        );

        return {};
    }

    /* ── /config ────────────────────────────────────────────────────────── */

    private async handleConfig(
        stream: vscode.ChatResponseStream
    ): Promise<vscode.ChatResult> {
        const result = await this.bridge.sendRequest('get_config', {});

        stream.markdown(`## CaseCraft Configuration\n\n`);
        stream.markdown(
            '```json\n' + JSON.stringify(result, null, 2) + '\n```\n\n'
        );
        stream.markdown(
            '*Edit settings in VS Code Settings (`Ctrl+,`) → search "casecraft"*\n'
        );

        return {};
    }

    /* ── Freeform chat ──────────────────────────────────────────────────── */

    private async handleFreeform(
        request: vscode.ChatRequest,
        _context: vscode.ChatContext,
        stream: vscode.ChatResponseStream,
        token: vscode.CancellationToken
    ): Promise<vscode.ChatResult> {
        const query = request.prompt.trim();

        // Show help if empty
        if (!query) {
            stream.markdown(
                '## CaseCraft Commands\n\n' +
                    '| Command | Description |\n' +
                    '|---------|-------------|\n' +
                    '| `/generate` | Generate test cases from a feature document |\n' +
                    '| `/query` | Search the product knowledge base |\n' +
                    '| `/ingest` | Ingest documents or URLs into the KB |\n' +
                    '| `/config` | View current configuration |\n\n' +
                    'Or just type a question to chat about testing!\n'
            );
            return {};
        }

        // Retrieve KB context
        stream.progress('Searching knowledge base for context…');

        let kbContext = '';
        try {
            const kbResult = await this.bridge.sendRequest('query', {
                query,
                top_k: 3,
            });
            const chunks =
                (kbResult.chunks as Array<Record<string, unknown>>) ?? [];
            if (chunks.length > 0) {
                kbContext = chunks
                    .map(
                        (c, i) =>
                            `[Context ${i + 1}] ${c.text as string}`
                    )
                    .join('\n\n');
            }
        } catch {
            // Knowledge base not available — proceed without context
        }

        // Call Copilot model directly for the conversational answer
        stream.progress('Thinking…');

        const models = await vscode.lm.selectChatModels({
            vendor: 'copilot',
        });
        if (!models || models.length === 0) {
            stream.markdown(
                'No Copilot models available. ' +
                    'Please ensure you have a GitHub Copilot subscription.\n'
            );
            return {};
        }

        const systemPrompt =
            'You are CaseCraft, an expert QA assistant that helps users ' +
            'understand testing concepts, review test strategies, and answer ' +
            'questions about their product. Be concise and practical.' +
            (kbContext
                ? `\n\nRelevant product knowledge:\n${kbContext}`
                : '');

        const messages = [
            vscode.LanguageModelChatMessage.User(
                systemPrompt + '\n\nUser question: ' + query
            ),
        ];

        const response = await models[0].sendRequest(messages, {}, token);

        for await (const chunk of response.text) {
            stream.markdown(chunk);
        }

        return {};
    }

    dispose(): void {
        this.participant.dispose();
    }
}
