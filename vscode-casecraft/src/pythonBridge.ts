/**
 * Python Bridge
 *
 * Manages a long-lived Python child process running `bridge_server.py`.
 * Communicates via line-delimited JSON over stdin/stdout.
 *
 * Key responsibilities:
 *  - Spawn / restart the Python process
 *  - Send method requests and collect results
 *  - Forward progress events
 *  - Proxy LLM requests to the VS Code Language Model API
 */

import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as path from 'path';
import * as crypto from 'crypto';
import { LmBridge } from './lmBridge';
import { CaseCraftConfig } from './config';

/* ── Protocol types ─────────────────────────────────────────────────────── */

interface BridgeMessage {
    id?: string;
    type?: string;
    method?: string;
    params?: Record<string, unknown>;
    data?: Record<string, unknown>;
}

interface PendingRequest {
    resolve: (data: Record<string, unknown>) => void;
    reject: (error: Error) => void;
    onProgress?: (step: string, detail: string) => void;
}

/* ── Constants ──────────────────────────────────────────────────────────── */

const REQUEST_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes
const MAX_BUFFER_SIZE = 50 * 1024 * 1024; // 50 MB safety limit
const SPAWN_RETRY_DELAY_MS = 2000;
const MAX_SPAWN_RETRIES = 3;

/* ── Bridge class ───────────────────────────────────────────────────────── */

export class PythonBridge implements vscode.Disposable {
    private process: cp.ChildProcess | null = null;
    private pendingRequests = new Map<string, PendingRequest>();
    private buffer = '';
    private ready = false;
    private readyPromise: Promise<void>;
    private readyResolve!: () => void;
    private lmBridge: LmBridge;
    private config: CaseCraftConfig;
    private disposed = false;
    private spawnRetries = 0;
    private outputChannel: vscode.OutputChannel;

    constructor(
        _context: vscode.ExtensionContext,
        config: CaseCraftConfig
    ) {
        this.config = config;
        this.lmBridge = new LmBridge();
        this.outputChannel = vscode.window.createOutputChannel('CaseCraft');
        this.readyPromise = new Promise((resolve) => {
            this.readyResolve = resolve;
        });
        this.spawn();
    }

    /* ── Process lifecycle ──────────────────────────────────────────────── */

    private spawn(): void {
        const pythonPath = this.config.getPythonPath();
        const projectRoot = this.config.getProjectRoot();
        const bridgePath = path.join(projectRoot, 'bridge_server.py');

        const envOverrides = this.config.toEnvOverrides();

        this.outputChannel.appendLine(
            `[CaseCraft] Spawning: ${pythonPath} ${bridgePath}`
        );
        this.outputChannel.appendLine(`[CaseCraft] CWD: ${projectRoot}`);

        try {
            this.process = cp.spawn(pythonPath, [bridgePath], {
                cwd: projectRoot,
                stdio: ['pipe', 'pipe', 'pipe'],
                env: { ...process.env, ...envOverrides },
            });
        } catch (err: unknown) {
            const msg = err instanceof Error ? err.message : String(err);
            this.outputChannel.appendLine(`[CaseCraft] Spawn failed: ${msg}`);
            vscode.window.showErrorMessage(
                `CaseCraft: Failed to start Python bridge. ` +
                `Check that "${pythonPath}" is a valid Python path. Error: ${msg}`
            );
            return;
        }

        /* stdout → parse JSON lines */
        this.process.stdout?.on('data', (data: Buffer) => {
            this.buffer += data.toString();

            // Guard against runaway memory
            if (this.buffer.length > MAX_BUFFER_SIZE) {
                this.outputChannel.appendLine(
                    '[CaseCraft] Buffer overflow — resetting. Possible protocol error.'
                );
                this.buffer = '';
                return;
            }

            let idx: number;
            while ((idx = this.buffer.indexOf('\n')) !== -1) {
                const line = this.buffer.substring(0, idx).trim();
                this.buffer = this.buffer.substring(idx + 1);
                if (line) {
                    this.handleMessage(line);
                }
            }
        });

        /* stderr → dedicated OutputChannel for debugging */
        this.process.stderr?.on('data', (data: Buffer) => {
            const text = data.toString().trim();
            if (text) {
                this.outputChannel.appendLine(`[Python] ${text}`);
            }
        });

        /* spawn error (e.g. ENOENT for missing python) */
        this.process.on('error', (err: Error) => {
            this.outputChannel.appendLine(
                `[CaseCraft] Process error: ${err.message}`
            );
            vscode.window.showErrorMessage(
                `CaseCraft: Python bridge error — ${err.message}. ` +
                'Check the "CaseCraft" output channel for details.'
            );
        });

        /* process exit → reject pending, optionally restart */
        this.process.on('exit', (code) => {
            this.outputChannel.appendLine(
                `[CaseCraft] Python bridge exited (code ${code})`
            );
            this.process = null;
            this.ready = false;

            for (const [, req] of this.pendingRequests) {
                req.reject(new Error('Python bridge process exited unexpectedly'));
            }
            this.pendingRequests.clear();

            // Auto-restart unless intentionally disposed
            if (!this.disposed) {
                this.spawnRetries++;
                if (this.spawnRetries > MAX_SPAWN_RETRIES) {
                    this.outputChannel.appendLine(
                        `[CaseCraft] Max retries (${MAX_SPAWN_RETRIES}) exceeded.`
                    );
                    vscode.window.showErrorMessage(
                        'CaseCraft: Python bridge failed after multiple attempts. ' +
                        'Check the "CaseCraft" output channel.',
                        'Show Output'
                    ).then((choice) => {
                        if (choice === 'Show Output') {
                            this.outputChannel.show();
                        }
                    });
                    return;
                }

                this.readyPromise = new Promise((resolve) => {
                    this.readyResolve = resolve;
                });
                setTimeout(() => this.spawn(), SPAWN_RETRY_DELAY_MS);
            }
        });
    }

    /* ── Message routing ────────────────────────────────────────────────── */

    private handleMessage(line: string): void {
        let msg: BridgeMessage;
        try {
            msg = JSON.parse(line);
        } catch {
            this.outputChannel.appendLine(
                `[CaseCraft] Malformed JSON: ${line.substring(0, 200)}`
            );
            return;
        }

        // Ready handshake
        if (msg.type === 'ready') {
            this.ready = true;
            this.spawnRetries = 0; // Reset on successful start
            this.readyResolve();
            this.outputChannel.appendLine('[CaseCraft] Python bridge is ready');
            return;
        }

        // LLM proxy request — Python needs an LLM completion from Copilot
        if (msg.type === 'llm_request') {
            this.handleLlmRequest(msg);
            return;
        }

        // Match to a pending request by id
        const id = msg.id;
        if (!id) {
            return;
        }

        const pending = this.pendingRequests.get(id);
        if (!pending) {
            return;
        }

        switch (msg.type) {
            case 'progress':
                pending.onProgress?.(
                    (msg.data?.step as string) ?? '',
                    (msg.data?.detail as string) ?? ''
                );
                break;

            case 'result':
                this.pendingRequests.delete(id);
                pending.resolve((msg.data as Record<string, unknown>) ?? {});
                break;

            case 'error':
                this.pendingRequests.delete(id);
                pending.reject(
                    new Error(
                        (msg.data?.message as string) ?? 'Unknown Python error'
                    )
                );
                break;
        }
    }

    /* ── LLM proxy ──────────────────────────────────────────────────────── */

    private async handleLlmRequest(msg: BridgeMessage): Promise<void> {
        const reqId = msg.id;
        const prompt = (msg.data?.prompt as string) ?? '';
        const model = (msg.data?.model as string) ?? '';
        const jsonMode = (msg.data?.json_mode as boolean) ?? false;

        // Guard: reject excessively large prompts (>1MB)
        if (prompt.length > 1_000_000) {
            this.outputChannel.appendLine(
                `[CaseCraft] Rejecting oversized LLM prompt (${prompt.length} chars)`
            );
            this.writeMessage({
                id: reqId,
                type: 'llm_response',
                data: { text: '', error: 'Prompt too large (>1MB)' },
            });
            return;
        }

        try {
            const text = await this.lmBridge.generate(prompt, model, jsonMode);
            this.writeMessage({
                id: reqId,
                type: 'llm_response',
                data: { text },
            });
        } catch (err: unknown) {
            const message =
                err instanceof Error ? err.message : String(err);
            this.outputChannel.appendLine(
                `[CaseCraft] LLM proxy error: ${message}`
            );
            this.writeMessage({
                id: reqId,
                type: 'llm_response',
                data: { text: '', error: message },
            });
        }
    }

    /* ── Low-level I/O ──────────────────────────────────────────────────── */

    private writeMessage(msg: BridgeMessage): void {
        if (!this.process?.stdin?.writable) {
            this.outputChannel.appendLine(
                '[CaseCraft] Cannot write — Python stdin not available'
            );
            throw new Error('Python bridge not available');
        }
        const line = JSON.stringify(msg) + '\n';
        this.process.stdin.write(line);
    }

    /* ── Public API ─────────────────────────────────────────────────────── */

    /**
     * Send a request to the Python bridge and wait for the result.
     *
     * @param method    RPC method name (generate, query, ingest, get_config)
     * @param params    Method parameters
     * @param onProgress  Callback for progress events
     * @returns The result payload from Python
     */
    async sendRequest(
        method: string,
        params: Record<string, unknown>,
        onProgress?: (step: string, detail: string) => void,
        token?: vscode.CancellationToken
    ): Promise<Record<string, unknown>> {
        await this.readyPromise;

        if (!this.process) {
            this.spawn();
            await this.readyPromise;
        }

        const id = crypto.randomUUID();

        return new Promise<Record<string, unknown>>((resolve, reject) => {
            this.pendingRequests.set(id, { resolve, reject, onProgress });

            this.writeMessage({ id, method, params });

            // Cancellation token support
            let disposable: vscode.Disposable | undefined;
            if (token) {
                disposable = token.onCancellationRequested(() => {
                    if (this.pendingRequests.has(id)) {
                        this.pendingRequests.delete(id);
                        // Tell Python to cancel the running operation
                        try {
                            this.writeMessage({
                                id: crypto.randomUUID(),
                                method: 'cancel',
                                params: { target_id: id },
                            });
                        } catch {
                            // bridge may already be dead
                        }
                        reject(new Error('Request was cancelled'));
                    }
                    disposable?.dispose();
                });
            }

            // Safety timeout
            setTimeout(() => {
                if (this.pendingRequests.has(id)) {
                    this.pendingRequests.delete(id);
                    disposable?.dispose();
                    reject(new Error('Request timed out after 30 minutes'));
                }
            }, REQUEST_TIMEOUT_MS);
        });
    }

    /** Return the LM bridge for direct use (e.g. freeform chat). */
    getLmBridge(): LmBridge {
        return this.lmBridge;
    }

    /** Show the CaseCraft output channel. */
    showOutput(): void {
        this.outputChannel.show();
    }

    dispose(): void {
        this.disposed = true;
        this.process?.kill();
        this.process = null;
        this.outputChannel.dispose();
    }
}
