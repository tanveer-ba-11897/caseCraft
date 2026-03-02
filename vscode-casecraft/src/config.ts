/**
 * CaseCraft Configuration Bridge
 *
 * Reads VS Code settings (casecraft.*) and maps them to the Python backend
 * config format.  Settings here override casecraft.yaml when the extension
 * is active.
 */

import * as vscode from 'vscode';

export class CaseCraftConfig {
    private get cfg(): vscode.WorkspaceConfiguration {
        return vscode.workspace.getConfiguration('casecraft');
    }

    /** Resolve the CaseCraft project root directory. */
    getProjectRoot(): string {
        const explicit = this.cfg.get<string>('projectRoot', '');
        if (explicit) {
            return explicit;
        }
        // Auto-detect: use the first workspace folder
        const folders = vscode.workspace.workspaceFolders;
        if (folders && folders.length > 0) {
            return folders[0].uri.fsPath;
        }
        return '.';
    }

    getPythonPath(): string {
        return this.cfg.get<string>('pythonPath', 'python');
    }

    getLlmProvider(): string {
        return this.cfg.get<string>('llmProvider', 'copilot');
    }

    getModel(): string {
        return this.cfg.get<string>('model', 'gpt-4o');
    }

    getAppType(): string {
        return this.cfg.get<string>('appType', 'web');
    }

    getSemanticDedup(): boolean {
        return this.cfg.get<boolean>('semanticDedup', true);
    }

    getReviewerPass(): boolean {
        return this.cfg.get<boolean>('reviewerPass', false);
    }

    getBaseUrl(): string {
        return this.cfg.get<string>('baseUrl', 'http://localhost:11434');
    }

    /**
     * Build environment variables that override casecraft.yaml settings.
     * The Python config system (core/config.py) reads CASECRAFT_SECTION_KEY
     * env vars automatically.
     */
    toEnvOverrides(): Record<string, string> {
        const provider = this.getLlmProvider();
        const env: Record<string, string> = {
            PYTHONUNBUFFERED: '1',
        };

        if (provider === 'copilot') {
            // Tell the Python side to use the vscode LLM proxy
            env['CASECRAFT_GENERAL_LLM_PROVIDER'] = 'vscode';
        } else {
            env['CASECRAFT_GENERAL_LLM_PROVIDER'] = provider;
            env['CASECRAFT_GENERAL_BASE_URL'] = this.getBaseUrl();
        }

        env['CASECRAFT_GENERAL_MODEL'] = this.getModel();
        env['CASECRAFT_GENERATION_APP_TYPE'] = this.getAppType();
        env['CASECRAFT_QUALITY_SEMANTIC_DEDUPLICATION'] = String(this.getSemanticDedup());
        env['CASECRAFT_QUALITY_REVIEWER_PASS'] = String(this.getReviewerPass());

        return env;
    }
}
