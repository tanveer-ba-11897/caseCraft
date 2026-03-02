/**
 * CaseCraft VS Code Extension — Entry Point
 *
 * Registers the @casecraft chat participant and wires up the Python bridge.
 */

import * as vscode from 'vscode';
import * as cp from 'child_process';
import { CaseCraftParticipant } from './participant';
import { PythonBridge } from './pythonBridge';
import { CaseCraftConfig } from './config';

let bridge: PythonBridge | undefined;

export function activate(context: vscode.ExtensionContext): void {
    const config = new CaseCraftConfig();

    // Verify Python is accessible before spawning the bridge
    const pythonPath = config.getPythonPath();
    try {
        const check = cp.spawnSync(pythonPath, ['--version'], {
            timeout: 10_000,
            encoding: 'utf-8',
        });
        if (check.error) {
            vscode.window.showErrorMessage(
                `CaseCraft: Python not found at "${pythonPath}". ` +
                'Please set "casecraft.pythonPath" in Settings.',
                'Open Settings'
            ).then((choice) => {
                if (choice === 'Open Settings') {
                    vscode.commands.executeCommand(
                        'workbench.action.openSettings',
                        'casecraft.pythonPath'
                    );
                }
            });
            return;
        }
    } catch {
        // spawnSync itself can throw — proceed anyway, PythonBridge
        // will surface more specific errors on failure.
    }

    bridge = new PythonBridge(context, config);
    const participant = new CaseCraftParticipant(context, bridge, config);

    context.subscriptions.push(bridge);
    context.subscriptions.push(participant);

    // Register a command to show the output channel
    context.subscriptions.push(
        vscode.commands.registerCommand('casecraft.showOutput', () => {
            bridge?.showOutput();
        })
    );

    // Notify user on first activation
    const hasShownWelcome = context.globalState.get<boolean>('casecraft.welcomed');
    if (!hasShownWelcome) {
        vscode.window
            .showInformationMessage(
                'CaseCraft extension activated! Type @casecraft in Copilot Chat to get started.',
                'Open Chat'
            )
            .then((choice) => {
                if (choice === 'Open Chat') {
                    vscode.commands.executeCommand('workbench.action.chat.open');
                }
            });
        context.globalState.update('casecraft.welcomed', true);
    }
}

export function deactivate(): void {
    bridge?.dispose();
}
