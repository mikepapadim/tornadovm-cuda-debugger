// Starts `tcd dap` as the debug adapter for "tornadovm-cuda" launch configurations.
const vscode = require("vscode");

function activate(context) {
  context.subscriptions.push(vscode.debug.registerDebugAdapterDescriptorFactory("tornadovm-cuda", {
    createDebugAdapterDescriptor(session) {
      const tcd = vscode.workspace.getConfiguration("tornadovmCuda").get("tcdPath") || "tcd";
      const cwd = session.workspaceFolder ? session.workspaceFolder.uri.fsPath : undefined;
      return new vscode.DebugAdapterExecutable(tcd, ["dap"], { cwd });
    },
  }));
}

function deactivate() {}

module.exports = { activate, deactivate };
