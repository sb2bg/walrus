const vscode = require("vscode");
const fs = require("fs");
const path = require("path");

let client;

function activate(context) {
  let languageClient;
  try {
    languageClient = require("vscode-languageclient/node");
  } catch (err) {
    vscode.window.showErrorMessage(
      "Walrus extension is missing dependencies. Run `npm install` in vscode/walrus."
    );
    return;
  }

  const { LanguageClient, TransportKind } = languageClient;
  const config = vscode.workspace.getConfiguration("walrus");
  const configuredCommand = config.get("languageServer.path", "walrus-lsp");
  const command = resolveLanguageServerPath(configuredCommand, context);
  const args = config.get("languageServer.args", []);

  const serverOptions = {
    command,
    args,
    transport: TransportKind.stdio
  };

  const clientOptions = {
    documentSelector: [{ language: "walrus", scheme: "file" }],
    synchronize: {
      fileEvents: vscode.workspace.createFileSystemWatcher("**/*.walrus")
    }
  };

  client = new LanguageClient(
    "walrusLanguageServer",
    "Walrus Language Server",
    serverOptions,
    clientOptions
  );

  context.subscriptions.push(client.start());
}

function resolveLanguageServerPath(configuredCommand, context) {
  if (configuredCommand && configuredCommand !== "walrus-lsp") {
    return configuredCommand;
  }

  const repoRoot = path.resolve(context.extensionPath, "..", "..");
  const debugBin = path.join(repoRoot, "target", "debug", "walrus-lsp");
  const releaseBin = path.join(repoRoot, "target", "release", "walrus-lsp");

  if (fs.existsSync(debugBin)) {
    return debugBin;
  }

  if (fs.existsSync(releaseBin)) {
    return releaseBin;
  }

  return "walrus-lsp";
}

function deactivate() {
  if (!client) {
    return undefined;
  }
  return client.stop();
}

module.exports = {
  activate,
  deactivate
};
