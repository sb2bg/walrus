# Walrus VS Code Extension

This extension provides:

- Syntax highlighting for `.walrus` files
- Language Server Protocol (LSP) support powered by `walrus-lsp`

## Setup

1. Build the language server from the repository root:

   ```bash
   cargo build --release --bin walrus-lsp
   ```

2. Install extension dependencies:

   ```bash
   cd vscode/walrus
   npm install
   ```

3. Launch the extension in VS Code:

   - Open `vscode/walrus` in VS Code.
   - Press `F5` to start an Extension Development Host.
   - Open a `.walrus` file.

## Settings

- `walrus.languageServer.path`: Executable path (default: `walrus-lsp`)
- `walrus.languageServer.args`: Extra args for the server

If `walrus-lsp` is not in your `PATH`, point `walrus.languageServer.path` to:

`/absolute/path/to/walrus/target/release/walrus-lsp`
