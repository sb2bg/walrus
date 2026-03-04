# Git Hooks

Enable repo-managed hooks with:

```bash
git config core.hooksPath .githooks
```

Installed hooks:

- `pre-commit`: runs `cargo fmt --all` and re-stages `.rs` files.
