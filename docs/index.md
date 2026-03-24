# talk-tag

`talk-tag` is an adapter-only morphosyntactic error annotator for TalkBank CHAT.

## What it does

- Annotates supported transcript inputs with deployed CHAT-style tags.
- Operates with a fixed base-model + adapter runtime contract.
- Provides preflight validation (`doctor`) and model warmup (`model pull`).

## Supported input formats

- `.cha`
- `.jsonl` (with explicit speaker/text field mapping)

## Quick links

- Installation: [Installation](installation.md)
- Command usage: [CLI Usage](cli.md)
- Runtime/auth details: [Runtime and Auth](runtime.md)
- Python API docs: [API Reference](api.md)
- Project repository: <https://github.com/OliverHennhoefer/talk-tag>

