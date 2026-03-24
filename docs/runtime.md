# Runtime and Auth

## Fixed runtime path

`talk-tag` uses a fixed deployment pair:

- Base model: `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
- Adapter: `mash-mash/Llama_TalkTag_CHAT_error_annotator_adapter`

No merged-model runtime path is used.

## Authentication

Set `HF_TOKEN` before first model operation (`doctor`, `model pull`, or `annotate`)
to avoid unauthorized or gated-repository errors.

## Device selection

Supported device requests:

- `auto`
- `cuda`
- `mps`
- `cpu`

## Inference defaults

- `batch_size = 4`
- `max_new_tokens = 128`
- `max_seq_length = 512`
- `max_context_chars = 1200`
- `limit = 0`
- Greedy decoding only (`do_sample = false`)
