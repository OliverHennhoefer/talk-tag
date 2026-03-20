# talk-tag

Adapter-only TalkBank CHAT morphosyntactic error annotator for `.cha` and `.jsonl`.

The runtime deployment path is fixed to:

1. Base model: `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
2. Adapter: `mash-mash/Llama_TalkTag_CHAT_error_annotator_adapter`

No merged-model runtime path is used.

## Install

Python requirement: `>=3.10`.

```bash
pip install "talk-tag[runtime]"
```

Runtime extras include `torch`, `transformers`, and `peft`.

## Hugging Face access

You need Hub access to both repositories above. Set a token before first run:

```bash
export HF_TOKEN=...
```

If token or access is missing, `talk-tag doctor`/`talk-tag model pull` will report
auth or gated-repo errors.

## First-run workflow

1. Check environment:

```bash
talk-tag doctor
```

2. Pull/warm model assets:

```bash
talk-tag model pull --device auto
```

3. Run annotation:

```bash
talk-tag annotate \
  --input-dir ./input \
  --output-dir ./output \
  --target-speaker "*CHI" \
  --device auto
```

## Inference defaults

- `batch_size = 4`
- `max_new_tokens = 128`
- `max_seq_length = 512`
- `max_context_chars = 1200`
- `limit = 0`
- greedy decoding (`do_sample = false`)

## Supported runtime inputs

- `.cha`
- `.jsonl` (requires `--speaker-field` and `--text-field`)

Other previously supported formats (`.txt`, `.csv`, `.json`, `.xlsx`) are rejected in adapter-only deployment mode.

## Colab quickstart

See [`examples/colab_quickstart.ipynb`](examples/colab_quickstart.ipynb) for a minimal setup flow.
