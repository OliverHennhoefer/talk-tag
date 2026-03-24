# talk-tag

[![PyPI](https://img.shields.io/pypi/v/talk-tag.svg)](https://pypi.org/project/talk-tag/)
[![Python](https://img.shields.io/pypi/pyversions/talk-tag.svg)](https://pypi.org/project/talk-tag/)
[![License](https://img.shields.io/pypi/l/talk-tag.svg)](LICENSE)
[![CI](https://github.com/OliverHennhoefer/talk-tag/actions/workflows/ci.yml/badge.svg)](https://github.com/OliverHennhoefer/talk-tag/actions/workflows/ci.yml)
[![Docs](https://github.com/OliverHennhoefer/talk-tag/actions/workflows/docs.yml/badge.svg)](https://github.com/OliverHennhoefer/talk-tag/actions/workflows/docs.yml)

`talk-tag` is an adapter-only TalkBank CHAT morphosyntactic error annotator for
`.cha` and `.jsonl` inputs.

## Runtime model contract

The deployment path is fixed:

1. Base model: `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
2. Adapter: `mash-mash/talkbank-morphosyntax-annotator-final-recon_full_comp_preserve_final_seed3407`

No merged-model runtime path is used. The package bundles CHAT token augmentation
entries and injects them into the tokenizer before adapter loading so tokenizer
and checkpoint vocabulary stay aligned.

## Install

Python `>=3.10` is required.

```bash
pip install "talk-tag[runtime]"
```

## Quickstart

Set Hugging Face credentials (required for the fixed base + adapter repositories):

```bash
export HF_TOKEN=...
```

On PowerShell:

```powershell
$env:HF_TOKEN = "..."
```

Run preflight checks:

```bash
talk-tag doctor
```

Warm model assets:

```bash
talk-tag model pull --device auto
```

Annotate a folder:

```bash
talk-tag annotate \
  --input-dir ./input \
  --output-dir ./output \
  --target-speaker "*CHI" \
  --device auto
```

Annotate one file:

```bash
talk-tag annotate \
  --input-path ./input/sample.cha \
  --output-dir ./output \
  --target-speaker "*CHI" \
  --device auto
```

## CLI commands

- `talk-tag annotate`: annotate `.cha` or `.jsonl` data.
- `talk-tag doctor`: run runtime, dependency, and model-access checks.
- `talk-tag model pull`: pre-download model assets and optionally verify load.

`.jsonl` inputs require `--speaker-field` and `--text-field`.

## Inference defaults

- `batch_size = 4`
- `max_new_tokens = 128`
- `max_seq_length = 512`
- `max_context_chars = 1200`
- `limit = 0`
- Greedy decoding (`do_sample = false`)

## Documentation and support

- Documentation: <https://oliverhennhoefer.github.io/talk-tag/>
- Changelog: [CHANGELOG.md](CHANGELOG.md)
- Security policy: [SECURITY.md](SECURITY.md)
- Contributing guide: [CONTRIBUTING.md](CONTRIBUTING.md)

## Notebook example

See [`examples/colab_quickstart.ipynb`](examples/colab_quickstart.ipynb).
