# Installation

## Requirements

- Python `>=3.10`
- A platform supported by `bitsandbytes>=0.46.1`
- Access to the two runtime Hugging Face repositories:
  - `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
  - `mash-mash/Llama_TalkTag_CHAT_error_annotator_adapter`

## Runtime install

```bash
pip install "talk-tag[runtime]"
```

## Set authentication token

```bash
export HF_TOKEN=...
```

On PowerShell:

```powershell
$env:HF_TOKEN = "..."
```

## Verify installation

```bash
talk-tag doctor
```

## Try the bundled sample

From a repository checkout:

```bash
talk-tag annotate \
  --input-path ./examples/sample.cha \
  --output-dir ./examples/sample_out \
  --target-speaker "*CHI" \
  --device auto \
  --limit 2 \
  --show-target
```
