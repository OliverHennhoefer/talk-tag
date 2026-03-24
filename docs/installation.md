# Installation

## Requirements

- Python `>=3.10`
- Access to the two runtime Hugging Face repositories:
  - `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
  - `mash-mash/talkbank-morphosyntax-annotator-final-recon_full_comp_preserve_final_seed3407`

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

