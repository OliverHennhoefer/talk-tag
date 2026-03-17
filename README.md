# talk-tag

GPU-only Hugging Face transcript annotator for speaker-scoped CHAT correction tasks.

## CLI

```bash
talk-tag annotate \
  --input-dir ./input \
  --output-dir ./output \
  --target-speaker "*CHI" \
  --investigator-speaker "*INV"
```

Optional Hugging Face overrides:

```bash
talk-tag annotate \
  --input-dir ./input \
  --output-dir ./output \
  --target-speaker "*CHI" \
  --hf-repo-id your-org/your-model \
  --hf-filename config.json \
  --hf-token "$HF_TOKEN" \
  --hf-cache-dir ./hf-cache
```

## Notes

- Input files are never modified in-place. Outputs are mirrored into `--output-dir`.
- Supported inputs: `.cha`, `.txt`, `.csv`, `.json`, `.jsonl`, `.xlsx`.
- CHAT-like files (`.cha`, `.txt`, `.csv`) annotate only the selected target speaker.
