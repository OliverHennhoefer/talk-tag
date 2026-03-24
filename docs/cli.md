# CLI Usage

## Command groups

- `talk-tag annotate`
- `talk-tag doctor`
- `talk-tag model pull`

## `annotate`

Annotate either a directory of supported files or a single file:

```bash
talk-tag annotate \
  --input-dir ./input \
  --output-dir ./output \
  --target-speaker "*CHI" \
  --device auto
```

Single-file mode:

```bash
talk-tag annotate \
  --input-path ./input/sample.cha \
  --output-dir ./output \
  --target-speaker "*CHI"
```

For `.jsonl`, pass field mappings:

```bash
talk-tag annotate \
  --input-path ./input/records.jsonl \
  --output-dir ./output \
  --target-speaker "*CHI" \
  --speaker-field speaker \
  --text-field utterance
```

## `doctor`

Run environment and model-access checks:

```bash
talk-tag doctor --device auto
```

JSON output:

```bash
talk-tag doctor --json
```

## `model pull`

Download/cache model assets and optionally verify loading:

```bash
talk-tag model pull --device auto
```

Without runtime load verification:

```bash
talk-tag model pull --no-verify-load
```

