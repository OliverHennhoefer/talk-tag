# Input and Output

## Accepted input types

- `.cha`
- `.jsonl`

## `.jsonl` requirements

When annotating `.jsonl`, both fields are required:

- `--speaker-field`
- `--text-field`

## Input mode options

- `--input-dir`: annotate all supported files in a directory.
- `--input-path`: annotate one supported file.

## Output artifacts

- Annotated files in `--output-dir`.
- Structured run report at `_talk_tag_report.json` in `--output-dir`.

