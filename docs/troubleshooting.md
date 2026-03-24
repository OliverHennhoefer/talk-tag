# Troubleshooting

## `401` or unauthorized model errors

- Confirm `HF_TOKEN` is set in the shell running `talk-tag`.
- Confirm your token has access to both model repositories.

## `403` or gated repository errors

- Your account/token can authenticate but lacks repository permission.
- Request access to the model repositories and retry.

## Network/offline download failures

- Verify internet connectivity.
- Retry with a stable connection.
- Optionally pre-populate and reuse a Hugging Face cache directory.

## Unsupported input format errors

`talk-tag` supports only `.cha` and `.jsonl` in adapter-only mode.

## `.jsonl` field mapping errors

Provide both:

- `--speaker-field`
- `--text-field`

