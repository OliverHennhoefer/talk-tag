# talk-tag

**talk-tag** is a tool for automatic morphosyntactic error annotation in transcribed speech.

It adds inline CHAT-compatible error tags to utterances, helping researchers and annotators
pre-annotate transcripts for review. The current system follows a subset of the CHAT word-level
error coding scheme described in [Tools for Analyzing Talk, Part 1: The CHAT Transcription Format (Chapter 18.1)](https://doi.org/10.21415/3mhn-0z89).

## What It Annotates

TalkTag currently annotates:

- morphological errors: `[* m:*]`
- substitution errors (subset of semantic errors in the manual): `[* s:r:*]` and `[* s:r:gc:*]`

It also inserts target reconstructions inline, following CHAT conventions:

- `[: target]` when the produced form is a non-word
- `[:: target]` when the produced form is a real word but the intended target should still be recorded

For the current package behavior:

- non-word reconstructions such as `[: went]` are preserved
- real-word reconstructions are converted to `[= target]`
- `[= target]` output is hidden by default and included only when `--show-target` is set

This is intentional: according to the CHAT manual, `[= target]` is not required
for analysis in the way `[: target]` is, so TalkTag keeps it optional and
defaults to the cleaner output.

The CHAT manual distinguishes these because `[: target]` lets MOR use "the real
word target" for parsing, whereas the real-word replacement notation lets MOR
use "the actual word produced" while still preserving the target for other CLAN
analyses. See the [CHAT
manual](https://talkbank.org/0info/manuals/CHAT.html) and the [CLAN
manual](https://talkbank.org/0info/manuals/CLAN.html).

### Quick Examples

```text
Yesterday I walk [:: walked] [* m:0ed] to school .
Yesterday I goed [: went] [* m:=ed] to school . 
Yesterday me [:: I] [* s:r:gc:pro] walked to school .
Yesterday I went in [:: to] [* s:r:prep] school .
```

See the CHAT Transcription Guidelines.

## Annotation Scheme

### Morphological Labels
CHAT error tags are compositional: each part of a tag indicates, from general to fine-grained the error and its underline process. 
For example, in `[* m:0ed]`, `m` marks a morphosyntactic error, `0` marks a missing form, and `ed` marks past morpheme.

| Level 1     | Meaning                     |
|-------------|-----------------------------|
| `* m:`      | morphosyntactic error       |
| **Level 2** | **Meaning**                 |
| `0`         | missing regular form        |
| `=`         | over-regularisation         |
| `+`         | superfluous marking         |
| `++`        | double marking              |
| `base:`     | base for irregular form     |
| `irr:`      | irregular for base form     |
| `sub:`      | past/perfective substitution |
| `allo`      | allomorphic errors          |
| `vsg:`      | irregular verb 3SG          |
| `vun:`      | irregular verb unmarked     |
| **Level 3** | **Meaning**                 |
| `mor`       | target morpheme             |
| `a`         | agreement error             |
| `i`         | irregular target            |

Common level-3 morphemes include:

`ed`, `en`, `3s`, `ing`, `s`, `'s`, `er`, and `est`.

In practice, common outputs include:

- `[* m:0ed]` for missing past tense
- `[* m:=ed]` for over-regularised past forms
- `[* m:03s:a]` for missing 3SG agreement marking
- 
### Substitution Labels

| Level 1 | Meaning |
|---------| --- |
| `* s:`  | substitution error |
| **Level 2** | **Meaning** |
| `r:`    | related lexical substitution |
| `r:gc:` | related grammatical substitution |
| **Level 3** | **Meaning** |
| `POS`   | target part of speech |

Supported part-of-speech (`POS`) in the paper include:

`pro`(pronoun), `det` (determiner), and `prep` (preposition).

In practice, common outputs include:

- `[* s:r:gc:pro]` for pronoun substitutions: 
possessive for nominative: `her/his/their` for `she/he/they`)

- `[* s:r:prep]` for preposition substitutions: e.g., *he is married `with` (instead of `to`) Maria 

## Scope Notes

- The current runtime follows a narrow prototype scope and does not cover the full CHAT error inventory.
- The paper's model was developed on children's narrative data from the [ENNI corpus](https://talkbank.org/childes/access/Clinical-Eng/ENNI.html) under low-resource conditions.
- The most realistic use case is assisted annotation and review of plausible error candidates.

## Install

Python requirement: `>=3.10`.

```bash
pip install "talk-tag[runtime]"
```
Runtime extras include `torch`, `transformers`, and `peft`.

## Runtime support

The current fixed deployment is based on a `bnb-4bit` Hugging Face model. In
practice, this means:

- CUDA is the preferred accelerated runtime
- CPU is supported as a fallback
- Apple MPS is not supported for this deployment

## First-run workflow

1. Check environment:

```bash
talk-tag doctor
```

2. Pull/warm model assets:

```bash
talk-tag model pull --device auto
```

On Apple Silicon, `--device auto` will fall back to CPU instead of MPS.

3. Run annotation:

```bash
talk-tag annotate \
  --input-dir ./input \
  --output-dir ./output \
  --target-speaker "*CHI" \
  --batch-size 4 \
  --device auto
```

Single-file `.cha` example:

```bash
talk-tag annotate \
  --input-path ./input/sample.cha \
  --output-dir ./output \
  --target-speaker "*CHI" \
  --batch-size 4 \
  --device auto
```

Show optional real-word reconstructions in the output:

```bash
talk-tag annotate \
  --input-path ./input/sample.cha \
  --output-dir ./output \
  --target-speaker "*CHI" \
  --show-target \
  --device auto
```

`--show-target` only affects optional real-word reconstructions such as
`[= goes]`. Non-word reconstructions such as `[: went]`, which are needed for
analysis, are preserved either way.

If needed, you can also cap inference for quick local checks:

```bash
talk-tag annotate \
  --input-path ./input/sample.cha \
  --output-dir ./output \
  --target-speaker "*CHI" \
  --batch-size 2 \
  --limit 20 \
  --device auto
```

When `--limit` is greater than `0`, TalkTag still writes the output file. It
simply stops annotation after the first `N` target utterances and prints a
notice that the limit is active.

## Inference defaults

- `batch_size = 4`
- `max_new_tokens = 128`
- `max_seq_length = 512`
- `max_context_chars = 1200`
- `limit = 0` (`0` means no cap; use it as a debug/testing limit on target utterances)
- greedy decoding (`do_sample = false`)

The CLI currently exposes:

- `--batch-size` to tune inference throughput vs memory usage
- `--limit` to cap the number of target utterances processed in one run for testing/debugging; output files are still written

## Supported runtime inputs

- `.cha`
- `.jsonl` (requires `--speaker-field` and `--text-field`)

The `annotate` command accepts either:

- `--input-dir` for folder annotation
- `--input-path` for a single `.cha` or `.jsonl` file

Other previously supported formats (`.txt`, `.csv`, `.json`, `.xlsx`) are rejected in adapter-only deployment mode.

## Colab quickstart

See [`examples/colab_quickstart.ipynb`](examples/colab_quickstart.ipynb) for a minimal setup flow.
