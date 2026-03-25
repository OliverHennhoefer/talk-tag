# Contributing

Thanks for contributing to `talk-tag`.

## Development setup

```bash
uv sync --extra dev --extra docs --extra qa
```

If you do not use `uv`, use a virtual environment and install editable extras:

```bash
python -m pip install -e ".[dev,docs,qa]"
```

## Quality checks

Run the same checks used by CI:

```bash
python -m pytest -q
python -m ruff check src tests
python -m mypy src/talk_tag
python -m mkdocs build --strict
uv build
python -m twine check dist/*
```

## Pull requests

- Keep changes focused and minimal.
- Avoid changing runtime behavior unless explicitly scoped.
- Update docs/tests when interfaces or user-facing behavior change.
- Ensure all checks pass before requesting review.

## Release process

- Update `CHANGELOG.md`.
- Ensure the `pyproject.toml` version is correct.
- Create a GitHub release with tag `v<version>` or `<version>`.
- Publish workflow validates tag/version and pushes to PyPI via trusted publishing.

