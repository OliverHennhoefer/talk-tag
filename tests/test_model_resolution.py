from __future__ import annotations

from pathlib import Path

import pytest

from talk_tag.api import annotate_path, pull_model
from talk_tag.model.hf import resolve_auth_token


def test_resolve_auth_token_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "from-env")
    token, mode = resolve_auth_token()
    assert token == "from-env"
    assert mode == "env-token"


def test_resolve_auth_token_returns_none_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    token, mode = resolve_auth_token()
    assert token is None
    assert mode == "none"


def test_pull_model_verify_false_uses_env_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "from-env")
    calls: list[tuple[str, str | None]] = []

    def fake_probe(
        *,
        repo_id: str,
        filename: str,
        token: str | None = None,
        cache_dir: Path | None = None,
    ) -> Path:
        del cache_dir
        calls.append((repo_id, token))
        return Path(f"{filename}").resolve()

    monkeypatch.setattr("talk_tag.api.probe_model_access", fake_probe)

    context = pull_model(verify_load=False)
    assert context.auth_mode == "env-token"
    assert len(calls) == 2
    assert all(token == "from-env" for _, token in calls)


def test_pull_model_rejects_removed_override_kwargs() -> None:
    with pytest.raises(TypeError):
        pull_model(hf_repo_id="repo")  # type: ignore[call-arg]


def test_annotate_path_rejects_removed_override_kwargs() -> None:
    with pytest.raises(TypeError):
        annotate_path(
            input_path="in",
            output_dir="out",
            target_speaker="*CHI",
            hf_token="secret",  # type: ignore[call-arg]
        )
