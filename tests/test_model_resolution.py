from __future__ import annotations

import re
from pathlib import Path

import pytest

from talk_tag.api import annotate_folder, pull_model
from talk_tag.model.hf import resolve_auth_token


class StubEngine:
    def annotate_line(
        self,
        text: str,
        *,
        granularity: str,
        error_tags: list[str],
        show_target: bool,
    ):
        raise AssertionError("engine should not be used when config validation fails")


def test_resolve_auth_token_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "from-env")
    token, mode = resolve_auth_token(
        expert_model_token="from-expert",
        hf_token="from-hf",
    )
    assert token == "from-expert"
    assert mode == "explicit-token"

    token, mode = resolve_auth_token(
        expert_model_token=None,
        hf_token="from-hf",
    )
    assert token == "from-hf"
    assert mode == "explicit-token"

    token, mode = resolve_auth_token(
        expert_model_token=None,
        hf_token=None,
    )
    assert token == "from-env"
    assert mode == "env-token"


@pytest.mark.parametrize(
    "override_kwargs",
    [
        {"hf_repo_id": "repo"},
        {"hf_filename": "config.json"},
        {"hf_token": "secret"},
        {"expert_model_id": "repo"},
        {"expert_model_path": Path("fake-model")},
        {"expert_model_revision": "main"},
        {"expert_model_token": "secret"},
    ],
)
def test_pull_model_rejects_runtime_source_overrides(
    override_kwargs: dict[str, object],
) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Model source overrides are not supported in adapter-only deployment."
        ),
    ):
        pull_model(**override_kwargs)


@pytest.mark.parametrize(
    "override_kwargs",
    [
        {"hf_repo_id": "repo"},
        {"hf_filename": "config.json"},
        {"hf_token": "secret"},
        {"expert_model_id": "repo"},
        {"expert_model_path": Path("fake-model")},
        {"expert_model_revision": "main"},
        {"expert_model_token": "secret"},
    ],
)
def test_annotate_folder_rejects_runtime_source_overrides(
    case_root: Path,
    override_kwargs: dict[str, object],
) -> None:
    input_dir = case_root / "in"
    output_dir = case_root / "out"
    input_dir.mkdir(parents=True)
    (input_dir / "sample.cha").write_text("*CHI:\ttest\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Model source overrides are not supported in adapter-only deployment."
        ),
    ):
        annotate_folder(
            input_dir=input_dir,
            output_dir=output_dir,
            target_speaker="*CHI",
            engine=StubEngine(),
            **override_kwargs,
        )
