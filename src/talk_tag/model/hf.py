from __future__ import annotations

import os
from pathlib import Path

DEFAULT_HF_REPO_ID = "talk-tag/talk-tag-transformer"
DEFAULT_HF_FILENAME = "config.json"


def resolve_model_reference(
    *,
    hf_repo_id: str | None,
    hf_filename: str | None,
) -> tuple[str, str]:
    repo_id = hf_repo_id or DEFAULT_HF_REPO_ID
    filename = hf_filename or DEFAULT_HF_FILENAME
    return repo_id, filename


def resolve_model_directory(
    *,
    hf_repo_id: str | None,
    hf_filename: str | None,
    hf_token: str | None,
    hf_cache_dir: Path | None,
) -> Path:
    repo_id, filename = resolve_model_reference(
        hf_repo_id=hf_repo_id,
        hf_filename=hf_filename,
    )
    token = hf_token or os.environ.get("HF_TOKEN")
    cache_dir_str = str(hf_cache_dir) if hf_cache_dir is not None else None

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised without dependency
        raise RuntimeError(
            "huggingface_hub is required for Hugging Face model loading."
        ) from exc

    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,
        cache_dir=cache_dir_str,
    )
    snapshot_dir = snapshot_download(
        repo_id=repo_id,
        token=token,
        cache_dir=cache_dir_str,
    )
    return Path(snapshot_dir).resolve()
