from __future__ import annotations

import os
from pathlib import Path

BASE_MODEL_REPO_ID = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
BASE_MODEL_FILENAME = "config.json"
ADAPTER_REPO_ID = "mash-mash/talkbank-morphosyntax-annotator-final-recon_full_comp_preserve_final_seed3407"
ADAPTER_FILENAME = "adapter_config.json"


def resolve_auth_token(
    *,
    expert_model_token: str | None,
    hf_token: str | None,
) -> tuple[str | None, str]:
    # Keep compatibility with legacy arguments while preferring explicit token values.
    if expert_model_token:
        return expert_model_token, "explicit-token"
    if hf_token:
        return hf_token, "explicit-token"
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token, "env-token"
    return None, "none"


def _format_hf_error(exc: Exception, *, repo_id: str) -> RuntimeError:
    message = str(exc)
    lowered = message.lower()
    if "401" in lowered or "unauthorized" in lowered:
        return RuntimeError(
            f"Access to '{repo_id}' requires authentication. "
            "Set HF_TOKEN for base+adapter access."
        )
    if "403" in lowered or "gated" in lowered or "forbidden" in lowered:
        return RuntimeError(
            f"Access to '{repo_id}' is forbidden or gated for this account/token."
        )
    if "repository not found" in lowered:
        return RuntimeError(
            f"Model repository '{repo_id}' was not found. "
            "Verify repository access on Hugging Face Hub."
        )
    if (
        "offline" in lowered
        or "network" in lowered
        or "connection" in lowered
        or "client has been closed" in lowered
        or "socket" in lowered
    ):
        return RuntimeError(
            "Model download failed due to network/offline conditions. "
            "Retry with network access or pre-populate the HF cache."
        )
    return RuntimeError(f"Unable to download model '{repo_id}': {message}")


def probe_model_access(
    *,
    repo_id: str,
    filename: str,
    token: str | None = None,
    cache_dir: Path | None = None,
) -> Path:
    cache_dir_str = str(cache_dir) if cache_dir is not None else None
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - exercised without dependency
        raise RuntimeError(
            "huggingface_hub is required for Hugging Face model loading."
        ) from exc

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
            cache_dir=cache_dir_str,
        )
    except Exception as exc:  # pragma: no cover - depends on runtime/network
        raise _format_hf_error(exc, repo_id=repo_id) from exc
    return Path(path).resolve()
