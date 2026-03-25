from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from talk_tag.json_utils import loads
from talk_tag.model.hf import ADAPTER_REPO_ID, BASE_MODEL_REPO_ID
from talk_tag.runtime import Device, RuntimeSelection, select_fixed_deployment_device


@dataclass(slots=True)
class LoadedDeploymentModel:
    model: Any
    tokenizer: Any
    runtime: RuntimeSelection
    added_tokens: int


def load_chat_tokens() -> list[str]:
    payload = (
        resources.files("talk_tag.model")
        .joinpath("chat_tokens.json")
        .read_text(encoding="utf-8")
    )
    raw = loads(payload)
    if not isinstance(raw, list):
        raise ValueError("chat_tokens.json must contain a JSON list of token strings.")
    tokens: list[str] = []
    for item in raw:
        if item is None:
            continue
        value = str(item).strip()
        if value:
            tokens.append(value)
    return tokens


def load_deployment_model(
    *,
    device: Device,
    hf_token: str | None,
    hf_cache_dir: Path | None,
) -> LoadedDeploymentModel:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised without dependency
        raise RuntimeError(
            "torch is required for adapter-based inference."
        ) from exc

    runtime = select_fixed_deployment_device(requested=device, torch_module=torch)
    cache_dir_str = str(hf_cache_dir) if hf_cache_dir is not None else None

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised without dependency
        raise RuntimeError(
            "transformers, peft, and torch are required for adapter-based inference."
        ) from exc

    # 1) Load base model.
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_REPO_ID,
        token=hf_token,
        cache_dir=cache_dir_str,
        trust_remote_code=True,
    )
    # 2) Load tokenizer from the same base model.
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_REPO_ID,
        token=hf_token,
        cache_dir=cache_dir_str,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) Load chat_tokens.json.
    chat_tokens = load_chat_tokens()
    # 4) Add CHAT tokens to tokenizer.
    added = tokenizer.add_tokens(chat_tokens, special_tokens=False)
    # 5) Resize embeddings only when new tokens were added.
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))

    # 6) Load adapter repo on top of base model.
    model = PeftModel.from_pretrained(
        model,
        ADAPTER_REPO_ID,
        token=hf_token,
        cache_dir=cache_dir_str,
    )
    model.to(runtime.resolved)
    model.eval()

    return LoadedDeploymentModel(
        model=model,
        tokenizer=tokenizer,
        runtime=runtime,
        added_tokens=added,
    )
