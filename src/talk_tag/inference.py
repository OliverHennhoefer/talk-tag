from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from talk_tag.model.deployment_loader import (
    LoadedDeploymentModel,
    load_deployment_model,
)
from talk_tag.model.hf import resolve_auth_token
from talk_tag.runtime import Device, RuntimeSelection

DEFAULT_INSTRUCTION = """You are a TalkBank CHAT annotator for morphosyntactic error coding.

Task:
Annotate the input utterance by inserting valid CHAT error tags inline.

Rules:
1. Preserve original token order, spelling, casing, punctuation, disfluencies, and CHAT symbols.
2. Do NOT rewrite, paraphrase, or correct the utterance.
3. Insert only error tags inline, following the error token.
4. If no target error is present, return the utterance unchanged.
5. Write the target form as [: target] when the incorrect morpheme yields a nonword, and as [:: target] when the error involves an attested produced word. For addition errors where the target is zero (no overt form), do not add a target-form marker.
6. Build each CHAT error tag compositionally from licensed scheme parts rather than relying on a memorized whole-label form.
7. Use m:* only for same-lexeme morphological contrasts and s:* only for substitutional contrasts.
8. Use :a only for agreement-sensitive labels that license it.
9. Use :i only where an irregular-sensitive label licenses it; do not overgenerate it.
10. Output only licensed CHAT tags; do not invent unattested or unsupported combinations.
11. Output exactly one annotated utterance line and nothing else."""


@dataclass(slots=True)
class InferenceConfig:
    batch_size: int = 4
    max_new_tokens: int = 128
    max_seq_length: int = 512
    max_context_chars: int = 1200
    limit: int = 0
    do_sample: bool = False
    instruction: str = DEFAULT_INSTRUCTION

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.max_seq_length < 1:
            raise ValueError("max_seq_length must be >= 1")
        if self.max_context_chars < 1:
            raise ValueError("max_context_chars must be >= 1")
        if self.limit < 0:
            raise ValueError("limit must be >= 0")
        if self.do_sample:
            raise ValueError(
                "Greedy decoding is required for deployed adapter inference (do_sample=False)."
            )


def build_deployment_prompt(*, instruction: str, input_text: str) -> str:
    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Input:\n"
        f"{input_text}\n\n"
        "### Response:\n"
    )


def _chunked(values: Sequence[str], size: int) -> list[list[str]]:
    chunks: list[list[str]] = []
    for idx in range(0, len(values), size):
        chunks.append(list(values[idx : idx + size]))
    return chunks


def _first_nonempty_line(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for line in lines:
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


class TalkTagInference:
    def __init__(
        self,
        *,
        device: Device = "auto",
        hf_token: str | None = None,
        hf_cache_dir: Path | None = None,
        expert_model_token: str | None = None,
        config: InferenceConfig | None = None,
        loaded_model: LoadedDeploymentModel | None = None,
    ) -> None:
        self.config = config or InferenceConfig()
        token, auth_mode = resolve_auth_token(
            expert_model_token=expert_model_token,
            hf_token=hf_token,
        )
        self.auth_mode = auth_mode
        self.cache_dir = hf_cache_dir

        bundle = loaded_model or load_deployment_model(
            device=device,
            hf_token=token,
            hf_cache_dir=hf_cache_dir,
        )
        self._model = bundle.model
        self._tokenizer = bundle.tokenizer
        self._runtime = bundle.runtime

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised without dependency
            raise RuntimeError("torch is required for inference.") from exc
        self._torch = torch

    @property
    def runtime(self) -> RuntimeSelection:
        return self._runtime

    def annotate_utterance(self, text: str) -> str:
        outputs = self.annotate_batch([text])
        return outputs[0] if outputs else text

    def annotate_batch(self, utterances: Sequence[str]) -> list[str]:
        items = [str(item) for item in utterances]
        if self.config.limit > 0:
            items = items[: self.config.limit]
        if not items:
            return []

        results: list[str] = []
        for chunk in _chunked(items, self.config.batch_size):
            prompts: list[str] = []
            for utterance in chunk:
                limited = utterance[: self.config.max_context_chars]
                prompts.append(
                    build_deployment_prompt(
                        instruction=self.config.instruction,
                        input_text=limited,
                    )
                )

            encoded = self._tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
            )

            input_ids = encoded["input_ids"].to(self._runtime.resolved)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self._runtime.resolved)

            generation_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "max_new_tokens": self.config.max_new_tokens,
                "do_sample": False,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
            }
            if attention_mask is not None:
                generation_kwargs["attention_mask"] = attention_mask

            with self._torch.no_grad():
                generated = self._model.generate(**generation_kwargs)

            if attention_mask is not None:
                prompt_lengths = attention_mask.sum(dim=1).tolist()
            else:
                prompt_lengths = [input_ids.shape[1]] * len(chunk)

            for idx, prompt_len in enumerate(prompt_lengths):
                continuation = generated[idx][int(prompt_len) :]
                raw = self._tokenizer.decode(continuation, skip_special_tokens=True)
                line = _first_nonempty_line(raw)
                results.append(line if line else chunk[idx])
        return results
