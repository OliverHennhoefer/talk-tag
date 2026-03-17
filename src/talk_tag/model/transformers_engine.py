from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

from talk_tag.annotator.confidence import heuristic_confidence, line_confidence
from talk_tag.annotator.parser import AnnotationParseError, parse_annotation_payload
from talk_tag.annotator.prompt import build_annotation_prompt, resolve_active_error_tags
from talk_tag.annotator.renderer import render_chat_markers
from talk_tag.config import Granularity
from talk_tag.json_utils import loads
from talk_tag.models import Annotation, LineResult


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
        value = str(item).strip()
        if value:
            tokens.append(value)
    return tokens


class TransformersAnnotator:
    def __init__(
        self,
        *,
        model_dir: Path,
        chat_tokens: list[str],
        max_new_tokens: int = 512,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised without dependency
            raise RuntimeError(
                "transformers and torch are required for inference."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU is required for talk-tag inference. No CUDA device was detected."
            )

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
        )
        self._model.to("cuda")
        self._model.eval()

        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        existing_vocab = self._tokenizer.get_vocab()
        new_tokens = [token for token in chat_tokens if token not in existing_vocab]
        if new_tokens:
            added = self._tokenizer.add_tokens(new_tokens, special_tokens=False)
            if added > 0:
                self._model.resize_token_embeddings(len(self._tokenizer))

        self._max_new_tokens = max_new_tokens

    def annotate_line(
        self,
        text: str,
        *,
        granularity: Granularity,
        error_tags: list[str],
        show_target: bool,
    ) -> LineResult:
        active_tags = resolve_active_error_tags(
            granularity=granularity,
            error_tags=error_tags,
        )
        prompt = build_annotation_prompt(text=text, active_error_tags=active_tags)

        encoded = self._tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to("cuda")
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to("cuda")

        generation_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "max_new_tokens": self._max_new_tokens,
            "do_sample": False,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if attention_mask is not None:
            generation_kwargs["attention_mask"] = attention_mask

        with self._torch.no_grad():
            generated = self._model.generate(**generation_kwargs)

        prompt_len = input_ids.shape[1]
        raw_text = self._tokenizer.decode(
            generated[0][prompt_len:],
            skip_special_tokens=True,
        )

        try:
            parsed = parse_annotation_payload(raw_text)
        except AnnotationParseError:
            return LineResult(
                original_text=text,
                annotated_text=text,
                annotations=[],
                line_confidence=1.0,
                is_target_line=True,
                confidence_source="heuristic",
            )

        annotations = self._build_annotations(
            text=text,
            parsed_annotations=parsed.annotations,
        )
        annotated_text = render_chat_markers(
            text=text,
            annotations=annotations,
            show_target=show_target,
        )
        return LineResult(
            original_text=text,
            annotated_text=annotated_text,
            annotations=annotations,
            line_confidence=line_confidence(annotations),
            is_target_line=True,
            confidence_source="heuristic",
        )

    def _build_annotations(
        self,
        *,
        text: str,
        parsed_annotations: list[dict[str, Any]],
    ) -> list[Annotation]:
        annotations: list[Annotation] = []
        search_cursor = 0

        for item in parsed_annotations:
            source = str(item.get("source", "")).strip()
            target = str(item.get("target", "")).strip()
            if not source or not target:
                continue

            start, end = _coerce_span(item.get("start"), item.get("end"), text)
            if start is None or end is None:
                start, end = _locate_source_span(text, source, search_cursor)
            if start is None or end is None:
                continue

            search_cursor = end
            annotations.append(
                Annotation(
                    source=source,
                    target=target,
                    error_tag=str(item.get("error_tag", "unspecified")),
                    start=start,
                    end=end,
                    confidence=heuristic_confidence(source, target),
                    message=str(item.get("message", "")),
                )
            )
        return annotations


def _coerce_span(
    start_raw: Any,
    end_raw: Any,
    text: str,
) -> tuple[int | None, int | None]:
    if not isinstance(start_raw, int) or not isinstance(end_raw, int):
        return None, None
    if start_raw < 0 or end_raw < 0:
        return None, None
    if end_raw < start_raw:
        return None, None
    if end_raw > len(text):
        return None, None
    return start_raw, end_raw


def _locate_source_span(
    text: str,
    source: str,
    search_cursor: int,
) -> tuple[int | None, int | None]:
    idx = text.find(source, search_cursor)
    if idx < 0:
        idx = text.find(source)
    if idx < 0:
        return None, None
    return idx, idx + len(source)
