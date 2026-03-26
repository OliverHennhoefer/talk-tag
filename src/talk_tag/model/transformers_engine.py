from __future__ import annotations

from pathlib import Path

from talk_tag.config import Granularity
from talk_tag.inference import InferenceConfig, TalkTagInference
from talk_tag.models import LineResult
from talk_tag.runtime import Device, RuntimeSelection


class TransformersAnnotator:
    def __init__(
        self,
        *,
        device: Device = "auto",
        max_new_tokens: int = 128,
        hf_cache_dir: Path | None = None,
        max_seq_length: int = 512,
        max_context_chars: int = 1200,
        limit: int = 0,
    ) -> None:
        self._inference = TalkTagInference(
            device=device,
            hf_cache_dir=hf_cache_dir,
            config=InferenceConfig(
                max_new_tokens=max_new_tokens,
                max_seq_length=max_seq_length,
                max_context_chars=max_context_chars,
                limit=limit,
                do_sample=False,
            ),
        )

    @property
    def runtime(self) -> RuntimeSelection:
        return self._inference.runtime

    def annotate_line(
        self,
        text: str,
        *,
        granularity: Granularity,
        error_tags: list[str],
        show_target: bool,
    ) -> LineResult:
        # The deployed model handles tag generation directly from the utterance.
        del granularity, error_tags, show_target
        annotated = self._inference.annotate_utterance(text)
        changed = annotated != text
        return LineResult(
            original_text=text,
            annotated_text=annotated,
            annotations=[],
            line_confidence=0.95 if changed else 1.0,
            is_target_line=True,
            confidence_source="model-greedy",
        )
