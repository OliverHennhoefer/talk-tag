from __future__ import annotations

from pathlib import Path
from typing import Protocol

from talk_tag.config import Granularity, RunConfig
from talk_tag.model.hf import resolve_model_directory
from talk_tag.model.transformers_engine import TransformersAnnotator, load_chat_tokens
from talk_tag.models import LineResult, RunSummary
from talk_tag.pipeline import run_pipeline


class AnnotationEngine(Protocol):
    def annotate_line(
        self,
        text: str,
        *,
        granularity: str,
        error_tags: list[str],
        show_target: bool,
    ) -> LineResult:
        ...


def _build_engine(config: RunConfig) -> AnnotationEngine:
    model_dir = resolve_model_directory(
        hf_repo_id=config.hf_repo_id,
        hf_filename=config.hf_filename,
        hf_token=config.hf_token,
        hf_cache_dir=config.hf_cache_dir,
    )
    return TransformersAnnotator(
        model_dir=model_dir,
        chat_tokens=load_chat_tokens(),
    )


def annotate_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    target_speaker: str,
    *,
    investigator_speaker: str | None = None,
    hf_repo_id: str | None = None,
    hf_filename: str | None = None,
    hf_token: str | None = None,
    hf_cache_dir: str | Path | None = None,
    granularity: Granularity = "standard",
    error_tags: list[str] | None = None,
    show_target: bool = False,
    speaker_field: str | None = None,
    text_field: str | None = None,
    csv_line_field: str | None = None,
    case_insensitive_speaker: bool = False,
    continue_on_error: bool = True,
    show_progress: bool = True,
    engine: AnnotationEngine | None = None,
) -> RunSummary:
    config = RunConfig(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        target_speaker=target_speaker,
        investigator_speaker=investigator_speaker,
        hf_repo_id=hf_repo_id,
        hf_filename=hf_filename,
        hf_token=hf_token,
        hf_cache_dir=Path(hf_cache_dir) if hf_cache_dir is not None else None,
        granularity=granularity,
        error_tags=error_tags or [],
        show_target=show_target,
        speaker_field=speaker_field,
        text_field=text_field,
        csv_line_field=csv_line_field,
        case_insensitive_speaker=case_insensitive_speaker,
        continue_on_error=continue_on_error,
        show_progress=show_progress,
    )
    config.validate(require_model_source=engine is None)

    active_engine = engine if engine is not None else _build_engine(config)
    return run_pipeline(config=config, engine=active_engine)
