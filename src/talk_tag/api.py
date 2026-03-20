from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol

from talk_tag.config import (
    Granularity,
    RunConfig,
    validate_fixed_deployment_model_source,
)
from talk_tag.model.hf import (
    ADAPTER_FILENAME,
    ADAPTER_REPO_ID,
    BASE_MODEL_FILENAME,
    BASE_MODEL_REPO_ID,
    probe_model_access,
    resolve_auth_token,
)
from talk_tag.model.transformers_engine import TransformersAnnotator
from talk_tag.models import LineResult, RunSummary
from talk_tag.pipeline import run_pipeline
from talk_tag.runtime import Device


class AnnotationEngine(Protocol):
    def annotate_line(
        self,
        text: str,
        *,
        granularity: str,
        error_tags: list[str],
        show_target: bool,
    ) -> LineResult: ...


@dataclass(slots=True)
class StartupContext:
    """Startup metadata emitted by model pull and annotation startup callbacks.

    Backend semantics:
    - ``cuda`` / ``mps`` / ``cpu``: backend was verified by loading a
      ``TransformersAnnotator``.
    - ``external``: backend was not verified by talk-tag. This is used for both:
      - caller-managed engines (``annotate_folder(..., engine=...)``), and
      - unverified pulls (``pull_model(..., verify_load=False)``).

    Use ``model_source`` to disambiguate these two cases:
    - ``external_engine`` => caller supplied the engine.
    - any repository/local model source => backend is unknown/unverified.
    """

    backend: Literal["cuda", "mps", "cpu", "external"]
    model_source: str
    cache_dir: str | None
    auth_mode: str
    warning: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "model_source": self.model_source,
            "cache_dir": self.cache_dir,
            "auth_mode": self.auth_mode,
            "warning": self.warning,
        }


def _build_engine(config: RunConfig) -> tuple[AnnotationEngine, StartupContext]:
    token, auth_mode = resolve_auth_token(
        expert_model_token=config.expert_model_token,
        hf_token=config.hf_token,
    )
    engine = TransformersAnnotator(
        device=config.device,
        hf_token=token,
        hf_cache_dir=config.hf_cache_dir,
    )
    context = StartupContext(
        backend=engine.runtime.resolved,
        model_source="fixed_base_adapter",
        cache_dir=str(config.hf_cache_dir) if config.hf_cache_dir is not None else None,
        auth_mode=auth_mode,
        warning=engine.runtime.warning,
    )
    return engine, context


def pull_model(
    *,
    hf_repo_id: str | None = None,
    hf_filename: str | None = None,
    hf_token: str | None = None,
    hf_cache_dir: str | Path | None = None,
    expert_model_id: str | None = None,
    expert_model_path: str | Path | None = None,
    expert_model_revision: str | None = None,
    expert_model_token: str | None = None,
    device: Device = "auto",
    verify_load: bool = True,
) -> StartupContext:
    """Download/resolve model assets and optionally verify model loading.

    When ``verify_load=False``, the returned context uses ``backend="external"``
    to indicate an unverified/unknown runtime backend (not a caller-supplied
    engine). In that case, ``model_source`` still reflects the resolved source.
    """

    validate_fixed_deployment_model_source(
        hf_repo_id=hf_repo_id,
        hf_filename=hf_filename,
        expert_model_id=expert_model_id,
        expert_model_path=Path(expert_model_path)
        if expert_model_path is not None
        else None,
        expert_model_revision=expert_model_revision,
        expert_model_token=expert_model_token,
        hf_token=hf_token,
    )
    active_cache = Path(hf_cache_dir).resolve() if hf_cache_dir is not None else None
    token, auth_mode = resolve_auth_token(
        expert_model_token=expert_model_token,
        hf_token=hf_token,
    )

    warning: str | None = None
    backend: Literal["cuda", "mps", "cpu", "external"] = "external"
    if verify_load:
        engine = TransformersAnnotator(
            device=device,
            hf_token=token,
            hf_cache_dir=active_cache,
        )
        backend = engine.runtime.resolved
        warning = engine.runtime.warning
    else:
        probe_model_access(
            repo_id=BASE_MODEL_REPO_ID,
            filename=BASE_MODEL_FILENAME,
            token=token,
            cache_dir=active_cache,
        )
        probe_model_access(
            repo_id=ADAPTER_REPO_ID,
            filename=ADAPTER_FILENAME,
            token=token,
            cache_dir=active_cache,
        )

    return StartupContext(
        backend=backend,
        model_source="fixed_base_adapter",
        cache_dir=str(active_cache) if active_cache is not None else None,
        auth_mode=auth_mode,
        warning=warning,
    )


def annotate_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    target_speaker: str,
    *,
    investigator_speaker: str | None = None,
    device: Device = "auto",
    hf_repo_id: str | None = None,
    hf_filename: str | None = None,
    hf_token: str | None = None,
    hf_cache_dir: str | Path | None = None,
    expert_model_id: str | None = None,
    expert_model_path: str | Path | None = None,
    expert_model_revision: str | None = None,
    expert_model_token: str | None = None,
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
    startup_callback: Callable[[StartupContext], None] | None = None,
) -> RunSummary:
    """Run folder annotation using either an internal or caller-supplied engine.

    If ``engine`` is provided, startup metadata reports ``backend="external"`` and
    ``model_source="external_engine"`` because talk-tag did not resolve/load the
    model runtime itself.
    """

    config = RunConfig(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        target_speaker=target_speaker,
        investigator_speaker=investigator_speaker,
        device=device,
        hf_repo_id=hf_repo_id,
        hf_filename=hf_filename,
        hf_token=hf_token,
        hf_cache_dir=Path(hf_cache_dir) if hf_cache_dir is not None else None,
        expert_model_id=expert_model_id,
        expert_model_path=Path(expert_model_path)
        if expert_model_path is not None
        else None,
        expert_model_revision=expert_model_revision,
        expert_model_token=expert_model_token,
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
    config.validate(require_model_source=True)

    if engine is None:
        active_engine, context = _build_engine(config)
        if startup_callback is not None:
            startup_callback(context)
    else:
        active_engine = engine
        if startup_callback is not None:
            startup_callback(
                StartupContext(
                    backend="external",
                    model_source="external_engine",
                    cache_dir=None,
                    auth_mode="external",
                    warning=None,
                )
            )

    return run_pipeline(config=config, engine=active_engine)
