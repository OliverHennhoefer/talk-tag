from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from talk_tag.config import RunConfig
from talk_tag.formats.cha import process_cha_file
from talk_tag.formats.jsonl import process_jsonl_file
from talk_tag.models import FileResult, LineResult, RunSummary
from talk_tag.progress import wrap_progress
from talk_tag.reporting import build_summary, write_run_report


class AnnotationEngine(Protocol):
    def annotate_line(
        self,
        text: str,
        *,
        granularity: str,
        error_tags: list[str],
        show_target: bool,
    ) -> LineResult: ...


FormatHandler = Callable[[Path, Path, RunConfig, AnnotationEngine], FileResult]

HANDLERS: dict[str, FormatHandler] = {
    ".cha": process_cha_file,
    ".jsonl": process_jsonl_file,
}

LEGACY_UNSUPPORTED_SUFFIXES: frozenset[str] = frozenset(
    {".txt", ".csv", ".json", ".xlsx"}
)


def _discover_files(input_dir: Path) -> tuple[list[Path], list[Path]]:
    files: list[Path] = []
    unsupported: list[Path] = []
    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in HANDLERS:
            files.append(path)
        elif suffix in LEGACY_UNSUPPORTED_SUFFIXES:
            unsupported.append(path)
    return sorted(files), sorted(unsupported)


def run_pipeline(*, config: RunConfig, engine: AnnotationEngine) -> RunSummary:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()

    files, unsupported = _discover_files(config.input_dir)
    if unsupported:
        examples = ", ".join(
            str(path.relative_to(config.input_dir)) for path in unsupported[:3]
        )
        count_msg = f" ({len(unsupported)} total)" if len(unsupported) > 3 else ""
        raise ValueError(
            "Unsupported input format(s) detected. Only .cha and .jsonl are supported "
            f"in adapter-only deployment. Example path(s): {examples}{count_msg}"
        )
    file_results: list[FileResult] = []

    file_iter = wrap_progress(
        files,
        enabled=config.show_progress,
        total=len(files),
        desc="Files",
    )
    for input_path in file_iter:
        output_path = config.output_dir / input_path.relative_to(config.input_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        handler = HANDLERS[input_path.suffix.lower()]
        try:
            result = handler(input_path, output_path, config, engine)
        except Exception as exc:
            result = FileResult(
                input_path=str(input_path),
                output_path=str(output_path),
                status="failed",
                errors=[str(exc)],
            )
            file_results.append(result)
            if not config.continue_on_error:
                break
            continue

        file_results.append(result)

    ended_at = datetime.now(timezone.utc).isoformat()
    summary = build_summary(
        input_dir=config.input_dir,
        output_dir=config.output_dir,
        started_at=started_at,
        ended_at=ended_at,
        discovered_files=len(files),
        file_results=file_results,
    )
    report_path = write_run_report(summary, config.output_dir)
    summary.report_path = str(report_path)
    return summary
