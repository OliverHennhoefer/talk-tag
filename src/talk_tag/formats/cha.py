from __future__ import annotations

from pathlib import Path

from talk_tag.config import RunConfig
from talk_tag.formats.common import (
    AnnotationEngine,
    print_debug_line,
    process_speaker_prefixed_line,
    validate_participants_header,
)
from talk_tag.models import FileResult
from talk_tag.progress import wrap_progress


def process_cha_file(
    input_path: Path,
    output_path: Path,
    config: RunConfig,
    engine: AnnotationEngine,
) -> FileResult:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        lines = handle.readlines()

    warnings = validate_participants_header(lines, config)
    output_lines: list[str] = []
    target_lines = 0
    annotated_lines = 0

    iterator = wrap_progress(
        lines,
        enabled=config.show_progress,
        total=len(lines),
        desc=f"{input_path.name}:lines",
    )
    for line_number, line in enumerate(iterator, start=1):
        if line.startswith("@"):
            output_lines.append(line)
            continue

        processed = process_speaker_prefixed_line(
            line,
            config=config,
            engine=engine,
        )
        output_lines.append(processed.output_line)
        if processed.is_target_line:
            target_lines += 1
        if processed.was_annotated:
            annotated_lines += 1
            if config.print_debug_lines and processed.line_result is not None:
                print_debug_line(
                    source_name=input_path.name,
                    item_label=f"line {line_number}",
                    original_text=processed.line_result.original_text,
                    annotated_text=processed.line_result.annotated_text,
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        handle.writelines(output_lines)

    return FileResult(
        input_path=str(input_path),
        output_path=str(output_path),
        status="ok",
        target_lines=target_lines,
        annotated_lines=annotated_lines,
        warnings=warnings,
    )
