from __future__ import annotations

from pathlib import Path

from talk_tag.config import RunConfig
from talk_tag.formats.common import (
    AnnotationEngine,
    normalize_chat_reconstructions,
    passthrough_result,
)
from talk_tag.json_utils import dumps, loads
from talk_tag.models import FileResult
from talk_tag.progress import wrap_progress


def process_jsonl_file(
    input_path: Path,
    output_path: Path,
    config: RunConfig,
    engine: AnnotationEngine,
) -> FileResult:
    speaker_field, text_field = config.require_structured_fields(input_path)
    raw_lines = input_path.read_text(encoding="utf-8").splitlines()

    output_records: list[bytes] = []
    target_lines = 0
    annotated_lines = 0

    iterator = wrap_progress(
        raw_lines,
        enabled=config.show_progress,
        total=len(raw_lines),
        desc=f"{input_path.name}:records",
    )
    for line_number, raw_line in enumerate(iterator, start=1):
        if not raw_line.strip():
            raise ValueError(
                f"{input_path}:{line_number}: blank JSONL lines are invalid."
            )

        payload = loads(raw_line)
        if not isinstance(payload, dict):
            raise ValueError(
                f"{input_path}:{line_number}: each JSONL line must be a JSON object."
            )
        if speaker_field not in payload:
            raise ValueError(
                f"{input_path}:{line_number}: missing speaker field '{speaker_field}'."
            )
        if text_field not in payload:
            raise ValueError(
                f"{input_path}:{line_number}: missing text field '{text_field}'."
            )

        speaker = str(payload.get(speaker_field, ""))
        text = str(payload.get(text_field, ""))
        is_target = config.speaker_matches(speaker)
        if is_target and config.consume_target_utterance_slot():
            line_result = engine.annotate_line(
                text,
                granularity=config.granularity,
                error_tags=config.error_tags,
                show_target=config.show_target,
            )
            line_result.annotated_text = normalize_chat_reconstructions(
                line_result.annotated_text,
                show_target=config.show_target,
            )
            target_lines += 1
            if line_result.annotated_text != text:
                annotated_lines += 1
        else:
            line_result = passthrough_result(text, is_target_line=is_target)

        payload["tt_annotated_text"] = line_result.annotated_text
        payload["tt_annotations"] = [item.to_dict() for item in line_result.annotations]
        payload["tt_line_confidence"] = line_result.line_confidence
        payload["tt_is_target_line"] = is_target
        output_records.append(dumps(payload))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        for record in output_records:
            handle.write(record)
            handle.write(b"\n")

    return FileResult(
        input_path=str(input_path),
        output_path=str(output_path),
        status="ok",
        target_lines=target_lines,
        annotated_lines=annotated_lines,
    )
