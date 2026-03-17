from __future__ import annotations

from pathlib import Path
from typing import Any

from talk_tag.config import RunConfig
from talk_tag.formats.common import passthrough_result
from talk_tag.json_utils import dumps, loads
from talk_tag.models import FileResult
from talk_tag.progress import wrap_progress


def _process_record(
    record: dict[str, Any],
    *,
    speaker_field: str,
    text_field: str,
    config: RunConfig,
    engine: object,
) -> tuple[dict[str, Any], bool, bool]:
    if speaker_field not in record:
        raise ValueError(f"Record missing speaker field '{speaker_field}'.")
    if text_field not in record:
        raise ValueError(f"Record missing text field '{text_field}'.")

    speaker = str(record.get(speaker_field, ""))
    text = str(record.get(text_field, ""))
    is_target = config.speaker_matches(speaker)
    if is_target:
        line_result = engine.annotate_line(
            text,
            granularity=config.granularity,
            error_tags=config.error_tags,
            show_target=config.show_target,
        )
    else:
        line_result = passthrough_result(text, is_target_line=False)

    updated = dict(record)
    updated["tt_annotated_text"] = line_result.annotated_text
    updated["tt_annotations"] = [item.to_dict() for item in line_result.annotations]
    updated["tt_line_confidence"] = line_result.line_confidence
    updated["tt_is_target_line"] = is_target

    return updated, is_target, bool(line_result.annotations)


def process_json_file(
    input_path: Path,
    output_path: Path,
    config: RunConfig,
    engine: object,
) -> FileResult:
    speaker_field, text_field = config.require_structured_fields(input_path)
    payload = loads(input_path.read_bytes())

    target_lines = 0
    annotated_lines = 0

    if isinstance(payload, dict):
        updated, is_target, was_annotated = _process_record(
            payload,
            speaker_field=speaker_field,
            text_field=text_field,
            config=config,
            engine=engine,
        )
        target_lines = 1 if is_target else 0
        annotated_lines = 1 if was_annotated else 0
        result_payload: Any = updated
    elif isinstance(payload, list):
        result_payload = []
        iterator = wrap_progress(
            payload,
            enabled=config.show_progress,
            total=len(payload),
            desc=f"{input_path.name}:records",
        )
        for item in iterator:
            if not isinstance(item, dict):
                raise ValueError(f"{input_path}: expected list of JSON objects.")
            updated, is_target, was_annotated = _process_record(
                item,
                speaker_field=speaker_field,
                text_field=text_field,
                config=config,
                engine=engine,
            )
            result_payload.append(updated)
            if is_target:
                target_lines += 1
            if was_annotated:
                annotated_lines += 1
    else:
        raise ValueError(
            f"{input_path}: JSON root must be an object or list of objects."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(dumps(result_payload, pretty=True))

    return FileResult(
        input_path=str(input_path),
        output_path=str(output_path),
        status="ok",
        target_lines=target_lines,
        annotated_lines=annotated_lines,
    )
