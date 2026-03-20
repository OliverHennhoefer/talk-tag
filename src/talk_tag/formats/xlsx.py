from __future__ import annotations

from pathlib import Path

from talk_tag.config import RunConfig
from talk_tag.formats.common import passthrough_result
from talk_tag.json_utils import dumps
from talk_tag.models import FileResult
from talk_tag.progress import wrap_progress


def _serialize_annotations(annotations: list[object]) -> str:
    return dumps(annotations).decode("utf-8")


def process_xlsx_file(
    input_path: Path,
    output_path: Path,
    config: RunConfig,
    engine: object,
) -> FileResult:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised without dependency
        raise RuntimeError(
            "pandas and openpyxl are required for XLSX processing."
        ) from exc

    speaker_field, text_field = config.require_structured_fields(input_path)
    frame = pd.read_excel(input_path, dtype=str)
    frame = frame.fillna("")

    if speaker_field not in frame.columns:
        raise ValueError(f"{input_path}: missing speaker field '{speaker_field}'.")
    if text_field not in frame.columns:
        raise ValueError(f"{input_path}: missing text field '{text_field}'.")

    tt_annotated_text: list[str] = []
    tt_annotations: list[str] = []
    tt_line_confidence: list[float] = []
    tt_is_target_line: list[bool] = []

    target_lines = 0
    annotated_lines = 0
    row_indices = list(range(len(frame)))

    iterator = wrap_progress(
        row_indices,
        enabled=config.show_progress,
        total=len(row_indices),
        desc=f"{input_path.name}:rows",
    )
    for idx in iterator:
        speaker = str(frame.at[idx, speaker_field])
        text = str(frame.at[idx, text_field])
        is_target = config.speaker_matches(speaker)
        if is_target:
            line_result = engine.annotate_line(
                text,
                granularity=config.granularity,
                error_tags=config.error_tags,
                show_target=config.show_target,
            )
            target_lines += 1
            if line_result.annotations:
                annotated_lines += 1
        else:
            line_result = passthrough_result(text, is_target_line=False)

        tt_annotated_text.append(line_result.annotated_text)
        tt_annotations.append(
            _serialize_annotations([item.to_dict() for item in line_result.annotations])
        )
        tt_line_confidence.append(line_result.line_confidence)
        tt_is_target_line.append(is_target)

    frame["tt_annotated_text"] = tt_annotated_text
    frame["tt_annotations"] = tt_annotations
    frame["tt_line_confidence"] = tt_line_confidence
    frame["tt_is_target_line"] = tt_is_target_line

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_excel(output_path, index=False, engine="openpyxl")

    return FileResult(
        input_path=str(input_path),
        output_path=str(output_path),
        status="ok",
        target_lines=target_lines,
        annotated_lines=annotated_lines,
    )
