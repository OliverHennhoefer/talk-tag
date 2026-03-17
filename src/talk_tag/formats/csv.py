from __future__ import annotations

from pathlib import Path

from talk_tag.config import RunConfig
from talk_tag.formats.common import process_speaker_prefixed_line, validate_participants_header
from talk_tag.json_utils import dumps
from talk_tag.models import FileResult
from talk_tag.progress import wrap_progress


def _serialize_annotations(annotations: list[object]) -> str:
    return dumps(annotations).decode("utf-8")


def process_csv_file(
    input_path: Path,
    output_path: Path,
    config: RunConfig,
    engine: object,
) -> FileResult:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised without dependency
        raise RuntimeError("pandas is required for CSV processing.") from exc

    frame = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    if frame.columns.empty:
        raise ValueError(f"{input_path}: CSV file has no columns.")

    line_field = config.csv_line_field or str(frame.columns[0])
    if line_field not in frame.columns:
        raise ValueError(f"{input_path}: missing csv line field '{line_field}'.")

    warnings = validate_participants_header(
        [str(value) for value in frame[line_field].tolist()],
        config,
    )

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
        line_text = str(frame.at[idx, line_field])
        processed = process_speaker_prefixed_line(
            line_text,
            config=config,
            engine=engine,
        )
        frame.at[idx, line_field] = processed.output_line
        if processed.is_target_line:
            target_lines += 1
        if processed.was_annotated:
            annotated_lines += 1

        line_result = processed.line_result
        if line_result is None:
            tt_annotated_text.append(line_text)
            tt_annotations.append("[]")
            tt_line_confidence.append(1.0)
        else:
            tt_annotated_text.append(line_result.annotated_text)
            tt_annotations.append(
                _serialize_annotations([item.to_dict() for item in line_result.annotations])
            )
            tt_line_confidence.append(line_result.line_confidence)
        tt_is_target_line.append(processed.is_target_line)

    frame["tt_annotated_text"] = tt_annotated_text
    frame["tt_annotations"] = tt_annotations
    frame["tt_line_confidence"] = tt_line_confidence
    frame["tt_is_target_line"] = tt_is_target_line

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    return FileResult(
        input_path=str(input_path),
        output_path=str(output_path),
        status="ok",
        target_lines=target_lines,
        annotated_lines=annotated_lines,
        warnings=warnings,
    )
