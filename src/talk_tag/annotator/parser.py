from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from talk_tag.json_utils import loads


class AnnotationParseError(ValueError):
    pass


@dataclass(slots=True)
class ParsedOutput:
    corrected_text: str | None = None
    annotations: list[dict[str, Any]] = field(default_factory=list)


def _strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 2 and lines[-1].strip().startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return text


def _extract_json_blob(raw: str) -> str:
    stripped = _strip_code_fences(raw)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < 0 or end < start:
        raise AnnotationParseError("Model output did not contain a JSON object.")
    return stripped[start : end + 1]


def parse_annotation_payload(raw: str) -> ParsedOutput:
    blob = _extract_json_blob(raw)
    payload = loads(blob)
    if not isinstance(payload, dict):
        raise AnnotationParseError("Model output JSON root must be an object.")

    candidate_annotations = payload.get("annotations", [])
    if not isinstance(candidate_annotations, list):
        raise AnnotationParseError("'annotations' must be a list.")

    normalized: list[dict[str, Any]] = []
    for item in candidate_annotations:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        if not source or not target:
            continue
        normalized.append(
            {
                "source": source,
                "target": target,
                "error_tag": str(item.get("error_tag", "unspecified")).strip()
                or "unspecified",
                "start": item.get("start"),
                "end": item.get("end"),
                "message": str(item.get("message", "")).strip(),
            }
        )

    corrected_text = payload.get("corrected_text") or payload.get("annotated_text")
    if corrected_text is not None:
        corrected_text = str(corrected_text)
    return ParsedOutput(corrected_text=corrected_text, annotations=normalized)
