from __future__ import annotations

from dataclasses import asdict, dataclass, field


def _clamp_01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(slots=True)
class Annotation:
    source: str
    target: str
    error_tag: str
    start: int
    end: int
    confidence: float
    message: str = ""

    def __post_init__(self) -> None:
        self.start = max(0, int(self.start))
        self.end = max(self.start, int(self.end))
        self.confidence = _clamp_01(float(self.confidence))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class LineResult:
    original_text: str
    annotated_text: str
    annotations: list[Annotation] = field(default_factory=list)
    line_confidence: float = 1.0
    is_target_line: bool = True
    confidence_source: str = "heuristic"

    def __post_init__(self) -> None:
        self.line_confidence = _clamp_01(float(self.line_confidence))

    def to_dict(self) -> dict[str, object]:
        return {
            "original_text": self.original_text,
            "annotated_text": self.annotated_text,
            "annotations": [item.to_dict() for item in self.annotations],
            "line_confidence": self.line_confidence,
            "is_target_line": self.is_target_line,
            "confidence_source": self.confidence_source,
        }


@dataclass(slots=True)
class FileResult:
    input_path: str
    output_path: str
    status: str
    target_lines: int = 0
    annotated_lines: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class RunSummary:
    input_dir: str
    output_dir: str
    started_at: str
    ended_at: str
    total_files: int
    processed_files: int
    failed_files: int
    target_lines: int
    annotated_lines: int
    report_path: str
    files: list[FileResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "target_lines": self.target_lines,
            "annotated_lines": self.annotated_lines,
            "report_path": self.report_path,
            "files": [item.to_dict() for item in self.files],
        }
