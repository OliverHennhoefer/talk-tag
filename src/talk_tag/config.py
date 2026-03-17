from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Granularity = Literal["light", "standard", "strict"]
TARGET_SPEAKER_RE = re.compile(r"^\*[A-Z0-9]{1,8}$")
VALID_GRANULARITIES: set[str] = {"light", "standard", "strict"}


@dataclass(slots=True)
class RunConfig:
    input_dir: Path
    output_dir: Path
    target_speaker: str
    investigator_speaker: str | None = None
    hf_repo_id: str | None = None
    hf_filename: str | None = None
    hf_token: str | None = None
    hf_cache_dir: Path | None = None
    granularity: Granularity = "standard"
    error_tags: list[str] = field(default_factory=list)
    show_target: bool = False
    speaker_field: str | None = None
    text_field: str | None = None
    csv_line_field: str | None = None
    case_insensitive_speaker: bool = False
    continue_on_error: bool = True
    show_progress: bool = True

    def __post_init__(self) -> None:
        self.input_dir = Path(self.input_dir).resolve()
        self.output_dir = Path(self.output_dir).resolve()
        if self.hf_cache_dir is not None:
            self.hf_cache_dir = Path(self.hf_cache_dir).resolve()
        self.error_tags = [tag.strip() for tag in self.error_tags if tag.strip()]

    def validate(self, *, require_model_source: bool) -> None:
        self._validate_io_paths()
        self._validate_target_speaker()
        self._validate_investigator_speaker()
        self._validate_granularity()
        if require_model_source:
            self._validate_model_source()

    def _validate_io_paths(self) -> None:
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {self.input_dir}")
        if self.input_dir == self.output_dir:
            raise ValueError("input_dir and output_dir must be different directories.")
        if self.output_dir.is_relative_to(self.input_dir):
            raise ValueError("output_dir must not be nested under input_dir.")
        if self.input_dir.is_relative_to(self.output_dir):
            raise ValueError("output_dir must not be a parent of input_dir.")

    def _validate_target_speaker(self) -> None:
        if not TARGET_SPEAKER_RE.match(self.target_speaker):
            raise ValueError(
                "target_speaker must match '^\\*[A-Z0-9]{1,8}$' (example: '*CHI')."
            )

    def _validate_investigator_speaker(self) -> None:
        if self.investigator_speaker is None:
            return
        if not TARGET_SPEAKER_RE.match(self.investigator_speaker):
            raise ValueError(
                "investigator_speaker must match '^\\*[A-Z0-9]{1,8}$' (example: '*INV')."
            )

    def _validate_granularity(self) -> None:
        if self.granularity not in VALID_GRANULARITIES:
            raise ValueError("granularity must be one of: light, standard, strict.")

    def _validate_model_source(self) -> None:
        if (self.hf_repo_id and not self.hf_filename) or (
            self.hf_filename and not self.hf_repo_id
        ):
            raise ValueError(
                "hf_repo_id and hf_filename must be provided together when overriding defaults."
            )

    def speaker_matches(self, speaker_token: str) -> bool:
        if self.case_insensitive_speaker:
            return speaker_token.lower() == self.target_speaker.lower()
        return speaker_token == self.target_speaker

    def require_structured_fields(self, file_path: Path) -> tuple[str, str]:
        if not self.speaker_field or not self.text_field:
            raise ValueError(
                f"{file_path}: speaker_field and text_field are required for "
                f"{file_path.suffix.lower()} files."
            )
        return self.speaker_field, self.text_field
