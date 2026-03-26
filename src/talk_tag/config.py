from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args

from talk_tag.runtime import Device

Granularity = Literal["light", "standard", "strict"]
TARGET_SPEAKER_RE = re.compile(r"^\*[A-Z0-9]{1,8}$")
VALID_GRANULARITIES: set[str] = {"light", "standard", "strict"}
VALID_DEVICES: frozenset[str] = frozenset(get_args(Device))


@dataclass(slots=True)
class RunConfig:
    input_path: Path
    output_dir: Path
    target_speaker: str
    investigator_speaker: str | None = None
    device: Device = "auto"
    hf_cache_dir: Path | None = None
    granularity: Granularity = "standard"
    error_tags: list[str] = field(default_factory=list)
    limit: int = 0
    show_target: bool = False
    print_debug_lines: bool = False
    speaker_field: str | None = None
    text_field: str | None = None
    case_insensitive_speaker: bool = False
    continue_on_error: bool = True
    show_progress: bool = True
    _remaining_limit: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        self.input_path = Path(self.input_path).resolve()
        self.output_dir = Path(self.output_dir).resolve()
        if self.hf_cache_dir is not None:
            self.hf_cache_dir = Path(self.hf_cache_dir).resolve()
        self.error_tags = [tag.strip() for tag in self.error_tags if tag.strip()]
        self._remaining_limit = self.limit

    def validate(self) -> None:
        self._validate_io_paths()
        self._validate_target_speaker()
        self._validate_investigator_speaker()
        self._validate_device()
        self._validate_granularity()
        self._validate_inference_controls()

    def _validate_io_paths(self) -> None:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {self.input_path}")
        if not (self.input_path.is_dir() or self.input_path.is_file()):
            raise ValueError(
                f"Input path must be a file or directory: {self.input_path}"
            )
        if self.input_path == self.output_dir:
            raise ValueError("input_path and output_dir must be different paths.")
        if self.input_path.is_dir():
            if self.output_dir.is_relative_to(self.input_path):
                raise ValueError("output_dir must not be nested under input_path.")
            if self.input_path.is_relative_to(self.output_dir):
                raise ValueError("output_dir must not be a parent of input_path.")
        else:
            if self.output_dir == self.input_path.parent:
                raise ValueError(
                    "output_dir must not be the parent directory of input_path."
                )

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

    def _validate_device(self) -> None:
        if self.device not in VALID_DEVICES:
            raise ValueError("device must be one of: auto, cuda, mps, cpu.")

    def _validate_inference_controls(self) -> None:
        if self.limit < 0:
            raise ValueError("limit must be >= 0.")

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

    def can_annotate_target_utterance(self) -> bool:
        return self.limit == 0 or self._remaining_limit > 0

    def consume_target_utterance_slot(self) -> bool:
        if not self.can_annotate_target_utterance():
            return False
        if self.limit > 0:
            self._remaining_limit -= 1
        return True
