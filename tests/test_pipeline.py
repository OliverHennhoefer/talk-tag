from __future__ import annotations

from pathlib import Path

import pytest

from talk_tag.api import annotate_folder, annotate_path
from talk_tag.config import RunConfig
from talk_tag.formats.common import normalize_chat_punctuation
from talk_tag.model.deployment_loader import load_chat_tokens
from talk_tag.models import Annotation, LineResult


class StubEngine:
    def annotate_line(
        self,
        text: str,
        *,
        granularity: str,
        error_tags: list[str],
        show_target: bool,
    ) -> LineResult:
        if "bad" not in text:
            return LineResult(
                original_text=text,
                annotated_text=text,
                annotations=[],
                line_confidence=1.0,
                is_target_line=True,
                confidence_source="heuristic",
            )

        start = text.index("bad")
        end = start + len("bad")
        annotation = Annotation(
            source="bad",
            target="good",
            error_tag="lexical",
            start=start,
            end=end,
            confidence=0.83,
            message="replace bad with good",
        )
        marker_target = "good" if show_target else ""
        annotated_text = f"{text[:start]}bad [:: {marker_target}]{text[end:]}"
        return LineResult(
            original_text=text,
            annotated_text=annotated_text,
            annotations=[annotation],
            line_confidence=0.83,
            is_target_line=True,
            confidence_source="heuristic",
        )


def test_target_speaker_accepts_flexible_code(case_root: Path) -> None:
    input_dir = case_root / "in"
    output_dir = case_root / "out"
    input_dir.mkdir()

    config = RunConfig(
        input_path=input_dir,
        output_dir=output_dir,
        target_speaker="*SUBJ1",
    )
    config.validate(require_model_source=False)


def test_target_speaker_rejects_invalid_code(case_root: Path) -> None:
    input_dir = case_root / "in"
    output_dir = case_root / "out"
    input_dir.mkdir()

    config = RunConfig(
        input_path=input_dir,
        output_dir=output_dir,
        target_speaker="*bad",
    )
    with pytest.raises(ValueError):
        config.validate(require_model_source=False)


def test_output_dir_overlap_is_rejected(case_root: Path) -> None:
    input_dir = case_root / "in"
    output_dir = input_dir / "nested"
    input_dir.mkdir(parents=True)
    config = RunConfig(
        input_path=input_dir,
        output_dir=output_dir,
        target_speaker="*CHI",
    )
    with pytest.raises(ValueError):
        config.validate(require_model_source=False)


def test_cha_header_warning_for_missing_investigator(case_root: Path) -> None:
    input_dir = case_root / "in"
    output_dir = case_root / "out"
    input_dir.mkdir()

    source = "@Begin\n@Participants:\tCHI Child, MOT Mother\n*CHI:\tbad token.\n@End\n"
    (input_dir / "sample.cha").write_text(source, encoding="utf-8", newline="")

    annotate_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        target_speaker="*CHI",
        investigator_speaker="*INV",
        show_target=True,
        engine=StubEngine(),
    )

    report = (output_dir / "_talk_tag_report.json").read_text(encoding="utf-8")
    assert "investigator speaker '*INV'" in report


def test_only_target_speaker_is_annotated_in_cha(case_root: Path) -> None:
    input_dir = case_root / "in"
    output_dir = case_root / "out"
    input_dir.mkdir()
    (input_dir / "sample.cha").write_text(
        "*CHI:\tbad one\n*INV:\tbad two\n",
        encoding="utf-8",
    )

    annotate_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        target_speaker="*CHI",
        show_target=True,
        engine=StubEngine(),
    )
    output_lines = (output_dir / "sample.cha").read_text(encoding="utf-8").splitlines()
    assert output_lines[0] == "*CHI:\tbad [:: good] one"
    assert output_lines[1] == "*INV:\tbad two"


def test_single_cha_file_is_supported(case_root: Path) -> None:
    input_file = case_root / "sample.cha"
    output_dir = case_root / "out"
    input_file.write_text(
        "*CHI:\tbad one\n*INV:\tbad two\n",
        encoding="utf-8",
    )

    annotate_path(
        input_path=input_file,
        output_dir=output_dir,
        target_speaker="*CHI",
        show_target=True,
        engine=StubEngine(),
    )
    output_lines = (output_dir / "sample.cha").read_text(encoding="utf-8").splitlines()
    assert output_lines[0] == "*CHI:\tbad [:: good] one"
    assert output_lines[1] == "*INV:\tbad two"


def test_chat_punctuation_spacing_uses_chat_rules() -> None:
    assert normalize_chat_punctuation("I want that,please.") == "I want that , please ."
    assert (
        normalize_chat_punctuation("he bad [:: good],too? now:yes!")
        == "he bad [:: good] , too ? now : yes !"
    )


def test_cha_output_preserves_chat_tags_while_spacing_punctuation(
    case_root: Path,
) -> None:
    class PunctuationEngine:
        def annotate_line(
            self,
            text: str,
            *,
            granularity: str,
            error_tags: list[str],
            show_target: bool,
        ) -> LineResult:
            del text, granularity, error_tags, show_target
            return LineResult(
                original_text="bad text",
                annotated_text="bad [:: good],too?",
                annotations=[],
                line_confidence=0.95,
                is_target_line=True,
                confidence_source="heuristic",
            )

    input_file = case_root / "sample.cha"
    output_dir = case_root / "out"
    input_file.write_text("*CHI:\tbad text\n", encoding="utf-8")

    annotate_path(
        input_path=input_file,
        output_dir=output_dir,
        target_speaker="*CHI",
        engine=PunctuationEngine(),
    )

    output_lines = (output_dir / "sample.cha").read_text(encoding="utf-8").splitlines()
    assert output_lines[0] == "*CHI:\tbad [:: good] , too ?"


def test_jsonl_uses_tt_fields(case_root: Path) -> None:
    input_dir = case_root / "in"
    output_dir = case_root / "out"
    input_dir.mkdir()
    (input_dir / "records.jsonl").write_text(
        '{"speaker":"*CHI","utterance":"bad text"}\n'
        '{"speaker":"*INV","utterance":"bad text"}\n',
        encoding="utf-8",
    )

    annotate_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        target_speaker="*CHI",
        speaker_field="speaker",
        text_field="utterance",
        show_target=True,
        engine=StubEngine(),
    )

    payload = (output_dir / "records.jsonl").read_text(encoding="utf-8")
    assert "tt_annotated_text" in payload
    assert "tt_annotations" in payload
    assert "tt_line_confidence" in payload
    assert "tt_is_target_line" in payload


def test_single_jsonl_file_is_supported(case_root: Path) -> None:
    input_file = case_root / "records.jsonl"
    output_dir = case_root / "out"
    input_file.write_text(
        '{"speaker":"*CHI","utterance":"bad text"}\n'
        '{"speaker":"*INV","utterance":"bad text"}\n',
        encoding="utf-8",
    )

    annotate_path(
        input_path=input_file,
        output_dir=output_dir,
        target_speaker="*CHI",
        speaker_field="speaker",
        text_field="utterance",
        show_target=True,
        engine=StubEngine(),
    )

    payload = (output_dir / "records.jsonl").read_text(encoding="utf-8")
    assert "tt_annotated_text" in payload
    assert "tt_annotations" in payload
    assert "tt_line_confidence" in payload
    assert "tt_is_target_line" in payload


def test_unsupported_legacy_format_fails_with_clear_error(case_root: Path) -> None:
    input_dir = case_root / "in"
    output_dir = case_root / "out"
    input_dir.mkdir()
    (input_dir / "legacy.txt").write_text("*CHI:\tbad\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Only .cha and .jsonl are supported in adapter-only deployment.",
    ):
        annotate_folder(
            input_dir=input_dir,
            output_dir=output_dir,
            target_speaker="*CHI",
            engine=StubEngine(),
        )


def test_unsupported_legacy_format_reports_total_count_when_many(
    case_root: Path,
) -> None:
    input_dir = case_root / "in"
    output_dir = case_root / "out"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text("*CHI:\tone\n", encoding="utf-8")
    (input_dir / "b.csv").write_text("speaker,text\n*CHI,bad\n", encoding="utf-8")
    (input_dir / "c.json").write_text("{}", encoding="utf-8")
    (input_dir / "d.xlsx").write_bytes(b"placeholder")

    with pytest.raises(ValueError) as excinfo:
        annotate_folder(
            input_dir=input_dir,
            output_dir=output_dir,
            target_speaker="*CHI",
            engine=StubEngine(),
        )

    message = str(excinfo.value)
    assert "Only .cha and .jsonl are supported in adapter-only deployment." in message
    assert "(4 total)" in message


def test_chat_tokens_loaded_from_json() -> None:
    tokens = load_chat_tokens()
    assert isinstance(tokens, list)
    assert tokens
    assert len(tokens) == 53
    assert "[* m]" in tokens
    assert "&+" in tokens
