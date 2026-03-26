from __future__ import annotations

import json
from pathlib import Path

import pytest

from talk_tag.api import annotate_path
from talk_tag.config import RunConfig
from talk_tag.formats.common import (
    normalize_chat_punctuation,
    normalize_chat_reconstructions,
)
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
        del show_target
        annotated_text = f"{text[:start]}bad [:: good]{text[end:]}"
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
    config.validate()


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
        config.validate()


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
        config.validate()


def test_cha_header_warning_for_missing_investigator(case_root: Path) -> None:
    input_dir = case_root / "in"
    output_dir = case_root / "out"
    input_dir.mkdir()

    source = "@Begin\n@Participants:\tCHI Child, MOT Mother\n*CHI:\tbad token.\n@End\n"
    (input_dir / "sample.cha").write_text(source, encoding="utf-8", newline="")

    annotate_path(
        input_path=input_dir,
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

    annotate_path(
        input_path=input_dir,
        output_dir=output_dir,
        target_speaker="*CHI",
        engine=StubEngine(),
    )
    output_lines = (output_dir / "sample.cha").read_text(encoding="utf-8").splitlines()
    assert output_lines[0] == "*CHI:\tbad one"
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
        engine=StubEngine(),
    )
    output_lines = (output_dir / "sample.cha").read_text(encoding="utf-8").splitlines()
    assert output_lines[0] == "*CHI:\tbad one"
    assert output_lines[1] == "*INV:\tbad two"


def test_limit_caps_target_utterances_across_cha_run(case_root: Path) -> None:
    input_file = case_root / "sample.cha"
    output_dir = case_root / "out"
    input_file.write_text(
        "*CHI:\tbad one\n*CHI:\tbad two\n*INV:\tbad three\n",
        encoding="utf-8",
    )

    annotate_path(
        input_path=input_file,
        output_dir=output_dir,
        target_speaker="*CHI",
        limit=1,
        engine=StubEngine(),
    )

    output_lines = (output_dir / "sample.cha").read_text(encoding="utf-8").splitlines()
    assert output_lines[0] == "*CHI:\tbad one"
    assert output_lines[1] == "*CHI:\tbad two"
    assert output_lines[2] == "*INV:\tbad three"


def test_chat_reconstruction_normalization_uses_clan_compatible_real_word_target() -> None:
    assert (
        normalize_chat_reconstructions("bad [:: good] text", show_target=True)
        == "bad [= good] text"
    )
    assert (
        normalize_chat_reconstructions("bad [:: good] text", show_target=False)
        == "bad text"
    )
    assert (
        normalize_chat_reconstructions(
            "goed [: went] [* m:=ed]",
            show_target=False,
        )
        == "goed [: went] [* m:=ed]"
    )


def test_show_target_true_keeps_real_word_reconstruction(case_root: Path) -> None:
    input_file = case_root / "sample.cha"
    output_dir = case_root / "out"
    input_file.write_text("*CHI:\tbad one\n", encoding="utf-8")

    annotate_path(
        input_path=input_file,
        output_dir=output_dir,
        target_speaker="*CHI",
        show_target=True,
        engine=StubEngine(),
    )

    output_lines = (output_dir / "sample.cha").read_text(encoding="utf-8").splitlines()
    assert output_lines[0] == "*CHI:\tbad [= good] one"


def test_chat_punctuation_spacing_only_compacts_whitelisted_chat_symbols() -> None:
    assert normalize_chat_punctuation("this is +// .") == "this is +//."
    assert normalize_chat_punctuation("this is +// ?") == "this is +//?"
    assert normalize_chat_punctuation("this is +// !") == "this is +//!"
    assert normalize_chat_punctuation("this is +/ .") == "this is +/."
    assert normalize_chat_punctuation("this is +/ ?") == "this is +/?"
    assert normalize_chat_punctuation("this is +/ !") == "this is +/!"
    assert normalize_chat_punctuation('said rabbit +" .') == 'said rabbit +".'
    assert normalize_chat_punctuation('said rabbit +" ?') == 'said rabbit +"?'
    assert normalize_chat_punctuation('said rabbit +" !') == 'said rabbit +"!'
    assert normalize_chat_punctuation('and then he said +"/ .') == 'and then he said +"/.'
    assert normalize_chat_punctuation('and then he said +"/ ?') == 'and then he said +"/?'
    assert normalize_chat_punctuation('and then he said +"/ !') == 'and then he said +"/!'
    assert normalize_chat_punctuation("( . )") == "(.)"
    assert normalize_chat_punctuation("( .. )") == "(..)"
    assert normalize_chat_punctuation("( . . . )") == "(...)"
    assert normalize_chat_punctuation("foo ( . . . ) bar") == "foo (...) bar"
    assert normalize_chat_punctuation("and what happened ?") == "and what happened ?"
    assert normalize_chat_punctuation("balloo:n .") == "balloo:n ."
    assert normalize_chat_punctuation("<the balaloo [* p:n]> [//]") == "<the balaloo [* p:n]> [//]"


def test_cha_output_only_compacts_split_chat_symbols(
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
                annotated_text='bad [:: good] and then +"/ ?',
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
    assert output_lines[0] == '*CHI:\tbad and then +"/?'


def test_cha_output_preserves_atomic_chat_symbols(case_root: Path) -> None:
    class AtomicSymbolEngine:
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
                annotated_text=(
                    "and he's happy he get [:: gets] [* m:03s:a] "
                    "balloo:n . \x15461475_463776\x15"
                ),
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
        show_target=True,
        engine=AtomicSymbolEngine(),
    )

    output_lines = (output_dir / "sample.cha").read_text(encoding="utf-8").splitlines()
    assert (
        output_lines[0]
        == "*CHI:\tand he's happy he get [= gets] [* m:03s:a] balloo:n . \x15461475_463776\x15"
    )


def test_jsonl_uses_tt_fields(case_root: Path) -> None:
    input_dir = case_root / "in"
    output_dir = case_root / "out"
    input_dir.mkdir()
    (input_dir / "records.jsonl").write_text(
        '{"speaker":"*CHI","utterance":"bad text"}\n'
        '{"speaker":"*INV","utterance":"bad text"}\n',
        encoding="utf-8",
    )

    annotate_path(
        input_path=input_dir,
        output_dir=output_dir,
        target_speaker="*CHI",
        speaker_field="speaker",
        text_field="utterance",
        engine=StubEngine(),
    )

    payload = (output_dir / "records.jsonl").read_text(encoding="utf-8")
    assert "tt_annotated_text" in payload
    assert "tt_annotations" in payload
    assert "tt_line_confidence" in payload
    assert "tt_is_target_line" in payload

    lines = [json.loads(line) for line in payload.strip().splitlines()]
    chi_line = next(line for line in lines if line["speaker"] == "*CHI")
    inv_line = next(line for line in lines if line["speaker"] == "*INV")

    assert chi_line["tt_is_target_line"] is True
    assert chi_line["tt_annotated_text"] == "bad text"
    assert len(chi_line["tt_annotations"]) == 1
    assert inv_line["tt_is_target_line"] is False
    assert inv_line["tt_annotated_text"] == "bad text"
    assert inv_line["tt_annotations"] == []


def test_limit_caps_target_utterances_across_jsonl_run(case_root: Path) -> None:
    input_file = case_root / "records.jsonl"
    output_dir = case_root / "out"
    input_file.write_text(
        '{"speaker":"*CHI","utterance":"bad one"}\n'
        '{"speaker":"*CHI","utterance":"bad two"}\n'
        '{"speaker":"*INV","utterance":"bad three"}\n',
        encoding="utf-8",
    )

    annotate_path(
        input_path=input_file,
        output_dir=output_dir,
        target_speaker="*CHI",
        speaker_field="speaker",
        text_field="utterance",
        limit=1,
        engine=StubEngine(),
    )

    payload = (output_dir / "records.jsonl").read_text(encoding="utf-8")
    lines = [json.loads(line) for line in payload.strip().splitlines()]
    assert lines[0]["tt_annotated_text"] == "bad one"
    assert lines[1]["tt_annotated_text"] == "bad two"
    assert lines[2]["tt_annotated_text"] == "bad three"


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
        engine=StubEngine(),
    )

    payload = (output_dir / "records.jsonl").read_text(encoding="utf-8")
    assert "tt_annotated_text" in payload
    assert "tt_annotations" in payload
    assert "tt_line_confidence" in payload
    assert "tt_is_target_line" in payload

    lines = [json.loads(line) for line in payload.strip().splitlines()]
    chi_line = next(line for line in lines if line["speaker"] == "*CHI")
    inv_line = next(line for line in lines if line["speaker"] == "*INV")

    assert chi_line["tt_is_target_line"] is True
    assert chi_line["tt_annotated_text"] == "bad text"
    assert len(chi_line["tt_annotations"]) == 1
    assert inv_line["tt_is_target_line"] is False
    assert inv_line["tt_annotated_text"] == "bad text"
    assert inv_line["tt_annotations"] == []


def test_unsupported_legacy_format_fails_with_clear_error(case_root: Path) -> None:
    input_dir = case_root / "in"
    output_dir = case_root / "out"
    input_dir.mkdir()
    (input_dir / "legacy.txt").write_text("*CHI:\tbad\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Only .cha and .jsonl are supported in adapter-only deployment.",
    ):
        annotate_path(
            input_path=input_dir,
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
        annotate_path(
            input_path=input_dir,
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
    # Keep this resilient to intentional vocabulary updates while still guarding
    # against unexpectedly small or malformed token bundles.
    assert len(tokens) >= 50
    assert len(tokens) == len(set(tokens))
    assert "[* m]" in tokens
    assert "&+" in tokens
