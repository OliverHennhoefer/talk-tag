from __future__ import annotations

import csv
import shutil
import sys
import types
import uuid
from pathlib import Path

import pytest

from talk_tag.api import annotate_folder
from talk_tag.config import RunConfig
from talk_tag.formats.common import normalize_chat_punctuation
from talk_tag.model.transformers_engine import TransformersAnnotator, load_chat_tokens
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


@pytest.fixture
def case_root() -> Path:
    root = Path.cwd() / ".test_workspace" / str(uuid.uuid4())
    root.mkdir(parents=True, exist_ok=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)


def test_target_speaker_accepts_flexible_code(case_root: Path) -> None:
    input_dir = case_root / "in"
    output_dir = case_root / "out"
    input_dir.mkdir()

    config = RunConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        target_speaker="*SUBJ1",
    )
    config.validate(require_model_source=False)


def test_target_speaker_rejects_invalid_code(case_root: Path) -> None:
    input_dir = case_root / "in"
    output_dir = case_root / "out"
    input_dir.mkdir()

    config = RunConfig(
        input_dir=input_dir,
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
        input_dir=input_dir,
        output_dir=output_dir,
        target_speaker="*CHI",
    )
    with pytest.raises(ValueError):
        config.validate(require_model_source=False)


def test_chat_punctuation_normalizer_preserves_decimals() -> None:
    value = normalize_chat_punctuation("it is 3.14,bad.")
    assert "3.14" in value
    assert value.endswith("bad .")
    assert "3.14 , bad ." in value


def test_cha_header_warning_for_missing_investigator(case_root: Path) -> None:
    input_dir = case_root / "in"
    output_dir = case_root / "out"
    input_dir.mkdir()

    source = (
        "@Begin\n"
        "@Participants:\tCHI Child, MOT Mother\n"
        "*CHI:\tbad token.\n"
        "@End\n"
    )
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


def test_only_target_speaker_is_annotated(case_root: Path) -> None:
    input_dir = case_root / "in"
    output_dir = case_root / "out"
    input_dir.mkdir()
    (input_dir / "sample.txt").write_text(
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
    output_lines = (output_dir / "sample.txt").read_text(encoding="utf-8").splitlines()
    assert output_lines[0] == "*CHI:\tbad [:: good] one"
    assert output_lines[1] == "*INV:\tbad two"


def test_csv_chat_like_rows_and_tt_columns(case_root: Path) -> None:
    input_dir = case_root / "in"
    output_dir = case_root / "out"
    input_dir.mkdir()
    csv_path = input_dir / "records.csv"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["line"])
        writer.writerow(["@Participants:\tCHI Child, INV Investigator"])
        writer.writerow(["*CHI:\tbad,token."])
        writer.writerow(["*INV:\tbad,token."])

    annotate_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        target_speaker="*CHI",
        investigator_speaker="*INV",
        csv_line_field="line",
        show_target=True,
        engine=StubEngine(),
    )

    with (output_dir / "records.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert "tt_annotated_text" in reader.fieldnames
    assert "tt_annotations" in reader.fieldnames
    assert "tt_line_confidence" in reader.fieldnames
    assert "tt_is_target_line" in reader.fieldnames
    assert rows[1]["line"] == "*CHI:\tbad [:: good] , token ."
    assert rows[2]["line"] == "*INV:\tbad,token."
    assert rows[1]["tt_is_target_line"] in {"True", "true", "1"}
    assert rows[2]["tt_is_target_line"] in {"False", "false", "0"}


def test_json_requires_explicit_field_mapping(case_root: Path) -> None:
    input_dir = case_root / "in"
    output_dir = case_root / "out"
    input_dir.mkdir()
    (input_dir / "records.json").write_text(
        '[{"speaker":"*CHI","utterance":"bad text"}]',
        encoding="utf-8",
    )

    summary = annotate_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        target_speaker="*CHI",
        continue_on_error=True,
        engine=StubEngine(),
    )

    assert summary.failed_files == 1
    report = (output_dir / "_talk_tag_report.json").read_text(encoding="utf-8")
    assert "speaker_field and text_field are required" in report


def test_json_uses_tt_fields(case_root: Path) -> None:
    input_dir = case_root / "in"
    output_dir = case_root / "out"
    input_dir.mkdir()
    (input_dir / "records.json").write_text(
        '[{"speaker":"*CHI","utterance":"bad text"}]',
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

    payload = (output_dir / "records.json").read_text(encoding="utf-8")
    assert "tt_annotated_text" in payload
    assert "tt_annotations" in payload
    assert "tt_line_confidence" in payload
    assert "tt_is_target_line" in payload


def test_output_tree_mirror_and_report_name(case_root: Path) -> None:
    input_dir = case_root / "in"
    output_dir = case_root / "out"
    nested = input_dir / "a" / "b"
    nested.mkdir(parents=True)
    (nested / "sample.txt").write_text("*CHI:\tbad token\n", encoding="utf-8")

    annotate_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        target_speaker="*CHI",
        engine=StubEngine(),
    )

    assert (output_dir / "a" / "b" / "sample.txt").exists()
    assert (output_dir / "_talk_tag_report.json").exists()


def test_chat_tokens_loaded_from_json() -> None:
    tokens = load_chat_tokens()
    assert isinstance(tokens, list)
    assert tokens


def test_embedding_resize_called_for_new_chat_tokens(
    case_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeNoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
            return False

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakeTorch(types.SimpleNamespace):
        cuda = FakeCuda()

        @staticmethod
        def no_grad() -> FakeNoGrad:
            return FakeNoGrad()

    class FakeTokenizer:
        eos_token = "</s>"
        eos_token_id = 1
        pad_token = None
        pad_token_id = None

        def __init__(self) -> None:
            self._tokens: list[str] = []

        def get_vocab(self) -> dict[str, int]:
            return {}

        def add_tokens(self, tokens: list[str], special_tokens: bool = False) -> int:
            self._tokens.extend(tokens)
            return len(tokens)

        def __len__(self) -> int:
            return 100 + len(self._tokens)

    class FakeTensor:
        def __init__(self, length: int = 1) -> None:
            self.shape = (1, length)

        def to(self, device: str) -> "FakeTensor":
            return self

        def __getitem__(self, item: object) -> "FakeTensor":
            return self

    class FakeModel:
        def __init__(self) -> None:
            self.resized_to: int | None = None

        def to(self, device: str) -> None:
            return None

        def eval(self) -> None:
            return None

        def resize_token_embeddings(self, size: int) -> None:
            self.resized_to = size

    fake_tokenizer = FakeTokenizer()
    fake_model = FakeModel()
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: fake_tokenizer
        ),
        AutoModelForCausalLM=types.SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: fake_model
        ),
    )

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    TransformersAnnotator(
        model_dir=case_root,
        chat_tokens=["@Participants:", "@Begin"],
    )
    assert fake_model.resized_to == len(fake_tokenizer)
