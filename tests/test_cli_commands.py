from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from talk_tag.api import StartupContext
from talk_tag.cli import main
from talk_tag.doctor import DoctorCheck, DoctorReport
from talk_tag.models import RunSummary


def test_cli_annotate_prints_startup_summary(
    case_root: Path,
    monkeypatch,
    capsys,
) -> None:
    input_dir = case_root / "in"
    output_dir = case_root / "out"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    def fake_annotate_path(**kwargs):
        callback = kwargs["startup_callback"]
        callback(
            StartupContext(
                backend="cpu",
                model_source="fixed_base_adapter",
                cache_dir=str(case_root / "cache"),
                auth_mode="env-token",
                warning="cpu fallback",
            )
        )
        return RunSummary(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            started_at="s",
            ended_at="e",
            total_files=1,
            processed_files=1,
            failed_files=0,
            target_lines=1,
            annotated_lines=1,
            report_path=str(output_dir / "_talk_tag_report.json"),
            files=[],
        )

    monkeypatch.setattr("talk_tag.cli.annotate_path", fake_annotate_path)
    exit_code = main(
        [
            "annotate",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--target-speaker",
            "*CHI",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Runtime: backend=cpu" in captured.out
    assert "Processed files: 1" in captured.out


def test_cli_model_pull_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "talk_tag.cli.pull_model",
        lambda **_kwargs: StartupContext(
            backend="cuda",
            model_source="fixed_base_adapter",
            cache_dir="cache",
            auth_mode="env-token",
            warning=None,
        ),
    )
    exit_code = main(["model", "pull", "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["backend"] == "cuda"
    assert payload["model_source"] == "fixed_base_adapter"


def test_cli_model_pull_json_allows_null_cache(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "talk_tag.cli.pull_model",
        lambda **_kwargs: StartupContext(
            backend="external",
            model_source="external_engine",
            cache_dir=None,
            auth_mode="external",
            warning=None,
        ),
    )
    exit_code = main(["model", "pull", "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["cache_dir"] is None


def test_cli_model_without_subcommand_prints_model_help(capsys) -> None:
    exit_code = main(["model"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "usage: talk-tag model" in captured.out
    assert "pull" in captured.out


def test_cli_annotate_prints_na_cache_for_external_engine(
    case_root: Path,
    monkeypatch,
    capsys,
) -> None:
    input_dir = case_root / "in"
    output_dir = case_root / "out"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    def fake_annotate_path(**kwargs):
        callback = kwargs["startup_callback"]
        callback(
            StartupContext(
                backend="external",
                model_source="external_engine",
                cache_dir=None,
                auth_mode="external",
                warning=None,
            )
        )
        return RunSummary(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            started_at="s",
            ended_at="e",
            total_files=1,
            processed_files=1,
            failed_files=0,
            target_lines=1,
            annotated_lines=1,
            report_path=str(output_dir / "_talk_tag_report.json"),
            files=[],
        )

    monkeypatch.setattr("talk_tag.cli.annotate_path", fake_annotate_path)
    exit_code = main(
        [
            "annotate",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--target-speaker",
            "*CHI",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "cache=(n/a)" in captured.out


def test_cli_annotate_returns_nonzero_when_any_file_fails(
    case_root: Path,
    monkeypatch,
    capsys,
) -> None:
    input_dir = case_root / "in"
    output_dir = case_root / "out"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    def fake_annotate_path(**kwargs):
        callback = kwargs["startup_callback"]
        callback(
            StartupContext(
                backend="cpu",
                model_source="fixed_base_adapter",
                cache_dir=str(case_root / "cache"),
                auth_mode="env-token",
                warning=None,
            )
        )
        return RunSummary(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            started_at="s",
            ended_at="e",
            total_files=2,
            processed_files=2,
            failed_files=1,
            target_lines=1,
            annotated_lines=1,
            report_path=str(output_dir / "_talk_tag_report.json"),
            files=[],
        )

    monkeypatch.setattr("talk_tag.cli.annotate_path", fake_annotate_path)
    exit_code = main(
        [
            "annotate",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--target-speaker",
            "*CHI",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Failed files: 1" in captured.out


def test_cli_annotate_accepts_single_input_path(
    case_root: Path,
    monkeypatch,
    capsys,
) -> None:
    input_file = case_root / "sample.cha"
    output_dir = case_root / "out"
    input_file.write_text("*CHI:\ttest\n", encoding="utf-8")
    output_dir.mkdir(parents=True)
    captured_kwargs: dict[str, object] = {}

    def fake_annotate_path(**kwargs):
        captured_kwargs.update(kwargs)
        callback = kwargs["startup_callback"]
        callback(
            StartupContext(
                backend="cpu",
                model_source="fixed_base_adapter",
                cache_dir=str(case_root / "cache"),
                auth_mode="env-token",
                warning=None,
            )
        )
        return RunSummary(
            input_dir=str(input_file),
            output_dir=str(output_dir),
            started_at="s",
            ended_at="e",
            total_files=1,
            processed_files=1,
            failed_files=0,
            target_lines=1,
            annotated_lines=1,
            report_path=str(output_dir / "_talk_tag_report.json"),
            files=[],
        )

    monkeypatch.setattr("talk_tag.cli.annotate_path", fake_annotate_path)
    exit_code = main(
        [
            "annotate",
            "--input-path",
            str(input_file),
            "--output-dir",
            str(output_dir),
            "--target-speaker",
            "*CHI",
        ]
    )
    captured = capsys.readouterr()
    assert captured_kwargs["input_path"] == input_file
    assert exit_code == 0
    assert "Processed files: 1" in captured.out


def test_cli_annotate_passes_batch_size_and_limit(
    case_root: Path,
    monkeypatch,
    capsys,
) -> None:
    input_file = case_root / "sample.cha"
    output_dir = case_root / "out"
    input_file.write_text("*CHI:\ttest\n", encoding="utf-8")
    output_dir.mkdir(parents=True)
    captured_kwargs: dict[str, object] = {}

    def fake_annotate_path(**kwargs):
        captured_kwargs.update(kwargs)
        callback = kwargs["startup_callback"]
        callback(
            StartupContext(
                backend="cpu",
                model_source="fixed_base_adapter",
                cache_dir=str(case_root / "cache"),
                auth_mode="env-token",
                warning=None,
            )
        )
        return RunSummary(
            input_dir=str(input_file),
            output_dir=str(output_dir),
            started_at="s",
            ended_at="e",
            total_files=1,
            processed_files=1,
            failed_files=0,
            target_lines=1,
            annotated_lines=1,
            report_path=str(output_dir / "_talk_tag_report.json"),
            files=[],
        )

    monkeypatch.setattr("talk_tag.cli.annotate_path", fake_annotate_path)
    exit_code = main(
        [
            "annotate",
            "--input-path",
            str(input_file),
            "--output-dir",
            str(output_dir),
            "--target-speaker",
            "*CHI",
            "--batch-size",
            "2",
            "--limit",
            "5",
        ]
    )
    captured = capsys.readouterr()
    assert captured_kwargs["batch_size"] == 2
    assert captured_kwargs["limit"] == 5
    assert exit_code == 0
    assert "Processed files: 1" in captured.out
    assert "Inference limit active: annotating at most 5 target utterances." in captured.out


def test_cli_annotate_accepts_single_jsonl_input_path(
    case_root: Path,
    monkeypatch,
    capsys,
) -> None:
    input_file = case_root / "records.jsonl"
    output_dir = case_root / "out"
    input_file.write_text(
        '{"speaker":"*CHI","utterance":"bad text"}\n',
        encoding="utf-8",
    )
    output_dir.mkdir(parents=True)
    captured_kwargs: dict[str, object] = {}

    def fake_annotate_path(**kwargs):
        captured_kwargs.update(kwargs)
        callback = kwargs["startup_callback"]
        callback(
            StartupContext(
                backend="cpu",
                model_source="fixed_base_adapter",
                cache_dir=str(case_root / "cache"),
                auth_mode="env-token",
                warning=None,
            )
        )
        return RunSummary(
            input_dir=str(input_file),
            output_dir=str(output_dir),
            started_at="s",
            ended_at="e",
            total_files=1,
            processed_files=1,
            failed_files=0,
            target_lines=1,
            annotated_lines=1,
            report_path=str(output_dir / "_talk_tag_report.json"),
            files=[],
        )

    monkeypatch.setattr("talk_tag.cli.annotate_path", fake_annotate_path)
    exit_code = main(
        [
            "annotate",
            "--input-path",
            str(input_file),
            "--output-dir",
            str(output_dir),
            "--target-speaker",
            "*CHI",
            "--speaker-field",
            "speaker",
            "--text-field",
            "utterance",
        ]
    )
    captured = capsys.readouterr()
    assert captured_kwargs["input_path"] == input_file
    assert captured_kwargs["speaker_field"] == "speaker"
    assert captured_kwargs["text_field"] == "utterance"
    assert exit_code == 0
    assert "Processed files: 1" in captured.out


def test_cli_doctor_exit_code(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "talk_tag.cli.run_doctor",
        lambda **_kwargs: DoctorReport(
            checks=[
                DoctorCheck(
                    name="python-version",
                    ok=False,
                    detail="bad",
                    recommendation="fix",
                )
            ]
        ),
    )
    exit_code = main(["doctor"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Doctor status: failed" in captured.out


def test_cli_annotate_rejects_removed_model_override_flag(
    case_root: Path,
    capsys,
) -> None:
    input_dir = case_root / "in"
    output_dir = case_root / "out"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    (input_dir / "sample.cha").write_text("*CHI:\ttest\n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "annotate",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--target-speaker",
                "*CHI",
                "--expert-model-id",
                "acme/custom",
            ]
        )
    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert "unrecognized arguments: --expert-model-id" in captured.err


def test_cli_module_entrypoint_executes_main() -> None:
    env = os.environ.copy()
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        src_path
        if not current_pythonpath
        else f"{src_path}{os.pathsep}{current_pythonpath}"
    )
    result = subprocess.run(
        [sys.executable, "-m", "talk_tag.cli", "--help"],
        text=True,
        capture_output=True,
        env=env,
    )
    assert result.returncode == 0
    assert "usage: talk-tag" in result.stdout
