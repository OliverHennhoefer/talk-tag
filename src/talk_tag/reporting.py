from __future__ import annotations

from pathlib import Path

from talk_tag.json_utils import dumps
from talk_tag.models import FileResult, RunSummary


def build_summary(
    *,
    input_dir: Path,
    output_dir: Path,
    started_at: str,
    ended_at: str,
    discovered_files: int,
    file_results: list[FileResult],
) -> RunSummary:
    failed_files = sum(1 for item in file_results if item.status == "failed")
    target_lines = sum(item.target_lines for item in file_results)
    annotated_lines = sum(item.annotated_lines for item in file_results)
    return RunSummary(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        started_at=started_at,
        ended_at=ended_at,
        total_files=discovered_files,
        processed_files=len(file_results),
        failed_files=failed_files,
        target_lines=target_lines,
        annotated_lines=annotated_lines,
        report_path="",
        files=file_results,
    )


def write_run_report(summary: RunSummary, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "_talk_tag_report.json"
    report_path.write_bytes(dumps(summary.to_dict(), pretty=True))
    return report_path
