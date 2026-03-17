from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from talk_tag.api import annotate_folder


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="talk-tag",
        description="GPU-only transcript annotator for speaker-scoped CHAT corpora.",
    )
    subparsers = parser.add_subparsers(dest="command")

    annotate = subparsers.add_parser("annotate", help="Annotate files in a folder")
    annotate.add_argument("--input-dir", required=True, type=Path)
    annotate.add_argument("--output-dir", required=True, type=Path)
    annotate.add_argument("--target-speaker", required=True)
    annotate.add_argument("--investigator-speaker", default=None)

    annotate.add_argument("--hf-repo-id", default=None)
    annotate.add_argument("--hf-filename", default=None)
    annotate.add_argument("--hf-token", default=None)
    annotate.add_argument("--hf-cache-dir", type=Path, default=None)

    annotate.add_argument(
        "--granularity",
        choices=["light", "standard", "strict"],
        default="standard",
    )
    annotate.add_argument("--error-tag", action="append", default=[])
    annotate.add_argument("--show-target", action="store_true")
    annotate.add_argument("--speaker-field", default=None)
    annotate.add_argument("--text-field", default=None)
    annotate.add_argument("--csv-line-field", default=None)
    annotate.add_argument("--case-insensitive-speaker", action="store_true")
    annotate.add_argument("--no-progress", action="store_true")
    annotate.add_argument("--fail-fast", action="store_true")

    return parser


def _run_annotate(args: argparse.Namespace) -> int:
    try:
        summary = annotate_folder(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_speaker=args.target_speaker,
            investigator_speaker=args.investigator_speaker,
            hf_repo_id=args.hf_repo_id,
            hf_filename=args.hf_filename,
            hf_token=args.hf_token,
            hf_cache_dir=args.hf_cache_dir,
            granularity=args.granularity,
            error_tags=args.error_tag,
            show_target=args.show_target,
            speaker_field=args.speaker_field,
            text_field=args.text_field,
            csv_line_field=args.csv_line_field,
            case_insensitive_speaker=args.case_insensitive_speaker,
            continue_on_error=not args.fail_fast,
            show_progress=not args.no_progress,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Processed files: {summary.processed_files}")
    print(f"Failed files: {summary.failed_files}")
    print(f"Target lines: {summary.target_lines}")
    print(f"Annotated lines: {summary.annotated_lines}")
    print(f"Report: {summary.report_path}")
    if summary.failed_files > 0 and args.fail_fast:
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command != "annotate":
        parser.print_help()
        return 0
    return _run_annotate(args)


def app() -> None:
    raise SystemExit(main())
