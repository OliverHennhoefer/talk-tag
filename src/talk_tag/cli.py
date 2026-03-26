from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from talk_tag.api import StartupContext, annotate_path, pull_model
from talk_tag.doctor import run_doctor


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="talk-tag",
        description=(
            "Adapter-based TalkBank CHAT morphosyntactic annotator for .cha and .jsonl "
            "(CUDA-first, with CPU fallback; Apple MPS is not supported for the "
            "current fixed 4-bit deployment)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    annotate = subparsers.add_parser(
        "annotate",
        help="Annotate a single .cha/.jsonl file or all supported files in a folder",
    )
    input_group = annotate.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing .cha/.jsonl files to annotate",
    )
    input_group.add_argument(
        "--input-path",
        type=Path,
        help="Path to a single .cha/.jsonl file or a directory",
    )
    annotate.add_argument("--output-dir", required=True, type=Path)
    annotate.add_argument("--target-speaker", required=True)
    annotate.add_argument("--investigator-speaker", default=None)
    annotate.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
    )
    annotate.add_argument("--hf-cache-dir", type=Path, default=None)
    annotate.add_argument(
        "--granularity",
        choices=["light", "standard", "strict"],
        default="standard",
    )
    annotate.add_argument("--error-tag", action="append", default=[])
    annotate.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Debug/testing cap on the number of target utterances to annotate "
            "(0 = no limit). Output files are still written."
        ),
    )
    annotate.add_argument(
        "--show-target",
        action="store_true",
        help=(
            "Include optional real-word target reconstructions like [= target] "
            "in output. These are hidden by default."
        ),
    )
    annotate.add_argument(
        "--print-debug-lines",
        action="store_true",
        help=(
            "Print each changed target utterance as an original/annotated pair during "
            "the run. Intended for quick debugging."
        ),
    )
    annotate.add_argument("--speaker-field", default=None)
    annotate.add_argument("--text-field", default=None)
    annotate.add_argument("--case-insensitive-speaker", action="store_true")
    annotate.add_argument("--no-progress", action="store_true")
    annotate.add_argument("--fail-fast", action="store_true")

    doctor = subparsers.add_parser("doctor", help="Run environment preflight checks")
    doctor.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
    )
    doctor.add_argument("--hf-cache-dir", type=Path, default=None)
    doctor.add_argument("--fix", action="store_true")
    doctor.add_argument("--json", action="store_true")

    model = subparsers.add_parser("model", help="Model management commands")
    model.set_defaults(_command_parser=model)
    model_subparsers = model.add_subparsers(dest="model_command")

    pull = model_subparsers.add_parser(
        "pull",
        help="Download/cache the model and optionally verify runtime loading",
    )
    pull.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
    )
    pull.add_argument("--hf-cache-dir", type=Path, default=None)
    pull.add_argument("--no-verify-load", action="store_true")
    pull.add_argument("--json", action="store_true")

    return parser


def _print_startup_context(context: StartupContext) -> None:
    cache_display = context.cache_dir if context.cache_dir is not None else "(n/a)"
    print(
        "Runtime: "
        f"backend={context.backend} "
        f"model_source={context.model_source} "
        f"cache={cache_display} "
        f"auth={context.auth_mode}"
    )
    if context.warning:
        print(f"Runtime warning: {context.warning}")


def _run_annotate(args: argparse.Namespace) -> int:
    try:
        input_path = args.input_path if args.input_path is not None else args.input_dir
        if args.limit > 0:
            print(
                "Inference limit active: "
                f"annotating at most {args.limit} target utterances. "
                "Output files will still be written."
            )
        summary = annotate_path(
            input_path=input_path,
            output_dir=args.output_dir,
            target_speaker=args.target_speaker,
            investigator_speaker=args.investigator_speaker,
            device=args.device,
            hf_cache_dir=args.hf_cache_dir,
            granularity=args.granularity,
            error_tags=args.error_tag,
            limit=args.limit,
            show_target=args.show_target,
            print_debug_lines=args.print_debug_lines,
            speaker_field=args.speaker_field,
            text_field=args.text_field,
            case_insensitive_speaker=args.case_insensitive_speaker,
            continue_on_error=not args.fail_fast,
            show_progress=not args.no_progress,
            startup_callback=_print_startup_context,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Processed files: {summary.processed_files}")
    print(f"Failed files: {summary.failed_files}")
    print(f"Target lines: {summary.target_lines}")
    print(f"Annotated lines: {summary.annotated_lines}")
    print(f"Report: {summary.report_path}")
    if summary.failed_files > 0:
        return 1
    return 0


def _run_doctor(args: argparse.Namespace) -> int:
    report = run_doctor(
        cache_dir=args.hf_cache_dir,
        device=args.device,
        fix=args.fix,
    )
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"Doctor status: {'ok' if report.ok else 'failed'}")
        for check in report.checks:
            icon = "OK" if check.ok else "FAIL"
            print(f"[{icon}] {check.name}: {check.detail}")
            if check.recommendation and not check.ok:
                print(f"      recommendation: {check.recommendation}")
    return 0 if report.ok else 1


def _run_model_pull(args: argparse.Namespace) -> int:
    try:
        context = pull_model(
            hf_cache_dir=args.hf_cache_dir,
            device=args.device,
            verify_load=not args.no_verify_load,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(context.to_dict(), ensure_ascii=False, indent=2))
    else:
        _print_startup_context(context)
        print("Model pull completed.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "annotate":
        return _run_annotate(args)
    if args.command == "doctor":
        return _run_doctor(args)
    if args.command == "model" and args.model_command == "pull":
        return _run_model_pull(args)
    if args.command == "model":
        command_parser = getattr(args, "_command_parser", None)
        if command_parser is not None:
            command_parser.print_help()
            return 0
    parser.print_help()
    return 0


def app() -> None:
    raise SystemExit(main())


if __name__ == "__main__":  # pragma: no cover - exercised via `python -m talk_tag.cli`
    app()
