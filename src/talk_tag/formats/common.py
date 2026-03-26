from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from talk_tag.config import Granularity, RunConfig
from talk_tag.models import LineResult

CANONICAL_SPEAKER_LINE_RE = re.compile(r"^(?P<token>\*[A-Z0-9]{1,8}):\t(?P<body>.*)$")
SPEAKER_LINE_RE = re.compile(r"^(?P<token>\*[A-Z0-9]{1,8}):[ \t]+(?P<body>.*)$")
PARTICIPANTS_LINE_RE = re.compile(r"^@Participants:\s*(?P<body>.*)$")
PARTICIPANT_TOKEN_RE = re.compile(r"^\*?(?P<code>[A-Z0-9]{1,8})\b")
LINE_ENDING_RE = re.compile(r"(\r\n|\n|\r)$")
REAL_WORD_RECONSTRUCTION_RE = re.compile(r"\[::\s*([^\]]*?)\]")
OPTIONAL_REAL_WORD_TARGET_RE = re.compile(r"\s*\[(?:::|=)\s*[^\]]*?\]")
CHAT_SPLIT_TERMINATOR_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\+//\s+\."), "+//."),
    (re.compile(r"\+//\s+\?"), "+//?"),
    (re.compile(r"\+//\s+!"), "+//!"),
    (re.compile(r"\+/\s+\."), "+/."),
    (re.compile(r"\+/\s+\?"), "+/?"),
    (re.compile(r"\+/\s+!"), "+/!"),
    (re.compile(r'\+"\s+\.'), '+".'),
    (re.compile(r'\+"\s+\?'), '+"?'),
    (re.compile(r'\+"\s+!'), '+"!'),
    (re.compile(r'\+"/\s+\.'), '+"/.'),
    (re.compile(r'\+"/\s+\?'), '+"/?'),
    (re.compile(r'\+"/\s+!'), '+"/!'),
)
CHAT_SPLIT_PAUSE_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\(\s+\.\s+\)"), "(.)"),
    (re.compile(r"\(\s+\.\.\s+\)"), "(..)"),
    (re.compile(r"\(\s+\.\s+\.\s+\.\s+\)"), "(...)"),
)


@dataclass(slots=True)
class ProcessedTextLine:
    output_line: str
    is_target_line: bool
    was_annotated: bool
    line_result: LineResult | None


class AnnotationEngine(Protocol):
    def annotate_line(
        self,
        text: str,
        *,
        granularity: Granularity,
        error_tags: list[str],
        show_target: bool,
    ) -> LineResult: ...


def split_line_ending(line: str) -> tuple[str, str]:
    match = LINE_ENDING_RE.search(line)
    if not match:
        return line, ""
    return line[: match.start()], match.group(1)


def normalize_speaker_prefix(content: str) -> str:
    match = SPEAKER_LINE_RE.match(content)
    if not match:
        return content
    return f"{match.group('token')}:\t{match.group('body')}"


def normalize_chat_punctuation(text: str) -> str:
    normalized = text
    for pattern, replacement in CHAT_SPLIT_TERMINATOR_REPLACEMENTS:
        normalized = pattern.sub(replacement, normalized)
    for pattern, replacement in CHAT_SPLIT_PAUSE_REPLACEMENTS:
        normalized = pattern.sub(replacement, normalized)
    return normalized


def normalize_chat_reconstructions(text: str, *, show_target: bool) -> str:
    normalized = REAL_WORD_RECONSTRUCTION_RE.sub(r"[= \1]", text)
    if show_target:
        return normalized
    return OPTIONAL_REAL_WORD_TARGET_RE.sub("", normalized)


def passthrough_result(text: str, *, is_target_line: bool) -> LineResult:
    return LineResult(
        original_text=text,
        annotated_text=text,
        annotations=[],
        line_confidence=1.0,
        is_target_line=is_target_line,
        confidence_source="heuristic",
    )


def validate_participants_header(lines: list[str], config: RunConfig) -> list[str]:
    participants_line = ""
    for line in lines:
        content, _ = split_line_ending(line)
        if PARTICIPANTS_LINE_RE.match(content):
            participants_line = content
            break

    if not participants_line:
        if config.investigator_speaker is None:
            return []
        return [
            "Missing @Participants header; could not validate investigator speaker."
        ]

    match = PARTICIPANTS_LINE_RE.match(participants_line)
    if not match:
        return ["Invalid @Participants header format."]
    body = match.group("body")

    participant_codes: set[str] = set()
    for chunk in body.split(","):
        item = chunk.strip()
        token_match = PARTICIPANT_TOKEN_RE.match(item)
        if not token_match:
            continue
        participant_codes.add(token_match.group("code").upper())

    warnings: list[str] = []
    target_code = config.target_speaker.lstrip("*").upper()
    if target_code not in participant_codes:
        warnings.append(
            f"@Participants does not include target speaker '{config.target_speaker}'."
        )

    if config.investigator_speaker:
        investigator_code = config.investigator_speaker.lstrip("*").upper()
        if investigator_code not in participant_codes:
            warnings.append(
                "@Participants does not include investigator speaker "
                f"'{config.investigator_speaker}'."
            )
    return warnings


def process_speaker_prefixed_line(
    raw_line: str,
    *,
    config: RunConfig,
    engine: AnnotationEngine,
) -> ProcessedTextLine:
    content, ending = split_line_ending(raw_line)
    normalized = normalize_speaker_prefix(content)
    match = CANONICAL_SPEAKER_LINE_RE.match(normalized)
    if not match:
        return ProcessedTextLine(
            output_line=normalized + ending,
            is_target_line=False,
            was_annotated=False,
            line_result=None,
        )

    token = match.group("token")
    body = match.group("body")
    if not config.speaker_matches(token):
        return ProcessedTextLine(
            output_line=normalized + ending,
            is_target_line=False,
            was_annotated=False,
            line_result=None,
        )

    line_result = engine.annotate_line(
        body,
        granularity=config.granularity,
        error_tags=config.error_tags,
        show_target=config.show_target,
    )
    line_result.annotated_text = normalize_chat_reconstructions(
        line_result.annotated_text,
        show_target=config.show_target,
    )
    line_result.annotated_text = normalize_chat_punctuation(line_result.annotated_text)
    rebuilt_line = f"{token}:\t{line_result.annotated_text}{ending}"
    return ProcessedTextLine(
        output_line=rebuilt_line,
        is_target_line=True,
        was_annotated=line_result.annotated_text != body,
        line_result=line_result,
    )
