from __future__ import annotations

from talk_tag.config import Granularity

PRESET_ERROR_TAGS: dict[Granularity, list[str]] = {
    "light": ["orthography", "punctuation", "capitalization"],
    "standard": [
        "orthography",
        "punctuation",
        "capitalization",
        "morphology",
        "agreement",
    ],
    "strict": [
        "orthography",
        "punctuation",
        "capitalization",
        "morphology",
        "agreement",
        "syntax",
        "lexical",
    ],
}


def resolve_active_error_tags(
    *,
    granularity: Granularity,
    error_tags: list[str],
) -> list[str]:
    if error_tags:
        seen: set[str] = set()
        resolved: list[str] = []
        for raw in error_tags:
            value = raw.strip()
            if value and value not in seen:
                seen.add(value)
                resolved.append(value)
        return resolved
    return PRESET_ERROR_TAGS[granularity][:]


def build_annotation_prompt(
    *,
    text: str,
    active_error_tags: list[str],
) -> str:
    tags = ", ".join(active_error_tags) if active_error_tags else "none"
    return (
        "You are a deterministic language annotation assistant.\n"
        "Task: find only errors in the allowed categories and propose corrections.\n"
        f"Allowed categories: {tags}\n"
        "Output strictly valid JSON with this exact schema:\n"
        "{"
        '"annotations":[{"source":"...", "target":"...", "error_tag":"...", '
        '"start":0, "end":0, "message":"..."}],'
        '"corrected_text":"..."'
        "}\n"
        "Rules:\n"
        "- start/end are character offsets in the original text.\n"
        "- If there are no corrections, return annotations as an empty list.\n"
        "- Do not include markdown fences.\n"
        f"Text:\n{text}"
    )
