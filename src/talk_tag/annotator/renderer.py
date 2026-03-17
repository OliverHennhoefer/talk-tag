from __future__ import annotations

from talk_tag.models import Annotation


def render_chat_markers(
    text: str,
    annotations: list[Annotation],
    *,
    show_target: bool,
) -> str:
    if not annotations:
        return text

    chunks: list[str] = []
    cursor = 0
    for annotation in sorted(annotations, key=lambda item: (item.start, item.end)):
        if annotation.start < cursor or annotation.start > len(text):
            continue
        if annotation.end < annotation.start or annotation.end > len(text):
            continue

        chunks.append(text[cursor : annotation.start])
        source_fragment = text[annotation.start : annotation.end]
        target_fragment = annotation.target if show_target else ""
        chunks.append(f"{source_fragment} [:: {target_fragment}]")
        cursor = annotation.end

    chunks.append(text[cursor:])
    return "".join(chunks)
