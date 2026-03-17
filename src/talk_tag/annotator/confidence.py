from __future__ import annotations

from difflib import SequenceMatcher
from typing import Sequence

from talk_tag.models import Annotation


def _clamp_01(value: float) -> float:
    return max(0.0, min(1.0, value))


def heuristic_confidence(source: str, target: str) -> float:
    ratio = SequenceMatcher(a=source, b=target).ratio()
    return _clamp_01(0.55 + ratio * 0.4)


def line_confidence(annotations: Sequence[Annotation]) -> float:
    if not annotations:
        return 1.0
    return _clamp_01(
        sum(item.confidence for item in annotations) / float(len(annotations))
    )
