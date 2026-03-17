from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only without optional dependency
    tqdm = None


def wrap_progress(
    iterable: Iterable[T],
    *,
    enabled: bool,
    total: int | None = None,
    desc: str = "",
) -> Iterable[T] | Iterator[T]:
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)
