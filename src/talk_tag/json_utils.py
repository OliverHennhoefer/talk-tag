from __future__ import annotations

import json
from typing import Any

orjson: Any | None
try:
    import orjson  # type: ignore
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    orjson = None


def loads(payload: bytes | str) -> Any:
    if orjson is not None:
        if isinstance(payload, str):
            return orjson.loads(payload.encode("utf-8"))
        return orjson.loads(payload)
    if isinstance(payload, bytes):
        return json.loads(payload.decode("utf-8"))
    return json.loads(payload)


def dumps(payload: Any, *, pretty: bool = False) -> bytes:
    if orjson is not None:
        option = orjson.OPT_INDENT_2 if pretty else 0
        return orjson.dumps(payload, option=option)
    if pretty:
        encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return encoded.encode("utf-8")
