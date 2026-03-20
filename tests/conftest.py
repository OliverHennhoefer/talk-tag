from __future__ import annotations

import os
from pathlib import Path

import pytest

# Keep pytest's built-in tmp_path behavior by default. Allow callers to opt
# into a custom temp root via TALK_TAG_PYTEST_TEMPROOT when needed.
if (
    "PYTEST_DEBUG_TEMPROOT" not in os.environ
    and "TALK_TAG_PYTEST_TEMPROOT" in os.environ
):
    try:
        _pytest_temp_root = Path(os.environ["TALK_TAG_PYTEST_TEMPROOT"]).resolve()
        _pytest_temp_root.mkdir(parents=True, exist_ok=True)
        os.environ["PYTEST_DEBUG_TEMPROOT"] = str(_pytest_temp_root)
    except OSError:
        # Fall back to pytest defaults when the requested temp root is unusable.
        pass


@pytest.fixture
def case_root(tmp_path: Path) -> Path:
    return tmp_path
