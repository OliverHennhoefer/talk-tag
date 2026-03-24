from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

# Prefer project-local temp roots to avoid host TEMP permission issues.
_default_temp_root = (Path.cwd() / ".test_workspace").resolve()
_env_temp_root = os.environ.get("TALK_TAG_PYTEST_TEMPROOT")
_active_temp_root = (
    Path(_env_temp_root).resolve() if _env_temp_root else _default_temp_root
)
_active_temp_root.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def case_root() -> Generator[Path, None, None]:
    root = (_active_temp_root / str(uuid.uuid4())).resolve()
    root.mkdir(parents=True, exist_ok=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)

