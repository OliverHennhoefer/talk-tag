from __future__ import annotations

import types

import pytest

from talk_tag.runtime import select_runtime_device


def _fake_torch(*, cuda: bool, mps: bool) -> object:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return cuda

    class FakeMps:
        @staticmethod
        def is_available() -> bool:
            return mps

    return types.SimpleNamespace(
        cuda=FakeCuda(),
        backends=types.SimpleNamespace(mps=FakeMps()),
    )


def test_auto_prefers_cuda() -> None:
    selected = select_runtime_device(
        requested="auto",
        torch_module=_fake_torch(cuda=True, mps=True),
    )
    assert selected.resolved == "cuda"
    assert selected.warning is None


def test_auto_falls_back_to_mps() -> None:
    selected = select_runtime_device(
        requested="auto",
        torch_module=_fake_torch(cuda=False, mps=True),
    )
    assert selected.resolved == "mps"
    assert selected.warning is not None


def test_auto_falls_back_to_cpu() -> None:
    selected = select_runtime_device(
        requested="auto",
        torch_module=_fake_torch(cuda=False, mps=False),
    )
    assert selected.resolved == "cpu"
    assert selected.warning is not None


def test_explicit_cuda_missing_raises() -> None:
    with pytest.raises(RuntimeError):
        select_runtime_device(
            requested="cuda",
            torch_module=_fake_torch(cuda=False, mps=False),
        )


def test_auto_handles_missing_cuda_attr() -> None:
    selected = select_runtime_device(
        requested="auto",
        torch_module=types.SimpleNamespace(
            backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(is_available=lambda: False)
            )
        ),
    )
    assert selected.resolved == "cpu"
    assert selected.warning is not None


def test_explicit_cpu_is_supported() -> None:
    selected = select_runtime_device(
        requested="cpu",
        torch_module=_fake_torch(cuda=False, mps=False),
    )
    assert selected.resolved == "cpu"
    assert selected.warning is None
