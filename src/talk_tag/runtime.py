from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Device = Literal["auto", "cuda", "mps", "cpu"]
ResolvedDevice = Literal["cuda", "mps", "cpu"]


@dataclass(slots=True)
class RuntimeSelection:
    requested: Device
    resolved: ResolvedDevice
    warning: str | None = None

    @property
    def uses_fallback(self) -> bool:
        return self.requested == "auto" and self.resolved != "cuda"


def _mps_is_available(torch_module: Any) -> bool:
    backends = getattr(torch_module, "backends", None)
    if backends is None:
        return False
    mps = getattr(backends, "mps", None)
    if mps is None:
        return False
    checker = getattr(mps, "is_available", None)
    if checker is None:
        return False
    try:
        return bool(checker())
    except Exception:
        return False


def _cuda_is_available(torch_module: Any) -> bool:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        return False
    checker = getattr(cuda, "is_available", None)
    if checker is None:
        return False
    try:
        return bool(checker())
    except Exception:
        return False


def select_runtime_device(
    *,
    requested: Device = "auto",
    torch_module: Any | None = None,
) -> RuntimeSelection:
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, mps, cpu.")

    try:
        torch = torch_module if torch_module is not None else __import__("torch")
    except Exception as exc:  # pragma: no cover - exercised when dependency is absent
        raise RuntimeError(
            "torch is required for runtime selection. "
            "Install runtime dependencies with 'talk-tag[runtime]'."
        ) from exc

    cuda_available = _cuda_is_available(torch)
    mps_available = _mps_is_available(torch)

    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "CUDA was requested but no CUDA device is available. "
                "Use --device auto or --device cpu."
            )
        return RuntimeSelection(requested=requested, resolved="cuda")

    if requested == "mps":
        if not mps_available:
            raise RuntimeError(
                "MPS was requested but no Apple Metal backend is available. "
                "Use --device auto or --device cpu."
            )
        return RuntimeSelection(requested=requested, resolved="mps")

    if requested == "cpu":
        return RuntimeSelection(requested=requested, resolved="cpu")

    if cuda_available:
        return RuntimeSelection(requested="auto", resolved="cuda")
    if mps_available:
        return RuntimeSelection(
            requested="auto",
            resolved="mps",
            warning="CUDA not detected. Falling back to Apple MPS backend.",
        )
    return RuntimeSelection(
        requested="auto",
        resolved="cpu",
        warning="No CUDA/MPS backend detected. Falling back to CPU mode.",
    )
