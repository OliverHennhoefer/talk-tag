from __future__ import annotations

import importlib
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path

from talk_tag.model.hf import probe_model_access, resolve_auth_token
from talk_tag.model.hf import (
    ADAPTER_FILENAME,
    ADAPTER_REPO_ID,
    BASE_MODEL_FILENAME,
    BASE_MODEL_REPO_ID,
)
from talk_tag.runtime import Device, select_fixed_deployment_device

MIN_PYTHON = (3, 10)


def _resolve_default_hf_cache_dir() -> Path:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE).resolve()
    except Exception:
        env_cache = os.environ.get("HF_HUB_CACHE")
        if env_cache:
            return Path(env_cache).resolve()
        return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


@dataclass(slots=True)
class DoctorCheck:
    name: str
    ok: bool
    detail: str
    recommendation: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "ok": self.ok,
            "detail": self.detail,
            "recommendation": self.recommendation,
        }


@dataclass(slots=True)
class DoctorReport:
    checks: list[DoctorCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(item.ok for item in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "checks": [item.to_dict() for item in self.checks],
        }


def _check_python_version() -> DoctorCheck:
    current = sys.version_info[:3]
    if current >= MIN_PYTHON:
        return DoctorCheck(
            name="python-version",
            ok=True,
            detail=f"Python {platform.python_version()} is supported.",
        )
    return DoctorCheck(
        name="python-version",
        ok=False,
        detail=f"Python {platform.python_version()} is below required {MIN_PYTHON[0]}.{MIN_PYTHON[1]}.",
        recommendation=f"Install Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer.",
    )


def _check_import(
    module_name: str, *, recommendation: str
) -> tuple[DoctorCheck, object | None]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return (
            DoctorCheck(
                name=f"import-{module_name}",
                ok=False,
                detail=f"Import failed: {exc}",
                recommendation=recommendation,
            ),
            None,
        )
    version = getattr(module, "__version__", "unknown")
    return (
        DoctorCheck(
            name=f"import-{module_name}",
            ok=True,
            detail=f"Import OK (version={version}).",
        ),
        module,
    )


def _check_cache_dir(cache_dir: Path, *, fix: bool) -> DoctorCheck:
    """Check whether the cache directory is writable.

    The only automatic remediation attempt is creating the directory with
    ``mkdir(parents=True, exist_ok=True)`` before running the write probe.
    """

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        probe = cache_dir / ".tt-write-check"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return DoctorCheck(
            name="cache-write",
            ok=True,
            detail=f"Cache is writable: {cache_dir}",
        )
    except Exception as exc:
        recommendation = (
            f"Set --hf-cache-dir to a writable location (current={cache_dir})."
        )
        if fix:
            recommendation = (
                "Directory creation was attempted, but the cache is still not writable. "
                f"Choose a writable directory with --hf-cache-dir (current={cache_dir})."
            )
        return DoctorCheck(
            name="cache-write",
            ok=False,
            detail=f"Cache write failed: {exc}",
            recommendation=recommendation,
        )


def _check_runtime(torch_module: object | None, *, device: Device) -> DoctorCheck:
    if torch_module is None:
        return DoctorCheck(
            name="runtime-backend",
            ok=False,
            detail="torch is unavailable; runtime backend cannot be selected.",
            recommendation="Install runtime dependencies, for example: pip install 'talk-tag[runtime]'.",
        )
    try:
        selection = select_fixed_deployment_device(
            requested=device,
            torch_module=torch_module,
        )
    except Exception as exc:
        return DoctorCheck(
            name="runtime-backend",
            ok=False,
            detail=f"Backend selection failed for device='{device}': {exc}",
            recommendation="Use --device auto or install the requested accelerator stack.",
        )
    detail = (
        f"Resolved backend: {selection.resolved} (requested={selection.requested})."
    )
    if selection.warning:
        detail = f"{detail} {selection.warning}"
    return DoctorCheck(name="runtime-backend", ok=True, detail=detail)


def _check_default_model_access(*, cache_dir: Path) -> DoctorCheck:
    token, auth_mode = resolve_auth_token()
    try:
        base_path = probe_model_access(
            repo_id=BASE_MODEL_REPO_ID,
            filename=BASE_MODEL_FILENAME,
            token=token,
            cache_dir=cache_dir,
        )
        adapter_path = probe_model_access(
            repo_id=ADAPTER_REPO_ID,
            filename=ADAPTER_FILENAME,
            token=token,
            cache_dir=cache_dir,
        )
    except Exception as exc:
        return DoctorCheck(
            name="hf-default-model",
            ok=False,
            detail=str(exc),
            recommendation=(
                "Ensure network access and a valid HF token when needed. "
                "For private/gated repos, set HF_TOKEN."
            ),
        )
    return DoctorCheck(
        name="hf-default-model",
        ok=True,
        detail=(
            "Deployment model metadata reachable "
            f"(base={base_path}, adapter={adapter_path}). auth={auth_mode}"
        ),
    )


def run_doctor(
    *,
    cache_dir: Path | None = None,
    device: Device = "auto",
    fix: bool = False,
) -> DoctorReport:
    active_cache = (
        cache_dir.resolve()
        if cache_dir is not None
        else _resolve_default_hf_cache_dir()
    )
    report = DoctorReport()
    report.checks.append(_check_python_version())

    numpy_check, _numpy_module = _check_import(
        "numpy",
        recommendation="Install numpy in this environment.",
    )
    report.checks.append(numpy_check)

    torch_check, torch_module = _check_import(
        "torch",
        recommendation="Install runtime dependencies, for example: pip install 'talk-tag[runtime]'.",
    )
    report.checks.append(torch_check)

    transformers_check, _transformers_module = _check_import(
        "transformers",
        recommendation="Install transformers, for example: pip install 'talk-tag[runtime]'.",
    )
    report.checks.append(transformers_check)

    peft_check, _peft_module = _check_import(
        "peft",
        recommendation="Install peft, for example: pip install 'talk-tag[runtime]'.",
    )
    report.checks.append(peft_check)

    report.checks.append(_check_runtime(torch_module, device=device))
    report.checks.append(_check_cache_dir(active_cache, fix=fix))
    report.checks.append(_check_default_model_access(cache_dir=active_cache))
    return report
