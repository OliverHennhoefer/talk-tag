from __future__ import annotations

import types
from pathlib import Path

from talk_tag.doctor import DoctorCheck, run_doctor


def test_run_doctor_collects_expected_checks(
    case_root: Path,
    monkeypatch,
) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeMps:
        @staticmethod
        def is_available() -> bool:
            return False

    fake_torch = types.SimpleNamespace(
        cuda=FakeCuda(),
        backends=types.SimpleNamespace(mps=FakeMps()),
        __version__="0.0",
    )

    def fake_check_import(module_name: str, *, recommendation: str):
        if module_name == "torch":
            return (
                DoctorCheck(
                    name="import-torch",
                    ok=True,
                    detail="ok",
                ),
                fake_torch,
            )
        return (
            DoctorCheck(
                name=f"import-{module_name}",
                ok=True,
                detail="ok",
            ),
            object(),
        )

    monkeypatch.setattr("talk_tag.doctor._check_import", fake_check_import)
    monkeypatch.setattr(
        "talk_tag.doctor.probe_model_access",
        lambda **_kwargs: case_root / "hf-cache" / "config.json",
    )

    report = run_doctor(
        cache_dir=case_root / "hf-cache",
        device="auto",
        fix=False,
    )
    names = [item.name for item in report.checks]
    assert "python-version" in names
    assert "import-numpy" in names
    assert "import-torch" in names
    assert "import-transformers" in names
    assert "import-peft" in names
    assert "runtime-backend" in names
    assert "cache-write" in names
    assert "hf-default-model" in names


def test_run_doctor_uses_runtime_default_cache_when_unspecified(
    case_root: Path,
    monkeypatch,
) -> None:
    default_cache = case_root / "hf-default-cache"
    captured: dict[str, Path] = {}

    def fake_check_import(module_name: str, *, recommendation: str):
        return (
            DoctorCheck(
                name=f"import-{module_name}",
                ok=True,
                detail="ok",
            ),
            object(),
        )

    def fake_check_runtime(torch_module: object | None, *, device: str) -> DoctorCheck:
        return DoctorCheck(name="runtime-backend", ok=True, detail="ok")

    def fake_check_cache_dir(cache_dir: Path, *, fix: bool) -> DoctorCheck:
        captured["cache-write"] = cache_dir
        return DoctorCheck(name="cache-write", ok=True, detail="ok")

    def fake_check_default_model_access(*, cache_dir: Path) -> DoctorCheck:
        captured["hf-default-model"] = cache_dir
        return DoctorCheck(name="hf-default-model", ok=True, detail="ok")

    monkeypatch.setattr(
        "talk_tag.doctor._resolve_default_hf_cache_dir",
        lambda: default_cache,
    )
    monkeypatch.setattr("talk_tag.doctor._check_import", fake_check_import)
    monkeypatch.setattr("talk_tag.doctor._check_runtime", fake_check_runtime)
    monkeypatch.setattr("talk_tag.doctor._check_cache_dir", fake_check_cache_dir)
    monkeypatch.setattr(
        "talk_tag.doctor._check_default_model_access",
        fake_check_default_model_access,
    )

    run_doctor(cache_dir=None, device="auto", fix=False)

    assert captured["cache-write"] == default_cache
    assert captured["hf-default-model"] == default_cache
