from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from talk_tag.model import deployment_loader


def test_load_deployment_model_uses_required_sequence(
    monkeypatch,
    case_root: Path,
) -> None:
    call_order: list[str] = []

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakeTorch(types.SimpleNamespace):
        cuda = FakeCuda()

    class FakeTokenizer:
        eos_token = "</s>"
        eos_token_id = 1
        pad_token_id = None
        pad_token = None

        def add_tokens(self, tokens: list[str], special_tokens: bool = False) -> int:
            call_order.append("add_tokens")
            assert tokens == ["@Begin", "@End"]
            assert special_tokens is False
            return len(tokens)

        def __len__(self) -> int:
            return 200

    class FakeModel:
        def resize_token_embeddings(self, size: int) -> None:
            call_order.append("resize_embeddings")
            assert size == 200

        def to(self, device: str) -> None:
            call_order.append(f"move_to:{device}")

        def eval(self) -> None:
            call_order.append("eval")

    fake_tokenizer = FakeTokenizer()
    fake_model = FakeModel()

    def _load_base_model(repo_id: str, **kwargs: object) -> FakeModel:
        call_order.append(f"load_base_model:{repo_id}")
        return fake_model

    def _load_tokenizer(repo_id: str, **kwargs: object) -> FakeTokenizer:
        call_order.append(f"load_tokenizer:{repo_id}")
        return fake_tokenizer

    def _load_adapter(model: FakeModel, repo_id: str, **kwargs: object) -> FakeModel:
        call_order.append(f"load_adapter:{repo_id}")
        return model

    fake_transformers = types.SimpleNamespace(
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=_load_base_model),
        AutoTokenizer=types.SimpleNamespace(from_pretrained=_load_tokenizer),
    )
    fake_peft = types.SimpleNamespace(
        PeftModel=types.SimpleNamespace(from_pretrained=_load_adapter)
    )

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setattr(
        deployment_loader,
        "load_chat_tokens",
        lambda: ["@Begin", "@End"],
    )

    deployment_loader.load_deployment_model(
        device="auto",
        hf_token=None,
        hf_cache_dir=case_root / "cache",
    )

    assert call_order[:5] == [
        "load_base_model:unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "load_tokenizer:unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "add_tokens",
        "resize_embeddings",
        "load_adapter:mash-mash/Llama_TalkTag_CHAT_error_annotator_adapter",
    ]
    assert "move_to:cuda" in call_order
    assert call_order[-1] == "eval"


def test_resize_only_when_tokens_added(monkeypatch, case_root: Path) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakeTorch(types.SimpleNamespace):
        cuda = FakeCuda()

    class FakeTokenizer:
        eos_token = "</s>"
        eos_token_id = 1
        pad_token_id = None
        pad_token = None

        def add_tokens(self, tokens: list[str], special_tokens: bool = False) -> int:
            return 0

        def __len__(self) -> int:
            return 123

    class FakeModel:
        resized = False

        def resize_token_embeddings(self, size: int) -> None:
            self.resized = True

        def to(self, device: str) -> None:
            return None

        def eval(self) -> None:
            return None

    fake_tokenizer = FakeTokenizer()
    fake_model = FakeModel()
    fake_transformers = types.SimpleNamespace(
        AutoModelForCausalLM=types.SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: fake_model
        ),
        AutoTokenizer=types.SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: fake_tokenizer
        ),
    )
    fake_peft = types.SimpleNamespace(
        PeftModel=types.SimpleNamespace(
            from_pretrained=lambda model, *_args, **_kwargs: model
        )
    )

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setattr(
        deployment_loader,
        "load_chat_tokens",
        lambda: ["@Begin"],
    )

    deployment_loader.load_deployment_model(
        device="auto",
        hf_token=None,
        hf_cache_dir=case_root / "cache",
    )
    assert fake_model.resized is False


def test_load_deployment_model_auto_falls_back_to_cpu_when_only_mps_is_available(
    monkeypatch,
    case_root: Path,
) -> None:
    moved_to: list[str] = []

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeMps:
        @staticmethod
        def is_available() -> bool:
            return True

    fake_torch = types.SimpleNamespace(
        cuda=FakeCuda(),
        backends=types.SimpleNamespace(mps=FakeMps()),
    )

    class FakeTokenizer:
        eos_token = "</s>"
        eos_token_id = 1
        pad_token_id = None
        pad_token = None

        def add_tokens(self, tokens: list[str], special_tokens: bool = False) -> int:
            return 0

        def __len__(self) -> int:
            return 123

    class FakeModel:
        def resize_token_embeddings(self, size: int) -> None:
            return None

        def to(self, device: str) -> None:
            moved_to.append(device)

        def eval(self) -> None:
            return None

    fake_model = FakeModel()
    fake_tokenizer = FakeTokenizer()
    fake_transformers = types.SimpleNamespace(
        AutoModelForCausalLM=types.SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: fake_model
        ),
        AutoTokenizer=types.SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: fake_tokenizer
        ),
    )
    fake_peft = types.SimpleNamespace(
        PeftModel=types.SimpleNamespace(
            from_pretrained=lambda model, *_args, **_kwargs: model
        )
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setattr(deployment_loader, "load_chat_tokens", lambda: ["@Begin"])

    bundle = deployment_loader.load_deployment_model(
        device="auto",
        hf_token=None,
        hf_cache_dir=case_root / "cache",
    )

    assert moved_to == ["cpu"]
    assert bundle.runtime.resolved == "cpu"
    assert bundle.runtime.warning is not None


def test_load_deployment_model_explicit_mps_raises_clear_error(
    monkeypatch,
    case_root: Path,
) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeMps:
        @staticmethod
        def is_available() -> bool:
            return True

    fake_torch = types.SimpleNamespace(
        cuda=FakeCuda(),
        backends=types.SimpleNamespace(mps=FakeMps()),
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(RuntimeError, match="Apple MPS is not supported"):
        deployment_loader.load_deployment_model(
            device="mps",
            hf_token=None,
            hf_cache_dir=case_root / "cache",
        )
