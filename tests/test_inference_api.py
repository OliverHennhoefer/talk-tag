from __future__ import annotations

import sys
import types

from talk_tag.inference import (
    DEFAULT_INSTRUCTION,
    InferenceConfig,
    TalkTagInference,
    build_deployment_prompt,
)
from talk_tag.model.deployment_loader import LoadedDeploymentModel
from talk_tag.runtime import RuntimeSelection


def test_prompt_wrapper_is_exact() -> None:
    wrapped = build_deployment_prompt(instruction="Do X", input_text="sample")
    assert wrapped == "### Instruction:\nDo X\n\n### Input:\nsample\n\n### Response:\n"


def test_default_instruction_sanitized_tail() -> None:
    assert not DEFAULT_INSTRUCTION.endswith("ß")
    assert (
        "Output exactly one annotated utterance line and nothing else."
        in DEFAULT_INSTRUCTION
    )


def test_inference_truncates_context_and_returns_one_line(monkeypatch) -> None:
    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    class FakeTorch(types.SimpleNamespace):
        @staticmethod
        def no_grad() -> FakeNoGrad:
            return FakeNoGrad()

    class FakeList:
        def __init__(self, values: list[int]) -> None:
            self._values = values

        def tolist(self) -> list[int]:
            return list(self._values)

    class FakeTensor:
        def __init__(self, rows: list[list[int]]) -> None:
            self.rows = rows
            self.shape = (len(rows), max((len(r) for r in rows), default=0))

        def to(self, device: str) -> "FakeTensor":
            return self

        def sum(self, dim: int) -> FakeList:
            assert dim == 1
            return FakeList([len(r) for r in self.rows])

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __init__(self) -> None:
            self.prompts: list[str] = []

        def __call__(
            self,
            prompts: list[str],
            *,
            return_tensors: str,
            padding: bool,
            truncation: bool,
            max_length: int,
        ) -> dict[str, FakeTensor]:
            self.prompts.extend(prompts)
            rows = [[1] * max(1, len(prompt)) for prompt in prompts]
            return {
                "input_ids": FakeTensor(rows),
                "attention_mask": FakeTensor(rows),
            }

        def decode(self, tokens: list[int], *, skip_special_tokens: bool) -> str:
            if not tokens:
                return ""
            if tokens[0] == 100:
                return "annotated one\nextra line"
            if tokens[0] == 101:
                return "   "
            return ""

    class FakeModel:
        def generate(self, **kwargs):
            input_ids = kwargs["input_ids"]
            generated: list[list[int]] = []
            for idx, row in enumerate(input_ids.rows):
                generated.append(row + [100 + idx])
            return generated

    fake_tokenizer = FakeTokenizer()
    loaded = LoadedDeploymentModel(
        model=FakeModel(),
        tokenizer=fake_tokenizer,
        runtime=RuntimeSelection(requested="auto", resolved="cpu", warning=None),
        added_tokens=0,
    )

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())

    inference = TalkTagInference(
        loaded_model=loaded,
        config=InferenceConfig(
            max_context_chars=5,
            max_new_tokens=32,
            max_seq_length=128,
            limit=0,
            do_sample=False,
        ),
    )

    outputs = inference.annotate_batch(["abcdefghi", "xy"])
    assert outputs == ["annotated one", "xy"]
    assert "### Input:\nabcde\n\n### Response:\n" in fake_tokenizer.prompts[0]
