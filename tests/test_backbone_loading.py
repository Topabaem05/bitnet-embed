from __future__ import annotations

import sys
from types import SimpleNamespace

import torch.nn as nn
from _pytest.monkeypatch import MonkeyPatch

from bitnet_embed.modeling.backbone import BackboneConfig, BitNetBackbone


class _FakeTokenizer:
    pad_token: str | None = None
    eos_token = "<eos>"


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16, use_cache=True)
        self.gradient_checkpoint_enabled = False

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpoint_enabled = True


def test_bitnet_backbone_uses_configured_trust_remote_code(monkeypatch: MonkeyPatch) -> None:
    tokenizer_calls: list[dict[str, object]] = []
    model_calls: list[dict[str, object]] = []

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs: object) -> _FakeTokenizer:
            tokenizer_calls.append({"model_name": model_name, **kwargs})
            return _FakeTokenizer()

    class _AutoModel:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs: object) -> _FakeModel:
            model_calls.append({"model_name": model_name, **kwargs})
            return _FakeModel()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModel=_AutoModel, AutoTokenizer=_AutoTokenizer),
    )
    backbone = BitNetBackbone(
        BackboneConfig(
            model_name="example/bitnet",
            trust_remote_code=False,
            gradient_checkpointing=True,
        )
    )

    assert tokenizer_calls == [{"model_name": "example/bitnet", "trust_remote_code": False}]
    assert model_calls == [
        {
            "model_name": "example/bitnet",
            "dtype": backbone.config.dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": False,
            "use_safetensors": True,
        }
    ]
    assert backbone.tokenizer is not None
    assert backbone.tokenizer.pad_token == "<eos>"
    assert isinstance(backbone.backbone, _FakeModel)
    assert backbone.backbone.config.use_cache is False
    assert backbone.backbone.gradient_checkpoint_enabled is True
