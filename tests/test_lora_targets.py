from __future__ import annotations

from bitnet_embed.modeling.lora import DEFAULT_LORA_TARGET_MODULES, resolve_lora_target_modules


def test_resolve_lora_target_modules_finds_expected_names() -> None:
    module_names = [f"layers.0.{name}" for name in DEFAULT_LORA_TARGET_MODULES] + ["layers.0.norm"]
    resolved = resolve_lora_target_modules(module_names)
    assert resolved == DEFAULT_LORA_TARGET_MODULES
