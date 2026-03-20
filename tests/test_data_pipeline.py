from __future__ import annotations

from bitnet_embed.data.collators import PairCollator
from bitnet_embed.data.loaders import build_smoke_pairs
from bitnet_embed.data.preprocess import normalize_text
from bitnet_embed.modeling.prompts import PromptConfig
from bitnet_embed.modeling.smoke import ToyTokenizer


def test_normalize_text_removes_control_characters() -> None:
    assert normalize_text("hello\x00\nworld") == "hello world"


def test_pair_collator_builds_anchor_and_positive_batches() -> None:
    tokenizer = ToyTokenizer()
    collator = PairCollator(tokenizer, max_length=16, prompt_config=PromptConfig())
    batch = collator(build_smoke_pairs()[:2])
    assert batch["anchor"]["input_ids"].shape[0] == 2
    assert batch["positive"]["attention_mask"].shape[0] == 2
