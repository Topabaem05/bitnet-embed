from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from bitnet_embed.data.preprocess import normalize_text
from bitnet_embed.data.schemas import PairExample, QueryDocumentExample, TripletExample
from bitnet_embed.modeling.prompts import PromptConfig, format_batch


class TokenizerLike(Protocol):
    def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, torch.Tensor]: ...


@dataclass(slots=True)
class PairCollator:
    tokenizer: TokenizerLike
    max_length: int
    prompt_config: PromptConfig

    def __call__(self, examples: Sequence[PairExample]) -> dict[str, Any]:
        anchor_texts = format_batch(
            [normalize_text(example.anchor) for example in examples],
            task="query",
            prompt_config=self.prompt_config,
        )
        positive_texts = format_batch(
            [normalize_text(example.positive) for example in examples],
            task="document",
            prompt_config=self.prompt_config,
        )
        return {
            "anchor": self.tokenizer(
                anchor_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ),
            "positive": self.tokenizer(
                positive_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ),
            "task": [example.task for example in examples],
            "source": [example.source for example in examples],
        }


@dataclass(slots=True)
class TripletCollator:
    tokenizer: TokenizerLike
    max_length: int
    prompt_config: PromptConfig

    def __call__(self, examples: Sequence[TripletExample]) -> dict[str, Any]:
        anchors = format_batch(
            [normalize_text(example.anchor) for example in examples],
            task="query",
            prompt_config=self.prompt_config,
        )
        positives = format_batch(
            [normalize_text(example.positive) for example in examples],
            task="document",
            prompt_config=self.prompt_config,
        )
        negatives = format_batch(
            [normalize_text(example.negative) for example in examples],
            task="document",
            prompt_config=self.prompt_config,
        )
        return {
            "anchor": self.tokenizer(
                anchors,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ),
            "positive": self.tokenizer(
                positives,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ),
            "negative": self.tokenizer(
                negatives,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ),
            "task": [example.task for example in examples],
            "source": [example.source for example in examples],
        }


@dataclass(slots=True)
class QueryDocumentCollator:
    tokenizer: TokenizerLike
    max_length: int
    prompt_config: PromptConfig

    def __call__(self, examples: Sequence[QueryDocumentExample]) -> dict[str, Any]:
        queries = format_batch(
            [normalize_text(example.query) for example in examples],
            task="query",
            prompt_config=self.prompt_config,
        )
        documents = format_batch(
            [normalize_text(example.document) for example in examples],
            task="document",
            prompt_config=self.prompt_config,
        )
        return {
            "query": self.tokenizer(
                queries,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ),
            "document": self.tokenizer(
                documents,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ),
            "label": torch.tensor([example.label for example in examples], dtype=torch.long),
            "source": [example.source for example in examples],
        }
