from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from torch.utils.data import DataLoader, IterableDataset

from bitnet_embed.data.collators import PairCollator
from bitnet_embed.data.loaders import ExampleDataset, build_smoke_pairs
from bitnet_embed.data.schemas import PairExample
from bitnet_embed.losses.contrastive import SymmetricInfoNCELoss
from bitnet_embed.modeling.prompts import PromptConfig
from bitnet_embed.modeling.smoke import build_toy_embedding_model
from bitnet_embed.train.factory import freeze_backbone
from bitnet_embed.train.trainer import EmbeddingTrainer, TrainingConfig


class UnknownLengthPairDataset(IterableDataset[PairExample]):
    def __iter__(self) -> Iterator[PairExample]:
        yield from build_smoke_pairs() * 4


def test_global_step_counts_optimizer_updates_under_grad_accum(tmp_path: Path) -> None:
    model = build_toy_embedding_model(projection_dim=8)
    freeze_backbone(model)
    assert model.tokenizer is not None
    collator = PairCollator(model.tokenizer, max_length=16, prompt_config=PromptConfig())
    dataloader: DataLoader[PairExample] = DataLoader(
        ExampleDataset(build_smoke_pairs()),
        batch_size=1,
        collate_fn=collator,
    )
    trainer = EmbeddingTrainer(
        model,
        SymmetricInfoNCELoss(),
        TrainingConfig(
            experiment_name="trainer_grad_accum_steps",
            epochs=1,
            grad_accum_steps=2,
            micro_batch_size=1,
            run_root=str(tmp_path),
            save_every_steps=50,
            eval_every_steps=50,
        ),
    )
    summary = trainer.train(dataloader)
    assert summary.global_step == 2


def test_unknown_length_dataset_requires_explicit_max_update_steps(tmp_path: Path) -> None:
    model = build_toy_embedding_model(projection_dim=8)
    freeze_backbone(model)
    assert model.tokenizer is not None
    collator = PairCollator(model.tokenizer, max_length=16, prompt_config=PromptConfig())
    dataloader: DataLoader[PairExample] = DataLoader(
        UnknownLengthPairDataset(),
        batch_size=2,
        collate_fn=collator,
    )
    trainer = EmbeddingTrainer(
        model,
        SymmetricInfoNCELoss(),
        TrainingConfig(
            experiment_name="trainer_unknown_length_requires_budget",
            epochs=1,
            run_root=str(tmp_path),
        ),
    )
    with pytest.raises(RuntimeError, match="max_update_steps"):
        trainer.train(dataloader)


def test_unknown_length_dataset_uses_explicit_max_update_steps(tmp_path: Path) -> None:
    model = build_toy_embedding_model(projection_dim=8)
    freeze_backbone(model)
    assert model.tokenizer is not None
    collator = PairCollator(model.tokenizer, max_length=16, prompt_config=PromptConfig())
    dataloader: DataLoader[PairExample] = DataLoader(
        UnknownLengthPairDataset(),
        batch_size=2,
        collate_fn=collator,
    )
    trainer = EmbeddingTrainer(
        model,
        SymmetricInfoNCELoss(),
        TrainingConfig(
            experiment_name="trainer_unknown_length_with_budget",
            epochs=3,
            grad_accum_steps=2,
            max_update_steps=3,
            run_root=str(tmp_path),
            save_every_steps=50,
            eval_every_steps=50,
        ),
    )
    summary = trainer.train(dataloader)
    assert summary.global_step == 3
