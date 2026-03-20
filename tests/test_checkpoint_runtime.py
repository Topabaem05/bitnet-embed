from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bitnet_embed.data.collators import PairCollator
from bitnet_embed.data.loaders import ExampleDataset, build_smoke_pairs
from bitnet_embed.data.schemas import PairExample
from bitnet_embed.losses.contrastive import SymmetricInfoNCELoss
from bitnet_embed.modeling.model import EncodeConfig
from bitnet_embed.modeling.prompts import PromptConfig
from bitnet_embed.modeling.smoke import build_toy_embedding_model
from bitnet_embed.serve.config import ServiceConfig
from bitnet_embed.serve.runtime import build_default_runtime
from bitnet_embed.serve.schemas import EmbeddingRequest
from bitnet_embed.train.factory import freeze_backbone, load_model_checkpoint
from bitnet_embed.train.trainer import EmbeddingTrainer, TrainingConfig


def test_load_model_checkpoint_restores_saved_weights(tmp_path: Path) -> None:
    model = build_toy_embedding_model(projection_dim=8)
    freeze_backbone(model)
    assert model.tokenizer is not None
    collator = PairCollator(model.tokenizer, max_length=16, prompt_config=PromptConfig())
    dataset: ExampleDataset[PairExample] = ExampleDataset(build_smoke_pairs())
    dataloader: DataLoader[PairExample] = DataLoader(dataset, batch_size=2, collate_fn=collator)
    trainer = EmbeddingTrainer(
        model,
        SymmetricInfoNCELoss(),
        TrainingConfig(
            experiment_name="checkpoint_reload_smoke",
            epochs=1,
            micro_batch_size=2,
            save_every_steps=1,
            eval_every_steps=10,
            run_root=str(tmp_path),
        ),
    )
    summary = trainer.train(
        dataloader,
        config_snapshot={"model": {"backend": "toy", "projection_dim": 8, "normalize": True}},
    )
    assert summary.checkpoint_dir is not None
    restored_model = load_model_checkpoint(summary.checkpoint_dir)
    original = model.encode(["reload me"], EncodeConfig(batch_size=1, truncate_dim=8))
    restored = restored_model.encode(["reload me"], EncodeConfig(batch_size=1, truncate_dim=8))
    assert original.shape == restored.shape
    assert torch.allclose(original, restored)


def test_runtime_can_boot_from_checkpoint_backend(tmp_path: Path) -> None:
    model = build_toy_embedding_model(projection_dim=8)
    freeze_backbone(model)
    assert model.tokenizer is not None
    collator = PairCollator(model.tokenizer, max_length=16, prompt_config=PromptConfig())
    dataset: ExampleDataset[PairExample] = ExampleDataset(build_smoke_pairs())
    dataloader: DataLoader[PairExample] = DataLoader(dataset, batch_size=2, collate_fn=collator)
    trainer = EmbeddingTrainer(
        model,
        SymmetricInfoNCELoss(),
        TrainingConfig(
            experiment_name="checkpoint_service_smoke",
            epochs=1,
            micro_batch_size=2,
            save_every_steps=1,
            eval_every_steps=10,
            run_root=str(tmp_path),
        ),
    )
    summary = trainer.train(
        dataloader,
        config_snapshot={"model": {"backend": "toy", "projection_dim": 8, "normalize": True}},
    )
    assert summary.checkpoint_dir is not None
    runtime = build_default_runtime(
        ServiceConfig(
            model_name="checkpoint-smoke",
            backend="checkpoint",
            checkpoint_dir=summary.checkpoint_dir,
            truncate_dim_default=8,
        )
    )
    response = runtime.embed(EmbeddingRequest(input=["checkpoint runtime"], task="document"))
    assert response.model == "checkpoint-smoke"
    assert len(response.data[0].embedding) == 8
