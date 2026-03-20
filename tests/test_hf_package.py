from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bitnet_embed.data.collators import PairCollator
from bitnet_embed.data.loaders import ExampleDataset, build_smoke_pairs
from bitnet_embed.data.schemas import PairExample
from bitnet_embed.export.hf_package import export_hf_package, load_hf_package
from bitnet_embed.losses.contrastive import SymmetricInfoNCELoss
from bitnet_embed.modeling.model import EncodeConfig
from bitnet_embed.modeling.prompts import PromptConfig
from bitnet_embed.modeling.smoke import build_toy_embedding_model
from bitnet_embed.train.factory import freeze_backbone
from bitnet_embed.train.trainer import EmbeddingTrainer, TrainingConfig


def test_export_and_load_hf_style_package(tmp_path: Path) -> None:
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
            experiment_name="hf_package_smoke",
            epochs=1,
            micro_batch_size=2,
            save_every_steps=1,
            eval_every_steps=10,
            run_root=str(tmp_path),
        ),
    )
    summary = trainer.train(
        dataloader,
        config_snapshot={
            "experiment_name": "hf_package_smoke",
            "model": {"backend": "toy", "projection_dim": 8, "normalize": True},
        },
    )
    assert summary.checkpoint_dir is not None
    output_dir = tmp_path / "exported-package"
    manifest = export_hf_package(summary.checkpoint_dir, output_dir, package_name="toy-export")
    assert manifest["package_name"] == "toy-export"
    assert (output_dir / "config.json").exists()
    assert (output_dir / "pytorch_model.bin").exists()
    restored = load_hf_package(output_dir)
    original_embeddings = model.encode(
        ["packaged model"], EncodeConfig(batch_size=1, truncate_dim=8)
    )
    restored_embeddings = restored.encode(
        ["packaged model"], EncodeConfig(batch_size=1, truncate_dim=8)
    )
    assert torch.allclose(original_embeddings, restored_embeddings)
