from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bitnet_embed.data.collators import PairCollator
from bitnet_embed.data.loaders import ExampleDataset, build_smoke_pairs
from bitnet_embed.data.schemas import PairExample
from bitnet_embed.losses.contrastive import SymmetricInfoNCELoss
from bitnet_embed.modeling.prompts import PromptConfig
from bitnet_embed.modeling.smoke import build_toy_embedding_model
from bitnet_embed.train.factory import freeze_backbone
from bitnet_embed.train.trainer import EmbeddingTrainer, TrainingConfig


def _build_dataloader(batch_size: int = 2) -> DataLoader[PairExample]:
    model = build_toy_embedding_model(projection_dim=8)
    assert model.tokenizer is not None
    collator = PairCollator(model.tokenizer, max_length=16, prompt_config=PromptConfig())
    dataset: ExampleDataset[PairExample] = ExampleDataset(build_smoke_pairs())
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collator)


def _build_frozen_model(state_dict: dict[str, torch.Tensor] | None = None) -> torch.nn.Module:
    model = build_toy_embedding_model(projection_dim=8)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    freeze_backbone(model)
    return model


def test_resume_from_checkpoint_matches_uninterrupted_training(tmp_path: Path) -> None:
    baseline_model = _build_frozen_model()
    baseline_initial_state = {
        name: parameter.detach().clone() for name, parameter in baseline_model.state_dict().items()
    }

    full_model = _build_frozen_model(baseline_initial_state)
    full_trainer = EmbeddingTrainer(
        full_model,
        SymmetricInfoNCELoss(),
        TrainingConfig(
            experiment_name="resume_full",
            epochs=3,
            micro_batch_size=4,
            max_update_steps=3,
            save_every_steps=3,
            eval_every_steps=50,
            run_root=str(tmp_path / "full"),
        ),
    )
    full_summary = full_trainer.train(_build_dataloader(batch_size=4))
    assert full_summary.global_step == 3

    phase_one_model = _build_frozen_model(baseline_initial_state)
    phase_one_trainer = EmbeddingTrainer(
        phase_one_model,
        SymmetricInfoNCELoss(),
        TrainingConfig(
            experiment_name="resume_partial",
            epochs=1,
            micro_batch_size=4,
            max_update_steps=3,
            save_every_steps=1,
            eval_every_steps=50,
            run_root=str(tmp_path / "resume"),
        ),
    )
    phase_one_summary = phase_one_trainer.train(_build_dataloader(batch_size=4))
    assert phase_one_summary.checkpoint_dir is not None
    assert phase_one_summary.global_step == 1

    resumed_model = _build_frozen_model(baseline_initial_state)
    resumed_trainer = EmbeddingTrainer(
        resumed_model,
        SymmetricInfoNCELoss(),
        TrainingConfig(
            experiment_name="resume_partial",
            epochs=3,
            micro_batch_size=4,
            max_update_steps=3,
            save_every_steps=3,
            eval_every_steps=50,
            run_root=str(tmp_path / "resume"),
            resume_from_checkpoint=phase_one_summary.checkpoint_dir,
        ),
    )
    resumed_summary = resumed_trainer.train(_build_dataloader(batch_size=4))
    assert resumed_summary.global_step == 3

    full_projection = full_trainer.accelerator.unwrap_model(
        full_trainer.model
    ).projection.state_dict()
    resumed_projection = resumed_trainer.accelerator.unwrap_model(
        resumed_trainer.model
    ).projection.state_dict()
    for name, tensor in full_projection.items():
        assert torch.allclose(tensor, resumed_projection[name])
